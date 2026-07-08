"""controller: collision-aware motion control for the Frank robot.

Provides two classes built on top of :class:`~frank_sapien.agents.frank.Frank`
and `mplib <https://github.com/haosulab/MPlib>`_:

* :class:`CollisionChecker` -- queries whether an arbitrary joint configuration
  is in collision. By default it checks the robot against the scene obstacles
  (the table, and any others added) and against itself (self-collision, with
  adjacent pairs disabled via the SRDF). The robot-vs-floor pair is *excluded*
  (the floor is never added as an obstacle), matching the SAPIEN scene setup.

* :class:`FrankController` -- a plan-then-execute controller. It moves an
  end-effector to / by a world pose, sends a joint to / by an angle, or returns
  home, for the ``"left"``, ``"right"``, or ``"both"`` arms. Each move is a
  collision-free plan (mplib RRTConnect) executed under bounded end-effector
  speed; if planning fails it is retried a few times and otherwise rejected.

Design notes discovered while integrating mplib (see also
:mod:`~frank_sapien.agents.planning_assets`):

* IK is done with SAPIEN's Pinocchio (``Frank.compute_ik``) because mplib 0.2.1's
  pose IK is unreliable for this robot; mplib is used only for collision-checked
  **joint-space** planning (``plan_qpos``).
* SAPIEN's Pinocchio reports poses in the robot **root** frame, so world targets
  are converted with the base pose before IK.
* The lift joint is never commanded here -- it is held by ``Frank``'s lock.
"""

import math
import struct
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import sapien

from frank_sapien.agents.frank import Frank, ASSETS_PATH
from frank_sapien.agents.planning_assets import get_planning_assets

ARMS = ("left", "right")
_TABLE_STL = ASSETS_PATH / "world" / "table" / "table.stl"


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def _read_stl_vertices(path: Path) -> np.ndarray:
    """Read a binary STL and return its triangle vertices, ``(3*ntri, 3)``."""
    with open(path, "rb") as fh:
        fh.read(80)
        ntri = struct.unpack("<I", fh.read(4))[0]
        v = np.zeros((ntri * 3, 3), np.float32)
        for i in range(ntri):
            fh.read(12)  # normal
            v[i * 3 : i * 3 + 3] = np.frombuffer(fh.read(36), np.float32).reshape(3, 3)
            fh.read(2)  # attribute byte count
    return v.astype(np.float64)


def _transform_points(points: np.ndarray, pose: sapien.Pose) -> np.ndarray:
    """Transform local ``points`` (N,3) into world by ``pose`` (SAPIEN Pose)."""
    from scipy.spatial.transform import Rotation as R

    w, x, y, z = pose.q
    rot = R.from_quat([x, y, z, w])
    return rot.apply(points) + np.asarray(pose.p)


def _obj_pose(obj) -> sapien.Pose:
    """Pose of a SAPIEN articulation or actor/entity."""
    if hasattr(obj, "get_root_pose"):
        return obj.get_root_pose()
    return obj.get_pose()


def _pose_to_mplib(pose: sapien.Pose):
    """SAPIEN Pose -> mplib Pose (both use (p, q=wxyz))."""
    import mplib

    return mplib.Pose(p=list(pose.p), q=list(pose.q))


# ---------------------------------------------------------------------------
# mplib planner setup (shared by controller and collision checker)
# ---------------------------------------------------------------------------
def table_point_cloud(world, n_points: int = 8000, seed: int = 0) -> np.ndarray:
    """Sample a world-frame point cloud of the table from its mesh.

    Args:
        world: a :class:`~frank_sapien.scene.table_top.TableTopScene`.
        n_points: number of points to subsample.
        seed: RNG seed for the subsample.

    Returns:
        (n_points, 3) float64 array of world points.
    """
    verts = _read_stl_vertices(_TABLE_STL)
    world_verts = _transform_points(verts, _obj_pose(world.table[0]))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(world_verts), min(n_points, len(world_verts)), replace=False)
    return world_verts[idx]


def _is_arm_link(name: str) -> bool:
    """A moving arm/gripper link (not the fixed trunk or arm mount base)."""
    return "kinova_arm" in name and "base_link" not in name


class _ArmPlanner:
    """A configured mplib planner for one arm, sharing obstacle + ACM setup.

    Holds the joint-index maps between SAPIEN order and mplib order so callers
    can convert freely. The floor is intentionally absent from the world, and
    trunk links are excluded from the point-cloud collision (they are rigidly
    inside the table, mirroring the SAPIEN collision filtering).
    """

    def __init__(
        self,
        frank: Frank,
        arm: str,
        table_mesh: Path,
        table_pose,
        extra_points: Optional[np.ndarray] = None,
        pcd_resolution: float = 0.02,
    ):
        import mplib

        urdf, srdf = get_planning_assets()
        self.planner = mplib.Planner(
            urdf=urdf,
            srdf=srdf,
            move_group=Frank.EE_LINKS[arm],
        )
        self.planner.set_base_pose(_pose_to_mplib(frank.robot.get_root_pose()))

        # SAPIEN<->mplib joint index maps.
        self.user_joint_names = list(self.planner.user_joint_names)
        self._s2m = [frank._joint_index[n] for n in self.user_joint_names]  # mplib->sapien
        self.move_group_sapien = [
            frank._joint_index[self.user_joint_names[i]]
            for i in self.planner.move_group_joint_indices
        ]
        self.move_group_names = [
            self.user_joint_names[i] for i in self.planner.move_group_joint_indices
        ]

        self._add_table_mesh(table_mesh, table_pose)
        if extra_points is not None and len(extra_points):
            self.set_point_cloud(extra_points, pcd_resolution)

    def _ignore_trunk(self, obstacle_name: str):
        """Exclude the fixed trunk links from colliding with ``obstacle_name``
        (mirrors the SAPIEN trunk<->table collision filtering)."""
        acm = self.planner.planning_world.get_allowed_collision_matrix()
        for link in self.planner.link_name_2_idx:
            if not _is_arm_link(link):
                acm.set_entry(link, obstacle_name, True)

    def _add_table_mesh(self, mesh_path: Path, pose):
        """Add the table as an exact triangle-mesh obstacle named ``"table"``."""
        import mplib
        import mplib.collision_detection.fcl as fcl

        bvh = fcl.load_mesh_as_BVH(str(mesh_path), scale=np.array([1.0, 1.0, 1.0]))
        obj = fcl.FCLObject("table", pose, [fcl.CollisionObject(bvh)], [mplib.Pose()])
        self.planner.planning_world.add_object(obj)
        self._ignore_trunk("table")

    def set_point_cloud(self, points: np.ndarray, resolution: float):
        """Set an extra environment point cloud (e.g. objects on the table)."""
        self.planner.update_point_cloud(np.asarray(points, np.float64), resolution=resolution)
        self._ignore_trunk("scene_pcd")

    def full_mplib_qpos(self, frank: Frank) -> np.ndarray:
        """Current full qpos reordered into mplib's joint order."""
        q = frank.robot.get_qpos()
        return np.array([q[i] for i in self._s2m])


# ---------------------------------------------------------------------------
# collision checker
# ---------------------------------------------------------------------------
class CollisionChecker:
    """Checks whether a configuration collides (self + environment).

    Excludes the robot-vs-floor pair (the floor is never an obstacle) and the
    fixed trunk vs. the table point cloud; everything else -- both arms vs. the
    table/objects, and self-collision between the arms and body -- is checked.
    """

    def __init__(self, arm_planner: "_ArmPlanner", frank: Frank):
        self._ap = arm_planner
        self._frank = frank

    def collisions(self, qpos: Optional[np.ndarray] = None) -> List[Tuple[str, str]]:
        """Return colliding link/object name pairs for ``qpos``.

        Args:
            qpos: full SAPIEN-order qpos to test; defaults to the robot's current
                configuration.

        Returns:
            List of ``(name1, name2)`` pairs (empty if collision-free).
        """
        if qpos is None:
            full = self._ap.full_mplib_qpos(self._frank)
        else:
            full = np.array([qpos[i] for i in self._ap._s2m])
        results = self._ap.planner.check_for_self_collision(full)
        results += self._ap.planner.check_for_env_collision(full)
        return [(r.link_name1, r.link_name2) for r in results]

    def in_collision(self, qpos: Optional[np.ndarray] = None) -> bool:
        """True if ``qpos`` (or the current config) is in collision."""
        return len(self.collisions(qpos)) > 0


# ---------------------------------------------------------------------------
# controller
# ---------------------------------------------------------------------------
class FrankController:
    """Plan-then-execute motion controller for the Frank robot.

    Args:
        world: a :class:`~frank_sapien.scene.table_top.TableTopScene`.
        obstacle_points: optional extra world-frame obstacle point cloud (N,3);
            the table is added automatically.
        pcd_resolution: collision radius (m) around each obstacle point.
        default_lin_vel / default_ang_vel: default EE speed caps (m/s, rad/s).
        default_pos_tol / default_rot_tol: default final-error tolerances (m, rad).
        max_replan: planning attempts before a move is rejected.
        planning_time: per-attempt RRT time budget (s).
    """

    def __init__(
        self,
        world,
        obstacle_points: Optional[np.ndarray] = None,
        pcd_resolution: float = 0.02,
        default_lin_vel: float = 0.25,
        default_ang_vel: float = 1.0,
        default_pos_tol: float = 5e-3,
        default_rot_tol: float = 0.05,
        max_replan: int = 5,
        planning_time: float = 5.0,
    ):
        self.world = world
        self.scene = world.scene
        self.frank: Frank = world.frank
        self.sim_dt = self.scene.get_timestep()
        self.default_lin_vel = default_lin_vel
        self.default_ang_vel = default_ang_vel
        self.default_pos_tol = default_pos_tol
        self.default_rot_tol = default_rot_tol
        self.max_replan = max_replan
        self.planning_time = planning_time
        #: Optional no-arg callable invoked after every simulation step during a
        #: move (set it to e.g. a viewer render so motions are visible live).
        self.step_hook = None

        # Obstacles: the table as an exact mesh, plus any extra point cloud
        # (e.g. objects placed on the table). The floor is never added, so
        # robot<->floor is excluded by construction.
        table_pose = _pose_to_mplib(_obj_pose(world.table[0]))
        self._planners: Dict[str, _ArmPlanner] = {
            arm: _ArmPlanner(
                self.frank, arm, _TABLE_STL, table_pose,
                extra_points=obstacle_points, pcd_resolution=pcd_resolution,
            )
            for arm in ARMS
        }
        #: Whole-robot collision checker (shares the left planner's world).
        self.collision = CollisionChecker(self._planners["left"], self.frank)

    # -- frames / FK ------------------------------------------------------
    def ee_pose(self, arm: str) -> sapien.Pose:
        """Current world pose of ``arm``'s end-effector."""
        for l in self.frank.robot.get_links():
            if l.get_name() == Frank.EE_LINKS[arm]:
                return l.get_entity().get_pose()
        raise KeyError(arm)

    def _fk_world(self, arm: str, qpos_full: np.ndarray) -> sapien.Pose:
        """World EE pose of ``arm`` at a full SAPIEN qpos (via Pinocchio + base)."""
        self.frank._pinocchio.compute_forward_kinematics(qpos_full)
        root_pose = self.frank._pinocchio.get_link_pose(
            self.frank._link_index[Frank.EE_LINKS[arm]]
        )
        return self.frank.robot.get_root_pose() * root_pose

    # -- IK ---------------------------------------------------------------
    def _ik(self, arm: str, world_pose: sapien.Pose) -> Tuple[np.ndarray, float]:
        """IK a world EE pose for ``arm`` -> (full SAPIEN qpos goal, pos error)."""
        root_pose = self.frank.robot.get_root_pose().inv() * world_pose
        qpos, _, err = self.frank.compute_ik(root_pose, arm=arm)
        return qpos, err

    # -- planning ---------------------------------------------------------
    def _plan(
        self, arm: str, goal_full: np.ndarray, context_full: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Plan a collision-free joint path for ``arm`` to ``goal_full`` (full
        SAPIEN qpos). ``context_full`` (default: current) is the full config used
        as the planning start -- its non-move-group joints (e.g. the other arm)
        set the collision context, so ``"both"`` moves can avoid each other's
        goals. Retries up to ``max_replan`` times; returns the move-group waypoint
        array ``(N, len(move_group))`` or ``None`` if all attempts fail.
        """
        ap = self._planners[arm]
        goal_mplib = np.array([goal_full[i] for i in ap._s2m])
        # Reject up front if the goal configuration itself is in collision --
        # no collision-free path can end there.
        if (ap.planner.check_for_self_collision(goal_mplib)
                or ap.planner.check_for_env_collision(goal_mplib)):
            print(f"[controller] move ({arm}) rejected: goal configuration is in collision")
            return None
        if context_full is None:
            context_full = self.frank.robot.get_qpos()
        start_mplib = np.array([context_full[i] for i in ap._s2m])

        # Already at the goal: nothing to plan (a zero-length path breaks TOPP).
        mg = list(ap.planner.move_group_joint_indices)
        if np.linalg.norm(goal_mplib[mg] - start_mplib[mg]) < 1e-3:
            return goal_mplib[mg][None, :]

        for _ in range(self.max_replan):
            try:
                res = ap.planner.plan_qpos(
                    [goal_mplib], start_mplib,
                    time_step=self.sim_dt, planning_time=self.planning_time,
                )
            except RuntimeError:
                continue  # e.g. TOPP failed to parameterize this path; retry
            if res["status"] == "Success":
                return res["position"]
        return None

    # -- execution --------------------------------------------------------
    def _execute(
        self,
        arm_trajs: Dict[str, np.ndarray],
        lin_vel: float,
        ang_vel: float,
        settle_steps: int = 150,
    ):
        """Execute one or two move-group trajectories with bounded EE speed.

        Waypoints of each arm are stepped in lockstep (the shorter is padded by
        repeating its last waypoint). The dwell per waypoint is set so neither
        end-effector exceeds ``lin_vel`` / ``ang_vel``.
        """
        arms = list(arm_trajs)
        n = max(len(t) for t in arm_trajs.values())
        # pad each trajectory to n waypoints
        padded = {a: np.vstack([t, np.repeat(t[-1:], n - len(t), axis=0)]) if len(t) < n else t
                  for a, t in arm_trajs.items()}

        prev_world = {a: self.ee_pose(a) for a in arms}
        for k in range(n):
            # command each arm's arm-joints (skip the lift: held by Frank's lock)
            dwell = 1
            for a in arms:
                ap = self._planners[a]
                wp = padded[a][k]
                for name, jidx, val in zip(ap.move_group_names, ap.move_group_sapien, wp):
                    if name in Frank.LOCKED_JOINTS:
                        continue
                    self.frank._active_joints[jidx].set_drive_target(val)
                # dwell from this arm's EE displacement (bound speed)
                full = self.frank.robot.get_qpos().copy()
                for jidx, val in zip(ap.move_group_sapien, wp):
                    full[jidx] = val
                wpose = self._fk_world(a, full)
                lin = np.linalg.norm(np.asarray(wpose.p) - np.asarray(prev_world[a].p))
                ang = _quat_angle(wpose.q, prev_world[a].q)
                dwell = max(dwell, math.ceil(max(lin / lin_vel, ang / ang_vel) / self.sim_dt))
                prev_world[a] = wpose
            for _ in range(dwell):
                self._sim_step()

        for _ in range(settle_steps):
            self._sim_step()

    def _sim_step(self):
        """One physics step with gravity compensation and the optional render hook."""
        self.frank.apply_gravity_compensation()
        self.scene.step()
        if self.step_hook is not None:
            self.step_hook()

    # -- public move API --------------------------------------------------
    def move_ee_pose(
        self,
        pose: Union[sapien.Pose, Dict[str, sapien.Pose]],
        arm: str = "left",
        lin_vel: Optional[float] = None,
        ang_vel: Optional[float] = None,
        pos_tol: Optional[float] = None,
        rot_tol: Optional[float] = None,
    ) -> bool:
        """Move end-effector(s) to a world pose.

        Args:
            pose: for ``arm="left"``/``"right"`` a single ``sapien.Pose``; for
                ``arm="both"`` a dict ``{"left": pose, "right": pose}``.
            arm: ``"left"``, ``"right"`` or ``"both"``.
            lin_vel / ang_vel: EE speed caps for this move (defaults if None).
            pos_tol / rot_tol: acceptance tolerance for this move.

        Returns:
            True if every arm reached its target within tolerance; False if a
            plan was rejected (after retries) or the tolerance was not met.
        """
        targets = self._normalize_targets(pose, arm)
        lin_vel = lin_vel or self.default_lin_vel
        ang_vel = ang_vel or self.default_ang_vel
        ik_tol = max(pos_tol or self.default_pos_tol, 1e-3)

        goals: Dict[str, np.ndarray] = {}
        for a, wp in targets.items():
            qpos, err = self._ik(a, wp)
            if err > ik_tol:
                print(f"[controller] move_ee_pose({a}) rejected: target unreachable "
                      f"(IK error {err*1e3:.1f}mm > {ik_tol*1e3:.1f}mm)")
                return False
            goals[a] = qpos
        trajs = self._plan_arms(goals)
        if trajs is None:
            return False
        self._execute(trajs, lin_vel, ang_vel)
        return self._check_reached(targets, pos_tol, rot_tol)

    def _plan_arms(self, goals: Dict[str, np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
        """Plan every arm in ``goals`` (full-qpos targets). For >1 arm, each
        subsequent arm is planned with the already-planned arms held at their
        goals (collision context), so they avoid each other. Returns per-arm
        move-group trajectories, or None if any arm's plan was rejected."""
        trajs: Dict[str, np.ndarray] = {}
        context = self.frank.robot.get_qpos().copy()
        for a in goals:
            traj = self._plan(a, goals[a], context_full=context)
            if traj is None:
                print(f"[controller] move ({a}) rejected: no collision-free plan "
                      f"after {self.max_replan} attempts")
                return None
            trajs[a] = traj
            # hold this arm at its goal for the next arm's collision context
            for jidx in self._planners[a].move_group_sapien:
                context[jidx] = goals[a][jidx]
        return trajs

    def move_ee_delta(
        self, delta: Union[Sequence[float], Dict[str, Sequence[float]]],
        arm: str = "left", **kwargs,
    ) -> bool:
        """Move end-effector(s) by a world-frame delta ``(dx,dy,dz,droll,dpitch,dyaw)``.

        ``delta`` is a 6-vector for one arm, or a dict of them for ``"both"``.
        Rotation deltas are applied in the world frame about the current pose.
        """
        from scipy.spatial.transform import Rotation as R

        deltas = delta if arm == "both" else {arm: delta}
        targets: Dict[str, sapien.Pose] = {}
        for a, d in deltas.items():
            d = np.asarray(d, float)
            cur = self.ee_pose(a)
            w, x, y, z = cur.q
            new_rot = R.from_euler("xyz", d[3:6]) * R.from_quat([x, y, z, w])
            q = new_rot.as_quat()  # xyzw
            targets[a] = sapien.Pose(p=np.asarray(cur.p) + d[:3], q=[q[3], q[0], q[1], q[2]])
        return self.move_ee_pose(targets if arm == "both" else targets[arm], arm=arm, **kwargs)

    def move_home(self, config: str = "upright", **kwargs) -> bool:
        """Move both arms to a named home configuration (``"upright"``/``"rest"``)."""
        goal_full = self.frank._to_qpos_array(Frank.NAMED_CONFIGS[config])
        return self.move_joints_to(goal_full, **kwargs)

    def move_joint(self, joint_name: str, angle: float, **kwargs) -> bool:
        """Move a single joint to an absolute ``angle`` (collision-checked)."""
        if joint_name in Frank.LOCKED_JOINTS:
            print(f"[controller] '{joint_name}' is locked; refusing to move it.")
            return False
        kwargs.pop("arm", None)  # arm is inferred from the joint name
        goal_full = self.frank.robot.get_qpos().copy()
        goal_full[self.frank._joint_index[joint_name]] = angle
        return self.move_joints_to(goal_full, arm=_arm_of_joint(joint_name), **kwargs)

    def move_joint_delta(self, joint_name: str, delta: float, **kwargs) -> bool:
        """Move a single joint by ``delta`` radians (collision-checked)."""
        cur = self.frank.robot.get_qpos()[self.frank._joint_index[joint_name]]
        return self.move_joint(joint_name, cur + delta, **kwargs)

    def move_joints_to(
        self, goal_full: np.ndarray, arm: str = "both",
        lin_vel: Optional[float] = None, ang_vel: Optional[float] = None,
    ) -> bool:
        """Plan+execute to a full-qpos joint goal for the given arm(s)."""
        lin_vel = lin_vel or self.default_lin_vel
        ang_vel = ang_vel or self.default_ang_vel
        arms = ARMS if arm == "both" else (arm,)
        trajs = self._plan_arms({a: goal_full for a in arms})
        if trajs is None:
            return False
        self._execute(trajs, lin_vel, ang_vel)
        return True

    # -- internals --------------------------------------------------------
    def _normalize_targets(self, pose, arm) -> Dict[str, sapien.Pose]:
        if arm == "both":
            if not isinstance(pose, dict) or set(pose) != set(ARMS):
                raise ValueError('arm="both" needs pose={"left": Pose, "right": Pose}')
            return dict(pose)
        if arm not in ARMS:
            raise ValueError(f"arm must be 'left', 'right' or 'both', got {arm!r}")
        return {arm: pose}

    def _check_reached(self, targets, pos_tol, rot_tol) -> bool:
        pos_tol = pos_tol or self.default_pos_tol
        rot_tol = rot_tol or self.default_rot_tol
        ok = True
        for a, wp in targets.items():
            cur = self.ee_pose(a)
            perr = float(np.linalg.norm(np.asarray(cur.p) - np.asarray(wp.p)))
            rerr = _quat_angle(cur.q, wp.q)
            if perr > pos_tol or rerr > rot_tol:
                ok = False
                print(f"[controller] {a} final error pos={perr*1e3:.1f}mm rot={math.degrees(rerr):.1f}deg "
                      f"(tol {pos_tol*1e3:.1f}mm/{math.degrees(rot_tol):.1f}deg)")
        return ok


def _arm_of_joint(joint_name: str) -> str:
    return "right" if joint_name.startswith("right") else "left"


def _quat_angle(qa, qb) -> float:
    """Angle (rad) between two wxyz quaternions."""
    qa = np.asarray(qa) / np.linalg.norm(qa)
    qb = np.asarray(qb) / np.linalg.norm(qb)
    return 2.0 * math.acos(min(1.0, abs(float(np.dot(qa, qb)))))
