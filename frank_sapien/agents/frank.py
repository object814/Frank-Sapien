"""frank: loads and controls the custom Frank robot in a SAPIEN scene.

Wraps the Frank articulation (a dual Kinova Gen3 7-DOF arm on an Ewellix lift
with a pan/tilt head and Robotiq grippers) behind a small :class:`Frank` class
that handles the three things every caller needs:

* **Loading** the URDF and placing the base in the scene.
* **Initialising a pose** -- either directly from joint angles, or from a
  desired end-effector pose solved with inverse kinematics. This lets you spawn
  the robot in a sensible, collision-free configuration instead of the default
  all-zeros pose, which buries the arms inside the table.
* **Gravity compensation** via the controller (not by disabling gravity): each
  simulation step the passive forces (gravity + Coriolis/centrifugal) are
  computed and fed back as joint torques, and PD drives hold the target pose so
  the robot stays put instead of collapsing.

Typical use::

    frank = Frank(scene)                 # loads at the default ready pose
    frank.set_init_qpos({"left_kinova_arm_joint_2": 0.4})   # or joint angles
    frank.set_init_ee_pose(target_pose, arm="left")         # or an EE pose

    while running:
        frank.apply_gravity_compensation()   # call once per scene.step()
        scene.step()
"""

import numpy as np
import sapien
from typing import Optional, Mapping, Union
from pathlib import Path

# Path to environment assets
ASSETS_PATH = Path(__file__).parent.parent / "assets"


class Frank:
    """The custom Frank robot loaded into a SAPIEN scene.

    Attributes:
        robot (sapien.physx.PhysxArticulation): The underlying articulation.
        init_qpos (np.ndarray): The configuration the robot is initialised at
            and that the hold-drives target.
    """

    # --- Joint / link naming -------------------------------------------------
    LIFT_JOINT = "ewellix_lift_top_joint"
    PTU_JOINTS = ("ptu_pan", "ptu_tilt")
    #: end-effector links used as the IK target for each arm.
    EE_LINKS = {
        "left": "left_kinova_arm_end_effector_link",
        "right": "right_kinova_arm_end_effector_link",
    }

    #: Joints pinned to a fixed value: they are forced to this value on every
    #: pose update and excluded from IK, so nothing (init pose, IK, drives) ever
    #: moves them. The stiff hold-drive keeps them there during simulation.
    #: 0.40 matches the MuJoCo bimanual-suite home (config.yaml `robot:`).
    LOCKED_JOINTS = {
        "ewellix_lift_top_joint": 0.40,
    }

    #: Named initial configurations, each taken verbatim from the MuJoCo
    #: reference so SAPIEN initialises Frank identically. Select one by name via
    #: the ``init_config`` argument (see :meth:`__init__`). Anything not listed
    #: defaults to 0 (open grippers); locked joints (see :attr:`LOCKED_JOINTS`)
    #: override whatever is requested. Both share lift=0.40, ptu_pan=0,
    #: ptu_tilt=-1.0 and differ only in the 14 arm-joint angles.
    #:
    #: - ``"upright"``: arms raised symmetric with grippers pointing down over
    #:   the table (ramp assembly home, ``configs/config.yaml`` ``robot:``).
    #: - ``"rest"``: arms spread out resting just above the tabletop (cube-suite
    #:   ``mjc/cubes.py`` ``CALIBRATION_POSE``).
    NAMED_CONFIGS = {
        "upright": {
            "ptu_pan": 0.0,
            "ptu_tilt": -1.0,
            "left_kinova_arm_joint_1": 1.7559,
            "left_kinova_arm_joint_2": 1.0865,
            "left_kinova_arm_joint_3": -0.9366,
            "left_kinova_arm_joint_4": 1.8296,
            "left_kinova_arm_joint_5": -0.2667,
            "left_kinova_arm_joint_6": 1.7525,
            # 0.6911 (MuJoCo) + pi: spins the left gripper 180 deg about world Z.
            "left_kinova_arm_joint_7": 3.8327,
            "right_kinova_arm_joint_1": -1.8403,
            "right_kinova_arm_joint_2": 1.1124,
            "right_kinova_arm_joint_3": 0.9143,
            "right_kinova_arm_joint_4": 1.8747,
            "right_kinova_arm_joint_5": 0.1793,
            "right_kinova_arm_joint_6": 1.7609,
            "right_kinova_arm_joint_7": -0.7308,
        },
        "rest": {
            "ptu_pan": 0.0,
            "ptu_tilt": -1.0,
            "left_kinova_arm_joint_1": -1.60,
            "left_kinova_arm_joint_2": 0.55,
            "left_kinova_arm_joint_3": -1.63,
            "left_kinova_arm_joint_4": -1.63,
            "left_kinova_arm_joint_5": -0.50,
            "left_kinova_arm_joint_6": -1.57,
            "left_kinova_arm_joint_7": -0.08,
            "right_kinova_arm_joint_1": -1.60,
            "right_kinova_arm_joint_2": -0.55,
            "right_kinova_arm_joint_3": 1.63,
            "right_kinova_arm_joint_4": 1.63,
            "right_kinova_arm_joint_5": 0.50,
            "right_kinova_arm_joint_6": 1.57,
            "right_kinova_arm_joint_7": 0.08,
        },
    }

    #: Which of :attr:`NAMED_CONFIGS` is used when no ``init_config`` is given.
    DEFAULT_INIT_CONFIG = "upright"

    def __init__(
        self,
        scene: sapien.Scene,
        urdf_path: Optional[Path] = None,
        root_position=(0.0, 0.0, 0.0),
        root_quaternion=(1.0, 0.0, 0.0, 0.0),
        init_config: str = DEFAULT_INIT_CONFIG,
        init_qpos: Optional[Union[np.ndarray, Mapping[str, float]]] = None,
    ):
        """
        Load the Frank robot into ``scene`` and initialise it.

        Args:
            scene (sapien.Scene): Scene to add the robot to.
            urdf_path (Path): Path to the Frank URDF (defaults to the bundled one).
            root_position: Base position ``(x, y, z)`` in the world.
            root_quaternion: Base orientation ``(w, x, y, z)``.
            init_config (str): Name of the initial configuration to use, one of
                :attr:`NAMED_CONFIGS` (``"upright"`` or ``"rest"``). Ignored if
                ``init_qpos`` is given.
            init_qpos: Explicit initial configuration -- either a full ``(dof,)``
                array or a ``{joint_name: angle}`` mapping. Overrides
                ``init_config`` when provided.
        """
        if urdf_path is None:
            urdf_path = (ASSETS_PATH / "frank" / "frank.urdf").resolve()

        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = False

        self.robot = loader.load(urdf_file=str(urdf_path))
        self.robot.set_root_pose(
            sapien.Pose(p=list(root_position), q=list(root_quaternion))
        )

        # Cache joint/link lookups.
        self._active_joints = self.robot.get_active_joints()
        self.active_joint_names = [j.get_name() for j in self._active_joints]
        self._joint_index = {n: i for i, n in enumerate(self.active_joint_names)}
        self._link_index = {
            l.get_name(): i for i, l in enumerate(self.robot.get_links())
        }
        self._pinocchio = self.robot.create_pinocchio_model()

        # PD hold-drives so the robot actively holds its target pose. Gripper
        # joints form a closed linkage that is unstable under stiff independent
        # PD, so they get gentler, damping-dominated gains.
        self.setup_drives()

        # Place the robot at its initial configuration.
        if init_qpos is None:
            if init_config not in self.NAMED_CONFIGS:
                raise ValueError(
                    f"init_config must be one of {list(self.NAMED_CONFIGS)}, "
                    f"got {init_config!r}"
                )
            init_qpos = self.NAMED_CONFIGS[init_config]
        self.set_init_qpos(init_qpos)

    # -- convenience ----------------------------------------------------------
    @property
    def dof(self) -> int:
        return self.robot.dof

    def _is_gripper_joint(self, name: str) -> bool:
        return name not in (
            self.LIFT_JOINT,
            *self.PTU_JOINTS,
        ) and "kinova_arm_joint_" not in name

    # -- drives / gravity compensation ---------------------------------------
    def setup_drives(
        self,
        arm_stiffness: float = 2000.0,
        arm_damping: float = 200.0,
        gripper_stiffness: float = 20.0,
        gripper_damping: float = 5.0,
    ):
        """Configure PD drive gains on every active joint.

        The arm, lift and pan/tilt joints get stiff position control; the
        Robotiq gripper joints get soft, damping-dominated gains to keep their
        coupled four-bar linkage stable.
        """
        for joint in self._active_joints:
            if self._is_gripper_joint(joint.get_name()):
                joint.set_drive_property(gripper_stiffness, gripper_damping)
            else:
                joint.set_drive_property(arm_stiffness, arm_damping)

    def apply_gravity_compensation(self):
        """Feed passive-force compensation to the joints for this step.

        Computes the joint torques that cancel gravity and Coriolis/centrifugal
        effects and applies them via ``set_qf``. Call once immediately before
        every ``scene.step()``. This is gravity compensation by control, leaving
        the physical gravity in the scene intact (unlike ``disable_gravity``).
        """
        qf = self.robot.compute_passive_force(
            gravity=True, coriolis_and_centrifugal=True
        )
        self.robot.set_qf(qf)

    # -- pose initialisation --------------------------------------------------
    def _to_qpos_array(
        self, qpos: Union[np.ndarray, Mapping[str, float]]
    ) -> np.ndarray:
        """Normalise a full array or a {joint_name: angle} mapping into a
        full ``(dof,)`` array, filling unspecified joints from the current pose."""
        if isinstance(qpos, Mapping):
            full = np.array(self.robot.get_qpos(), dtype=float)
            for name, value in qpos.items():
                if name not in self._joint_index:
                    raise KeyError(
                        f"Unknown joint '{name}'. Valid joints: "
                        f"{self.active_joint_names}"
                    )
                full[self._joint_index[name]] = value
            self._apply_locked_joints(full)
            return full

        full = np.asarray(qpos, dtype=float).ravel()
        if full.shape[0] != self.dof:
            raise ValueError(
                f"qpos has length {full.shape[0]}, expected {self.dof}"
            )
        self._apply_locked_joints(full)
        return full

    def _apply_locked_joints(self, full: np.ndarray):
        """Force every locked joint in ``full`` to its pinned value, in place."""
        for name, value in self.LOCKED_JOINTS.items():
            if name in self._joint_index:
                full[self._joint_index[name]] = value

    def set_init_qpos(self, qpos: Union[np.ndarray, Mapping[str, float]]):
        """Teleport the robot to a configuration and hold it there.

        Args:
            qpos: Either a full ``(dof,)`` array or a ``{joint_name: angle}``
                mapping (unspecified joints keep their current value).

        This resets positions and velocities and points the PD drives at the
        new configuration, so combined with :meth:`apply_gravity_compensation`
        the robot holds the pose instead of collapsing.
        """
        full = self._to_qpos_array(qpos)
        self.init_qpos = full
        self.robot.set_qpos(full)
        self.robot.set_qvel(np.zeros(self.dof))
        for joint, target in zip(self._active_joints, full):
            joint.set_drive_target(target)

    def compute_ik(
        self,
        ee_pose: sapien.Pose,
        arm: str = "left",
        reference_qpos: Optional[np.ndarray] = None,
        max_iterations: int = 100,
    ):
        """Solve inverse kinematics for one arm's end-effector.

        Only the chosen arm's seven joints are allowed to move; the other arm,
        the head, and any locked joints (e.g. the lift) stay put.

        Args:
            ee_pose (sapien.Pose): Desired end-effector pose in the world frame.
            arm (str): ``"left"`` or ``"right"``.
            reference_qpos: Seed configuration (defaults to the current pose).
            max_iterations: IK iteration budget.

        Returns:
            (qpos, success, pos_error): full ``(dof,)`` solution, the solver's
            success flag, and the achieved end-effector position error (metres).
        """
        if arm not in self.EE_LINKS:
            raise ValueError(f"arm must be one of {list(self.EE_LINKS)}, got {arm!r}")

        link_idx = self._link_index[self.EE_LINKS[arm]]
        seed = (
            np.array(self.robot.get_qpos(), dtype=float)
            if reference_qpos is None
            else np.asarray(reference_qpos, dtype=float)
        )
        # Keep locked joints (e.g. the lift) pinned throughout IK.
        self._apply_locked_joints(seed)

        # Only the chosen arm's seven joints move; the other arm, the head, and
        # any locked joints stay put.
        mask = np.zeros(self.dof)
        for i, name in enumerate(self.active_joint_names):
            if f"{arm}_kinova_arm_joint_" in name and name not in self.LOCKED_JOINTS:
                mask[i] = 1

        qpos, success, _ = self._pinocchio.compute_inverse_kinematics(
            link_idx,
            ee_pose,
            initial_qpos=seed,
            active_qmask=mask,
            max_iterations=max_iterations,
        )

        # Report the achieved position error -- the boolean flag is strict on
        # full 6-DOF convergence, so callers often care more about this.
        self._pinocchio.compute_forward_kinematics(qpos)
        achieved = self._pinocchio.get_link_pose(link_idx)
        pos_error = float(np.linalg.norm(achieved.p - ee_pose.p))
        return qpos, bool(success), pos_error

    def set_init_ee_pose(
        self,
        ee_pose: sapien.Pose,
        arm: str = "left",
        tolerance: float = 5e-3,
    ):
        """Initialise the robot so one arm's end-effector is at ``ee_pose``.

        Solves IK for the requested arm and applies the result with
        :meth:`set_init_qpos`. Raises if IK cannot reach the target within
        ``tolerance`` metres.

        Args:
            ee_pose (sapien.Pose): Desired end-effector pose in the world frame.
            arm (str): ``"left"`` or ``"right"``.
            tolerance (float): Maximum acceptable position error, in metres.

        Returns:
            float: The achieved end-effector position error (metres).
        """
        qpos, _, pos_error = self.compute_ik(ee_pose, arm=arm)
        if pos_error > tolerance:
            raise RuntimeError(
                f"IK for the {arm} arm did not converge: position error "
                f"{pos_error * 1e3:.1f} mm > tolerance {tolerance * 1e3:.1f} mm. "
                f"The target pose may be out of reach."
            )
        self.set_init_qpos(qpos)
        return pos_error


def add_frank_to_scene(
    scene: sapien.Scene, frank_urdf_path: Optional[Path] = None
) -> Frank:
    """Convenience wrapper: load Frank at its default ready pose.

    Kept for backwards compatibility; returns a :class:`Frank` so callers can
    still initialise poses and apply gravity compensation.
    """
    return Frank(scene, urdf_path=frank_urdf_path)
