"""control_frank: drive the Frank robot with the motion controller.

Builds the table-top scene, wraps it in a :class:`FrankController`, and runs a
short scripted sequence of moves so you can watch (and edit) the controller in
action: end-effector pose / delta moves, single-joint moves, coordinated
dual-arm moves, and a deliberately-blocked move that the collision-aware planner
rejects.

Run with a display (or X11 into the dev container -- ``xhost +local:root`` on the
host first) to see it live; it also runs headless (no viewer) and just prints.

    python examples/control_frank.py
"""

import os

import numpy as np
import sapien

from frank_sapien.scene.table_top import build_table_top_scene
from frank_sapien.agents.controller import FrankController


def main():
    world = build_table_top_scene(init_config="rest")
    controller = FrankController(world)

    # Open a viewer if a display is available; wire it in so moves render live.
    viewer = None
    if os.environ.get("DISPLAY"):
        viewer = world.scene.create_viewer()
        viewer.set_camera_xyz(x=-2.2, y=0, z=1.4)
        viewer.set_camera_rpy(r=0, p=-0.4, y=0)

        def render():
            world.scene.update_render()
            viewer.render()

        controller.step_hook = render
    else:
        print("[control_frank] no DISPLAY; running headless.")

    def show(label, ok, arm="left"):
        arms = ("left", "right") if arm == "both" else (arm,)
        errs = " ".join(
            f"{a}={np.linalg.norm(np.asarray(controller.ee_pose(a).p)):.3f}" for a in arms
        )
        print(f"[control_frank] {label}: {'OK' if ok else 'REJECTED'}  (|ee| {errs})")

    # 1. Go to the 'rest' home configuration.
    show("move_home('rest')", controller.move_home("rest"))

    # 2. Left EE: lift 8 cm and pull back 6 cm (world-frame delta).
    show("left move_ee_delta [-0.06,0,+0.08]",
         controller.move_ee_delta([-0.06, 0.0, 0.08, 0, 0, 0], arm="left"))

    # 3. Left EE: absolute world pose (keep current orientation).
    p = controller.ee_pose("left")
    target = sapien.Pose(p=[p.p[0], p.p[1] + 0.08, p.p[2]], q=p.q)
    show("left move_ee_pose (+0.08 y)", controller.move_ee_pose(target, arm="left"))

    # 4. Single joint: rotate the left wrist by 0.5 rad.
    show("left move_joint_delta(joint_7, +0.5)",
         controller.move_joint_delta("left_kinova_arm_joint_7", 0.5))

    # 5. Coordinated dual-arm: raise both grippers 6 cm.
    lp, rp = controller.ee_pose("left"), controller.ee_pose("right")
    both = {
        "left": sapien.Pose(p=[lp.p[0], lp.p[1], lp.p[2] + 0.06], q=lp.q),
        "right": sapien.Pose(p=[rp.p[0], rp.p[1], rp.p[2] + 0.06], q=rp.q),
    }
    show("both move_ee_pose (+0.06 z)", controller.move_ee_pose(both, arm="both"), arm="both")

    # 6. A blocked move: drive the gripper straight down through the table.
    #    The collision-aware planner should reject this.
    show("left move_ee_delta [0,0,-0.30] (into table)",
         controller.move_ee_delta([0.0, 0.0, -0.30, 0, 0, 0], arm="left"))

    # 7. Back home.
    show("move_home('upright')", controller.move_home("upright"))

    print("[control_frank] sequence done.")
    if viewer is not None:
        while not viewer.closed:
            controller.frank.apply_gravity_compensation()
            world.scene.step()
            world.scene.update_render()
            viewer.render()


if __name__ == "__main__":
    main()
