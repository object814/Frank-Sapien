"""visualise_table_top: open the SAPIEN GUI viewer on the static table-top scene.

Entry-point/demo script that builds the static table-top environment (floor,
table, lighting, Frank robot) and opens the interactive SAPIEN viewer so you can
orbit around and inspect it. Run it on a machine with a display, or with X11
forwarding into the dev container (``xhost +local:root`` on the host first). It
runs no policy and no task logic -- it is purely for visual inspection.

The whole scene is assembled by :func:`frank_sapien.scene.table_top.build_table_top_scene`;
this file only owns the viewer and the render loop.
"""

from frank_sapien.scene.table_top import build_table_top_scene


def main():
    # Starting configuration: "upright" (arms raised, grippers down) or
    # "rest" (arms spread just above the tabletop). Change this to switch.
    world = build_table_top_scene(hz=100.0, init_config="rest")

    viewer = world.scene.create_viewer()
    viewer.set_camera_xyz(x=-2, y=0, z=1)
    viewer.set_camera_rpy(r=0, p=-0.3, y=0)

    while not viewer.closed:
        # Gravity compensation by control: keep the arms from collapsing under
        # gravity while leaving the physics gravity intact.
        world.frank.apply_gravity_compensation()
        world.scene.step()
        world.scene.update_render()
        viewer.render()


if __name__ == "__main__":
    main()
