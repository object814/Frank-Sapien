"""control_frank_gui: jog the Frank robot from a pop-up panel in the SAPIEN viewer.

Opens the table-top scene in the interactive viewer with a docked control panel
(built as a native SAPIEN viewer plugin, so it lives in the same window). The
panel has:

* an arm selector -- ``left`` / ``right`` / ``both``;
* translation jogs -- ``X+ X-  Y+ Y-  Z+ Z-`` (world frame);
* rotation jogs -- ``Roll +/-  Pitch +/-  Yaw +/-`` (world frame);
* step-size sliders for translation (m) and rotation (rad);
* ``Rest`` and ``Upright`` home buttons.

Every jog is a collision-aware plan (via :class:`FrankController`): if the move
would drive the robot into the table or into itself -- or the target is
unreachable -- it is **refused** and the panel says so; the robot does not move.

Run with a display (or X11 into the dev container -- ``xhost +local:root`` on the
host first):

    python examples/control_frank_gui.py

This is a debugging / presentation tool; it is entirely self-contained here and
just drives the shared FrankController.
"""

import os
import sys

import numpy as np
import sapien
from sapien import internal_renderer as R
from sapien.utils.viewer import Plugin

from frank_sapien.scene.table_top import build_table_top_scene
from frank_sapien.agents.controller import FrankController

ARMS = ("left", "right", "both")


class ControlPanel(Plugin):
    """A SAPIEN viewer plugin: the jog panel. Button clicks queue an action that
    the main loop executes (moves must not run inside the render callback)."""

    def __init__(self):
        self.arm = "left"
        self.trans_step = 0.03   # metres per translation jog
        self.rot_step = 0.15     # radians per rotation jog (~8.6 deg)
        self.status = "ready"
        self._pending = None     # ("jog", delta6) or ("home", config)

    # -- Plugin API -------------------------------------------------------
    def init(self, viewer):
        self.viewer = viewer

    def get_ui_windows(self):
        win = R.UIWindow().Label("Frank Control").Pos(20, 60).Size(280, 560)
        win.append(
            self._arm_section(),
            self._translation_section(),
            self._rotation_section(),
            self._step_section(),
            self._home_section(),
            R.UIDisplayText().Text(f"Arm: {self.arm}"),
            R.UIDisplayText().Text(f"Status: {self.status}"),
        )
        return [win]

    # -- action queue (consumed by the main loop) ------------------------
    def take(self):
        act, self._pending = self._pending, None
        return act

    def _queue_jog(self, kind, idx, sign):
        delta = np.zeros(6)
        if kind == "t":
            delta[idx] = sign * self.trans_step
        else:
            delta[3 + idx] = sign * self.rot_step
        self._pending = ("jog", delta)

    # -- UI builders ------------------------------------------------------
    def _arm_section(self):
        sec = R.UISection().Label("Arm").Expanded(True)

        def btn(name):
            mark = "> " if self.arm == name else "  "
            return R.UIButton().Label(mark + name).Width(72).Callback(
                (lambda n: lambda _b: setattr(self, "arm", n))(name)
            )

        sec.append(R.UISameLine().append(btn("left"), btn("right"), btn("both")))
        return sec

    def _jog_pair(self, label, kind, idx):
        minus = R.UIButton().Label(f"{label} -").Width(64).Callback(
            (lambda k, i: lambda _b: self._queue_jog(k, i, -1))(kind, idx)
        )
        plus = R.UIButton().Label(f"{label} +").Width(64).Callback(
            (lambda k, i: lambda _b: self._queue_jog(k, i, +1))(kind, idx)
        )
        return R.UISameLine().append(minus, plus)

    def _translation_section(self):
        sec = R.UISection().Label("Translate (world, m)").Expanded(True)
        sec.append(
            self._jog_pair("X", "t", 0),
            self._jog_pair("Y", "t", 1),
            self._jog_pair("Z", "t", 2),
        )
        return sec

    def _rotation_section(self):
        sec = R.UISection().Label("Rotate (world, rad)").Expanded(True)
        sec.append(
            self._jog_pair("Roll", "r", 0),
            self._jog_pair("Pitch", "r", 1),
            self._jog_pair("Yaw", "r", 2),
        )
        return sec

    def _step_section(self):
        sec = R.UISection().Label("Step size").Expanded(True)
        sec.append(
            R.UISliderFloat().Label("trans (m)").Min(0.005).Max(0.10).Value(self.trans_step)
            .Callback(lambda s: setattr(self, "trans_step", s.value)),
            R.UISliderFloat().Label("rot (rad)").Min(0.02).Max(0.40).Value(self.rot_step)
            .Callback(lambda s: setattr(self, "rot_step", s.value)),
        )
        return sec

    def _home_section(self):
        sec = R.UISection().Label("Home").Expanded(True)
        rest = R.UIButton().Label("Rest").Width(96).Callback(
            lambda _b: setattr(self, "_pending", ("home", "rest"))
        )
        upright = R.UIButton().Label("Upright").Width(96).Callback(
            lambda _b: setattr(self, "_pending", ("home", "upright"))
        )
        sec.append(R.UISameLine().append(rest, upright))
        return sec


def _execute(controller: FrankController, panel: ControlPanel, action):
    """Run a queued panel action and report the outcome back to the panel."""
    kind, payload = action
    if kind == "home":
        panel.status = f"moving home ({payload})..."
        ok = controller.move_home(payload)
    else:  # jog
        delta = payload
        if panel.arm == "both":
            ok = controller.move_ee_delta({"left": delta, "right": delta}, arm="both")
        else:
            ok = controller.move_ee_delta(delta, arm=panel.arm)
    panel.status = "done" if ok else "REFUSED (collision or unreachable)"


def main():
    if not os.environ.get("DISPLAY"):
        sys.exit("control_frank_gui needs a display (set DISPLAY / X11-forward into the container).")

    world = build_table_top_scene(init_config="rest")
    controller = FrankController(world)

    viewer = world.scene.create_viewer()
    viewer.set_camera_xyz(x=-2.2, y=0.0, z=1.4)
    viewer.set_camera_rpy(r=0, p=-0.4, y=0)

    # Render live during a (blocking) move so motion is visible.
    controller.step_hook = lambda: (world.scene.update_render(), viewer.render())

    # Attach the control panel plugin to the existing viewer.
    panel = ControlPanel()
    panel.init(viewer)
    viewer.plugins = list(viewer.plugins) + [panel]

    print("[control_frank_gui] use the 'Frank Control' panel in the viewer window.")
    while not viewer.closed:
        controller.frank.apply_gravity_compensation()
        world.scene.step()
        world.scene.update_render()
        viewer.render()

        action = panel.take()
        if action is not None:
            _execute(controller, panel, action)


if __name__ == "__main__":
    main()
