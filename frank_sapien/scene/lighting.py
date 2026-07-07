"""lighting: sets up realistic lighting for a SAPIEN scene.

A single directional light leaves every surface it does not directly hit
falling back to the ambient term only, which reads as flat and lifeless. This
module installs a small, neutral three-light rig (a key light plus two fill
lights) on top of a modest white ambient so the Frank robot and the table read
with natural, untinted colours.

Note on the SAPIEN API: ``scene.add_directional_light(direction, color, ...)``
takes the *direction* first and the *colour* second. Passing them in the wrong
order (e.g. a colour of ``[0, -1, -1]``) injects negative colour channels and
tints the whole scene pink -- this module keeps the two straight.
"""

import sapien


def add_lighting(
    scene: sapien.Scene,
    ambient: float = 0.3,
    key_intensity: float = 3.0,
    fill_intensity: float = 1.0,
):
    """
    Add a neutral key + fill directional light rig and ambient light to a scene.

    Args:
        scene (sapien.Scene): The scene to light.
        ambient (float): Grey-level of the ambient light (applied to R, G, B).
        key_intensity (float): Brightness of the main (key) light from above.
        fill_intensity (float): Brightness of the two side fill lights that
            lift the shadows so no face is left flat.
    """
    scene.set_ambient_light([ambient, ambient, ambient])

    # Key light: from above, angled down -- the dominant source.
    scene.add_directional_light(
        direction=[0.0, -1.0, -1.0],
        color=[key_intensity, key_intensity, key_intensity],
        shadow=True,
    )
    # Fill lights: from either side to soften shadows and avoid flat faces.
    scene.add_directional_light(
        direction=[1.0, 0.5, -0.5],
        color=[fill_intensity, fill_intensity, fill_intensity],
    )
    scene.add_directional_light(
        direction=[-1.0, 0.5, -0.5],
        color=[fill_intensity, fill_intensity, fill_intensity],
    )
