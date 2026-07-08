"""planning_assets: derive and cache the mplib planning URDF + SRDF for Frank.

mplib needs two things the shipped ``frank.urdf`` does not provide directly:

* a URDF whose joints are all **bounded** -- mplib's OMPL planner rejects the
  Kinova ``continuous`` joints (joints 1/3/5/7), so those are converted to
  ``revolute`` with generous limits; and
* an **SRDF** listing self-collision pairs to disable (adjacent / always-touching
  links), which mplib otherwise regenerates on every ``Planner()`` construction
  (a slow ~1 min sampling pass).

Both are produced once from ``frank.urdf`` and written next to it as
``frank_planning.urdf`` / ``frank_planning_mplib.srdf``. They must sit in the
``frank`` directory so the URDF's relative mesh paths still resolve (mplib does
not handle absolute mesh paths well). They are generated artifacts -- untracked
by the ``frank`` submodule and not committed to this repo. Subsequent calls
reuse them.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple

import numpy as np

ASSETS_PATH = Path(__file__).parent.parent / "assets"
_FRANK_DIR = ASSETS_PATH / "frank"
_FRANK_URDF = _FRANK_DIR / "frank.urdf"
# Kept in the frank dir so the URDF's relative "./meshes/..." paths still resolve.
_PLANNING_URDF = _FRANK_DIR / "frank_planning.urdf"

#: Generous symmetric bound applied to the Kinova continuous joints so mplib can
#: plan them (real Gen3 joints 1/3/5/7 are endless; +/-2*pi is plenty of range).
_CONTINUOUS_JOINT_LIMIT = 2 * np.pi
#: Random configs sampled when finding always-colliding link pairs for the SRDF.
_SRDF_NUM_SAMPLES = 3000


def _make_planning_urdf(frank_urdf: Path, out_urdf: Path):
    """Write ``out_urdf``: ``frank_urdf`` with the Kinova continuous joints
    converted to bounded revolute joints. Mesh paths are left untouched (relative)
    since ``out_urdf`` is written into the same directory as ``frank_urdf``."""
    tree = ET.parse(frank_urdf)
    root = tree.getroot()

    for joint in root.findall("joint"):
        if joint.get("type") == "continuous":
            joint.set("type", "revolute")
            limit = joint.find("limit")
            if limit is None:
                limit = ET.SubElement(joint, "limit")
            limit.set("lower", f"{-_CONTINUOUS_JOINT_LIMIT:.6f}")
            limit.set("upper", f"{_CONTINUOUS_JOINT_LIMIT:.6f}")
            if limit.get("effort") is None:
                limit.set("effort", "39")
            if limit.get("velocity") is None:
                limit.set("velocity", "1.0")

    tree.write(out_urdf)


def get_planning_assets(rebuild: bool = False) -> Tuple[str, str]:
    """
    Return ``(planning_urdf_path, srdf_path)`` as strings, building+caching them
    on first use.

    Args:
        rebuild (bool): Regenerate even if the cache exists.

    Returns:
        (str, str): Paths to the bounded planning URDF and its SRDF.
    """
    # generate_srdf writes ``<stem>_mplib.srdf`` next to the URDF.
    srdf = _PLANNING_URDF.with_name(_PLANNING_URDF.stem + "_mplib.srdf")

    if rebuild or not _PLANNING_URDF.is_file():
        _make_planning_urdf(_FRANK_URDF, _PLANNING_URDF)
        srdf.unlink(missing_ok=True)

    if rebuild or not srdf.is_file():
        # Imported lazily so importing this module doesn't require mplib.
        from mplib.urdf_utils import generate_srdf

        generate_srdf(str(_PLANNING_URDF), num_samples=_SRDF_NUM_SAMPLES)

    return str(_PLANNING_URDF), str(srdf)
