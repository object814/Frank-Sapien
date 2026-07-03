"""frank: loads the custom Frank robot into a SAPIEN scene from its URDF.

Responsible for spawning and configuring the robot articulation — locating the
URDF under ``frank_sapien/assets/frank/``, setting the base pose, and applying
initial joint positions / drive properties. It exposes handles to the
articulation, its joints, and its links. No task rewards or high-level policy
live here.
"""

from pathlib import Path

import numpy as np
import sapien
