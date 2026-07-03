"""table_scene: builds the static table-top stage into a SAPIEN scene.

Adds a ground plane (floor), a table, and basic lighting — the physical stage
that the Frank robot and any future task objects sit on. This module is purely
about scene geometry and appearance; it holds no robot, no task, and no reward
logic.
"""

import numpy as np
import sapien
