"""table_top: assembles the static table-top environment.

Creates a SAPIEN scene, builds the table-top stage
(:mod:`frank_sapien.scene.table_scene`), and loads the Frank robot
(:mod:`frank_sapien.agents.frank`) on top of it. At this milestone it is a plain
scene container meant for visualization; the Gymnasium ``reset``/``step`` API is
added in a later step.
"""

import numpy as np
import sapien

from frank_sapien.scene import table_scene
from frank_sapien.agents import frank
