"""visualize_table_top: open the SAPIEN GUI viewer on the static table-top scene.

Entry-point/demo script that builds the static table-top environment (floor,
table, Frank robot) and opens the interactive SAPIEN viewer so you can orbit
around and inspect it. Run it on a machine with a display, or with X11
forwarding into the dev container (``xhost +local:root`` on the host first). It
runs no policy and no task logic — it is purely for visual inspection.
"""

import numpy as np
import sapien
from sapien.utils import Viewer

from frank_sapien.envs import table_top
