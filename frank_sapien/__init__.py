"""frank_sapien: GPU-accelerated, Gym-style SAPIEN environments for the Frank robot.

Top-level package. At this stage it provides the building blocks for a *static*
table-top scene (floor + table + Frank robot) that can be visualized in the
SAPIEN GUI viewer. Sub-packages:

- ``scene``:  reusable scene construction (floor, table, lighting).
- ``agents``: the Frank robot loader and (later) its controllers.
- ``envs``:   assembled environments that combine a scene with an agent.
"""
