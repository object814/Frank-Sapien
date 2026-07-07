"""scene: task-agnostic construction of the physical stage.

- :mod:`~frank_sapien.scene.lighting` -- a reusable lighting rig, shared across
  environments.
- :mod:`~frank_sapien.scene.table_top` -- the self-contained table-top scene
  builder (ground, table, lighting, and the Frank robot).

Scripts import the scene builder here; there is no task or reward logic.
"""
