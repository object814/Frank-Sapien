"""envs: Gymnasium environments (reserved for the RL interface layer).

This package will hold the task-facing environments -- the Gymnasium
``reset``/``step`` API, action/observation spaces, reward hooks, episode limits,
and ``gym.make`` registration -- plus the vectorized GPU variants. Each env
wraps a constructed scene (from :mod:`frank_sapien.scene`) and the Frank robot
(from :mod:`frank_sapien.agents`) with a task (from :mod:`frank_sapien.tasks`).

Nothing is implemented yet: the physical scene assembly lives in
:mod:`frank_sapien.scene.table_top` (``build_table_top_scene``), which these
environments will build on top of.
"""
