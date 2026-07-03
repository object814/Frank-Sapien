"""envs: assembled environments that combine a scene with an agent.

Each module wires together a scene (from :mod:`frank_sapien.scene`) and a robot
(from :mod:`frank_sapien.agents`) into a runnable environment. The Gymnasium
API and vectorized GPU variants will be layered on top of these later.
"""
