"""table_top: one-script construction of the static table-top scene.

Self-contained scene builder for the table-top stage: it loads and places the
**table**, adds the ground and the reusable **lighting** rig
(:mod:`frank_sapien.scene.lighting`), loads the **Frank robot**
(:mod:`frank_sapien.agents.frank`), and wires up the collision filtering that
keeps the robot holding its pose. It returns a small :class:`TableTopScene`
handle holding the pieces callers need.

This lives in :mod:`frank_sapien.scene` because it is pure scene construction:
it has no task, reward, or Gymnasium ``reset``/``step`` logic. Demo/visualisation
scripts import it directly; the future Gymnasium environment in
:mod:`frank_sapien.envs` will build on top of it. Build a scene with::

    from frank_sapien.scene.table_top import build_table_top_scene

    world = build_table_top_scene()
    while running:
        world.frank.apply_gravity_compensation()
        world.scene.step()
        world.scene.update_render()
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import sapien

from frank_sapien.scene import lighting
from frank_sapien.agents.frank import Frank

ASSETS_PATH = Path(__file__).parent.parent / "assets"  # Path to environment assets


@dataclass
class TableTopScene:
    """Handle bundling the assembled scene and its contents."""

    scene: sapien.Scene
    frank: Frank
    table: List


# --- table -------------------------------------------------------------------
def _set_physical_material_and_freeze(obj, material):
    """Apply ``material`` to every collision shape of ``obj`` and make it
    kinematic (frozen in place), whether ``obj`` is an articulation or an actor.
    """
    # obj is an articulation
    if hasattr(obj, "get_links"):
        components = [c for link in obj.get_links() for c in link.get_components()]
    # obj is an actor / entity
    elif hasattr(obj, "get_components"):
        components = obj.get_components()
    else:
        return

    for component in components:
        if isinstance(component, sapien.physx.PhysxRigidBaseComponent):
            for shape in component.get_collision_shapes():
                shape.set_physical_material(material)
        if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
            if hasattr(component, "set_kinematic"):
                component.set_kinematic(True)
            elif hasattr(component, "kinematic"):
                component.kinematic = True


def _set_pose(obj, pose):
    """Set the pose of ``obj`` (articulation root pose or actor/entity pose)."""
    if hasattr(obj, "set_root_pose"):
        obj.set_root_pose(pose)
    elif hasattr(obj, "set_pose"):
        obj.set_pose(pose)
    else:
        raise ValueError(
            f"Object of type {type(obj)} has no set_pose or set_root_pose method."
        )


def add_table_to_scene(scene: sapien.Scene, table_urdf_path: Optional[Path] = None):
    """
    Load the table into ``scene``: place it, give it a physical material, and
    freeze it in place. Returns the loaded object(s).

    Args:
        scene (sapien.Scene): The scene to add the table to.
        table_urdf_path (Path): Path to the table URDF (defaults to the bundled
            ``assets/world/table/table.urdf``).
    """
    if not table_urdf_path:
        table_urdf_path = (ASSETS_PATH / "world" / "table" / "table.urdf").resolve()

    # Fixed pose for the table in the scene (matches the MuJoCo reference frame).
    table_pose = sapien.Pose(p=[-0.2028, 0.38, 0.6996], q=[0.5, 0.5, 0.5, -0.5])

    table_material = scene.create_physical_material(
        static_friction=1.0, dynamic_friction=1.0, restitution=0.0
    )

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.load_multiple_collisions_from_file = False

    articulations, actors = loader.load_multiple(urdf_file=str(table_urdf_path))
    loaded_objects = list(articulations) + list(actors)
    if len(loaded_objects) == 0:
        raise RuntimeError(f"No table object loaded from URDF: {table_urdf_path}")

    for obj in loaded_objects:
        _set_pose(obj, table_pose)
        _set_physical_material_and_freeze(obj, table_material)

    return loaded_objects


# --- collision filtering -----------------------------------------------------
# SAPIEN collision-group word index 2 is the "ignore" mask: two shapes never
# collide if they share any bit here. We reserve one high bit per decoupling:
# the robot's fixed trunk vs. the table it is mounted through, and the whole
# robot vs. the ground plane (its base sits partly below the floor).
_TRUNK_TABLE_IGNORE_BIT = 1 << 20
_ROBOT_GROUND_IGNORE_BIT = 1 << 21


def _iter_collision_shapes(obj):
    """Yield every collision shape of an articulation or actor/entity."""
    links = obj.get_links() if hasattr(obj, "get_links") else [obj]
    for link in links:
        entity = link.get_entity() if hasattr(link, "get_entity") else link
        for comp in entity.get_components():
            if isinstance(comp, sapien.physx.PhysxRigidBaseComponent):
                for shape in comp.get_collision_shapes():
                    yield shape


def _add_ignore_bit(obj, bit: int):
    """Set an "ignore" bit on every collision shape of ``obj`` (articulation,
    actor, or entity). Two shapes that share an ignore bit never collide."""
    for shape in _iter_collision_shapes(obj):
        groups = shape.get_collision_groups()
        groups[2] |= bit
        shape.set_collision_groups(groups)


def _decouple_trunk_from_table(frank: Frank, table: List):
    """Stop the robot's fixed trunk from colliding with the table.

    The table loads as a single convex hull that engulfs the volume the robot's
    rigidly-mounted base/lift/arm-mount occupy, which otherwise shoves the robot
    out of its initial pose every physics step. This shares an "ignore" bit
    between the table and every non-arm (trunk) link so they stop colliding,
    while the arms still collide with the table for manipulation. It is a
    collision *filter* only -- the table's solid collision geometry is unchanged.
    """
    def is_arm_link(name: str) -> bool:
        return "kinova_arm" in name and "base_link" not in name

    for obj in table:
        _add_ignore_bit(obj, _TRUNK_TABLE_IGNORE_BIT)
    for link in frank.robot.get_links():
        if not is_arm_link(link.get_name()):
            _add_ignore_bit(link, _TRUNK_TABLE_IGNORE_BIT)


def _disable_robot_ground_collision(frank: Frank, ground):
    """Stop the whole robot from colliding with the ground plane.

    The base sits partly below the floor (the URDF base is taller than the
    MuJoCo mount, so the root is lowered to match arm/table heights), which would
    otherwise generate constant base<->ground contacts. Shares an "ignore" bit
    between the ground and every robot link. Collision filter only -- the ground
    still collides with everything else in the scene.
    """
    _add_ignore_bit(ground, _ROBOT_GROUND_IGNORE_BIT)
    for link in frank.robot.get_links():
        _add_ignore_bit(link, _ROBOT_GROUND_IGNORE_BIT)


# --- assembly ----------------------------------------------------------------
def build_table_top_scene(
    hz: float = 100.0, init_config: str = "upright"
) -> TableTopScene:
    """
    Build the static table-top scene: ground, lighting, Frank, and the table.

    The robot is initialised *before* the table is added: its joint angles are
    applied directly (a teleport, so collision never blocks reaching the pose).
    The table is then added and decoupled from the robot's fixed trunk so the
    robot holds its initial pose instead of being pushed out of it.

    Args:
        hz (float): Simulation frequency in Hz (sets the physics timestep).
        init_config (str): Which initial arm configuration Frank starts in, one
            of :attr:`Frank.NAMED_CONFIGS` -- ``"upright"`` (arms raised, grippers
            down) or ``"rest"`` (arms spread just above the tabletop). Change this
            one argument to switch the starting pose.

    Returns:
        TableTopScene: Handle with ``.scene``, ``.frank`` and ``.table``. Call
        ``frank.apply_gravity_compensation()`` once per ``scene.step()`` to keep
        the robot holding its pose.
    """
    scene = sapien.Scene()
    scene.set_timestep(1.0 / hz)

    ground = scene.add_ground(0)
    lighting.add_lighting(scene)

    # 1. Load Frank at the chosen MuJoCo-matching config (init_config). The base
    #    keeps the MuJoCo reference orientation (identity yaw) so the robot faces
    #    +x -- the table sits at MuJoCo's (x, y) and the home joint angles carry
    #    over directly. The root is lowered by 0.31 m because MuJoCo bypasses the
    #    ridgeback base chain and hardcodes the arm mount at z=0.815 (lift
    #    ref=0.325), 0.31 m below SAPIEN's full URDF base chain; this constant
    #    offset drops the trunk + arms onto the MuJoCo heights.
    frank = Frank(
        scene,
        root_position=(0.0, 0.0, -0.31),
        root_quaternion=(1.0, 0.0, 0.0, 0.0),
        init_config=init_config,
    )

    # The base dips below the floor, so filter robot<->ground collisions.
    _disable_robot_ground_collision(frank, ground)

    # 2. Add the table after the robot is posed, then stop its solid collision
    #    hull from shoving the robot's fixed trunk out of that pose.
    table = add_table_to_scene(scene)
    _decouple_trunk_from_table(frank, table)

    return TableTopScene(scene=scene, frank=frank, table=table)
