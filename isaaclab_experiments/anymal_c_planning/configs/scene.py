
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab_ros_experiments.planning.configs.assets.lidar import BetterRayCasterCfg

@configclass
class SquareCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(10, 10, 30)),
    )
    # walls
    north_wall = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/north_wall",
        spawn=sim_utils.MeshCuboidCfg(
            size=(20.0, 0.2, 3.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), # This make the wall static
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.21, 0.5, 0.73), metallic=0.2),
            physics_material=sim_utils.RigidBodyMaterialCfg()
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(10, 20.1, 1.5)),
    )
    south_wall = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/south_wall",
        spawn=sim_utils.MeshCuboidCfg(
            size=(20.0, 0.2, 3.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, rigid_body_enabled=True), # This make the wall static
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.21, 0.5, 0.73), metallic=0.2),
            physics_material=sim_utils.RigidBodyMaterialCfg()
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(10, -0.1, 1.5)),
    )
    east_wall = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/east_wall",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.2, 20.0, 3.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), # This make the wall static
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.21, 0.5, 0.73), metallic=0.2),
            physics_material=sim_utils.RigidBodyMaterialCfg()
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(20.1, 10, 1.5)),
    )
    west_wall = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/west_wall",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.2, 20.0, 3.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, rigid_body_enabled=True), # This make the wall static
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.21, 0.5, 0.73), metallic=0.2),
            physics_material=sim_utils.RigidBodyMaterialCfg()
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.1, 10, 1.5)),
    )










class UShapedCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    lidar_sensor = BetterRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        max_distance=20.,
        offset=RayCasterCfg.OffsetCfg(pos=(0., 0., 0.3),rot=(0., 0., 0., 1.)),
        attach_yaw_only=True,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=[-0, 0], # removed to represent a 2d lidar
            horizontal_fov_range=(0.0, 360.0), # real cfg
            horizontal_res=0.05),
        drift_range=(0.0, 0.0),
        debug_vis=False,
        mesh_prim_paths=[
            "/World/envs/env_.*/north_wall",\
            "/World/envs/env_.*/south_wall",\
            "/World/envs/env_.*/east_wall",\
            "/World/envs/env_.*/west_wall",\
            "/World/envs/env_.*/center_block",],
    )
    
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(10, 10, 30)),
    )
    # walls
    north_wall = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/north_wall",
        spawn=sim_utils.MeshCuboidCfg(
            size=(18.0, 0.2, 3.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), # This make the wall static
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.21, 0.5, 0.73), metallic=0.2),
            physics_material=sim_utils.RigidBodyMaterialCfg()
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(8.5, 16.5, 1.5)),
    )
    south_wall = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/south_wall",
        spawn=sim_utils.MeshCuboidCfg(
            size=(18.0, 0.2, 3.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, rigid_body_enabled=True), # This make the wall static
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.21, 0.5, 0.73), metallic=0.2),
            physics_material=sim_utils.RigidBodyMaterialCfg()
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(8.5, 0.5, 1.5)),
    )
    east_wall = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/east_wall",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.2, 18.0, 3.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), # This make the wall static
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.21, 0.5, 0.73), metallic=0.2),
            physics_material=sim_utils.RigidBodyMaterialCfg()
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(16.5, 8.5, 1.5)),
    )
    west_wall = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/west_wall",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.2, 18.0, 3.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, rigid_body_enabled=True), # This make the wall static
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.21, 0.5, 0.73), metallic=0.2),
            physics_material=sim_utils.RigidBodyMaterialCfg()
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 8.5, 1.5)),
    )
    center_block = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/center_block",
        spawn=sim_utils.MeshCuboidCfg(
            size=(12., 8.5, 3.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, rigid_body_enabled=True), # This make the wall static
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.21, 0.5, 0.73), metallic=0.2),
            physics_material=sim_utils.RigidBodyMaterialCfg()
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(10.5, 8.5, 1.5)),
    )


    box_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/box_1",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.25, 0.25, 1.),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, rigid_body_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.73, 0.23, 0.23), metallic=0.2),
            physics_material=sim_utils.RigidBodyMaterialCfg()
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(15, 14, 0.5)),
    )
    box_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/box_2",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.25, 0.25, 1.),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, rigid_body_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.23, 0.73, 0.23), metallic=0.2),
            physics_material=sim_utils.RigidBodyMaterialCfg()
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5, 8.5, 0.5)),
    )
    box_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/box_3",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.25, 0.25, 1.),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, rigid_body_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.23, 0.23, 0.73), metallic=0.2),
            physics_material=sim_utils.RigidBodyMaterialCfg()
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(15.5, 1.5, 0.5)),
    )