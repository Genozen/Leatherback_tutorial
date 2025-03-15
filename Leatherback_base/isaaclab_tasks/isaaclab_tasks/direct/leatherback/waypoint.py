import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg

##
# configuration
##

WAYPOINT_CFG = VisualizationMarkersCfg(
    prim_path="/World/Visuals/Cones",
    markers={
        "marker0": sim_utils.SphereCfg(
            radius=0.1,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        # this is wrong: VisualizationMarkersCfg.SphereCfg ---- VisualizationMarkersCfg has no attribute 'SphereCfg'
        "marker1": sim_utils.SphereCfg(
            radius=0.1,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
    }
)

"""Configuration for a single traffic cone"""
CONE_CFG = RigidObjectCfg(
    prim_path="/World/Cones",  # Base path for the collection
    spawn=sim_utils.ConeCfg(
        radius=0.15,
        height=0.5,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),  # Red cones
        # rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #     solver_position_iteration_count=4, solver_velocity_iteration_count=0
        # ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),  # Default position
)

"""Configuration for a collection of 10 traffic cones with unique paths"""
CONE_COLLECTION_CFG = RigidObjectCollectionCfg(
    rigid_objects={
        f"Cone_{i}": RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/Cone_{i}",
            spawn=sim_utils.ConeCfg(
                radius=0.15,
                height=0.5,
                # rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),  # Red cones
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=4, solver_velocity_iteration_count=0
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(i, 10.0, 2.0)),  # Default position
        )
        for i in range(5)  # Create 10 cones at different positions
        # "object_A": RigidObjectCfg(
        #     prim_path="/World/envs/env_.*/Object_A",
        #     spawn=sim_utils.SphereCfg(
        #         radius=0.1,
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=4, solver_velocity_iteration_count=0
        #         ),
        #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        #         collision_props=sim_utils.CollisionPropertiesCfg(),
        #     ),
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.5, 2.0)),
        # ),
    }
)