from __future__ import annotations

import torch
import numpy as np
from collections.abc import Sequence

from gymnasium.spaces import Box

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObjectCfg, RigidObjectCollectionCfg, RigidObjectCollection
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass


from .waypoint import WAYPOINT_CFG, CONE_CFG, CONE_COLLECTION_CFG #customed
import time

# from .....isaaclab_assets.isaaclab_assets.robots.Leatherback import LEATHERBACK_CFG
from isaaclab_assets.robots.Leatherback import LEATHERBACK_CFG
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers


@configclass
class LeatherbackEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4          # Decimation - number of time steps between actions, it was 2
    episode_length_s = 20.0 # Max each episode should last in seconds, 30 s seems a lot
    # action_scale = 100.0    # [N]
    action_space = 2        # Number of actions the neural network shuold return   
    observation_space = 8   # Number of observations fed into neural network
    state_space = 0         # Observations to be used in Actor Critic Training

    # simulation frames Hz
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation)

    # TODO
    # robot
    robot_cfg: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # region Waypoints
    waypoint_cfg: VisualizationMarkersCfg = WAYPOINT_CFG
    cone_collection_cfg: RigidObjectCollectionCfg = CONE_COLLECTION_CFG  # Ensure naming consistency


    throttle_dof_name = [
        "Wheel__Knuckle__Front_Left",
        "Wheel__Knuckle__Front_Right",
        "Wheel__Upright__Rear_Right",
        "Wheel__Upright__Rear_Left"
    ]
    steering_dof_name = [
        "Knuckle__Upright__Front_Right",
        "Knuckle__Upright__Front_Left",
    ]

    env_spacing = 32.0 # depends on the ammount of Goals, 32 is a lot

    # scene - 4096 environments
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=env_spacing, replicate_physics=True)


class LeatherbackEnv(DirectRLEnv):
    """Reinforcement learning environment for the Leatherback robot."""
    cfg: LeatherbackEnvCfg


    def __init__(self, cfg: LeatherbackEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._throttle_dof_idx, _ = self.leatherback.find_joints(self.cfg.throttle_dof_name)
        self._steering_dof_idx, _ = self.leatherback.find_joints(self.cfg.steering_dof_name)
        # self.action_scale = self.cfg.action_scale

        # self.joint_pos = self.leatherback.data.joint_pos
        # self.joint_vel = self.leatherback.data.joint_vel
        # self._throttle_action = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)
        # self._steering_action = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)

        self._throttle_state =  torch.zeros((self.num_envs,4), device=self.device, dtype=torch.float32)
        self._steering_state =  torch.zeros((self.num_envs,2), device=self.device, dtype=torch.float32)

        self._goal_reached =  torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)
        self.task_completed =  torch.zeros((self.num_envs), device=self.device, dtype=torch.bool)
        # region Number of Goals
        self._num_goals = 10 # 10 seems too much
        # end region Number of Goals
        self._target_positions =  torch.zeros((self.num_envs, self._num_goals, 2), device=self.device, dtype=torch.float32)
        self._markers_pos =  torch.zeros((self.num_envs, self._num_goals, 3), device=self.device, dtype=torch.float32)

        self.env_spacing = self.cfg.env_spacing
        self.course_length_coefficient = 2.5
        self.course_width_coefficient = 2.0

        # reward parameters
        # Tolerance
        # position_tolerance: float = 0.15, started at 0.2
        """Tolerance for the position of the robot. Defaults to 1cm."""
        self.position_tolerance: float = 0.15
        self.goal_reached_bonus: float = 10.0
        self.position_progress_weight: float = 1.0
        self.heading_coefficient = 0.25
        self.heading_progress_weight: float = 0.05

        self._target_index = torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)
        # self.object_states = []

    def _setup_scene(self):
        """Called by the parent to set up the scene and assets."""
        print("[INFO] Setting up scene...")
        self.leatherback = Articulation(self.cfg.robot_cfg)
        self.waypoints = VisualizationMarkers(self.cfg.waypoint_cfg)
        # self.cones = RigidObject(self.cfg.cone_cfg)
        self.cones = RigidObjectCollection(self.cfg.cone_collection_cfg)
        self.object_state = []

        # Add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Clone environment layout
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        # Add articulation to the scene
        self.scene.articulations["leatherback"] = self.leatherback
        self.scene.rigid_object_collections["cones"] = self.cones  # Add as a collection

        # Add dome light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # def _maybe_update_observation_space(self):
    #     """
    #     Overwrites the placeholder observation space if you want to measure 
    #     the real dimension at runtime (from self._get_observations()).
    #     """
    #     # For example, if we want to dynamically set the shape:
    #     sample_obs = self._get_observations()["policy"]
    #     obs_dim = sample_obs.shape[1]

    #     if obs_dim != self.cfg.num_observations:
    #         self.cfg.observation_space = Box(
    #             low=-np.inf,
    #             high=np.inf,
    #             shape=(obs_dim,),
    #             dtype=np.float32,
    #         )
    #         self.cfg.num_observations = obs_dim
    #         print(f"[INFO] Updated observation space to: {self.cfg.observation_space}")


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Multiplier for the throttle velocity. The action is in the range [-1, 1] and the radius of the wheel is 0.06m"""
        throttle_scale = 1 # when set to 2 it trains but the cars are flying, 3 you get NaNs
        # throttle_scale = 60.0
        throttle_max = 50.0
        """Multiplier for the steering position. The action is in the range [-1, 1]"""
        steering_scale = 0.1
        # steering_scale = math.pi / 4.0
        steering_max = 0.75

        self._throttle_action = actions[:, 0].repeat_interleave(4).reshape((-1, 4)) * throttle_scale
        self._throttle_action += self._throttle_state 
        self.throttle_action = torch.clamp(self._throttle_action, -throttle_max, throttle_max * 0.1)
        self._throttle_state = self._throttle_action
        
        self._steering_action = actions[:, 1].repeat_interleave(2).reshape((-1, 2)) * steering_scale
        self._steering_action += self._steering_state
        self._steering_action = torch.clamp(self._steering_action, -steering_max, steering_max)
        self._steering_state = self._steering_action
        # self.actions = self.action_scale * actions.clone()
        # end region _pre_physics_step

    def _apply_action(self) -> None:
        self.leatherback.set_joint_velocity_target(self._throttle_action, joint_ids=self._throttle_dof_idx)
        self.leatherback.set_joint_position_target(self._steering_state, joint_ids=self._steering_dof_idx)

    def _get_observations(self) -> dict:

        # position error
        current_target_positions = self._target_positions[self.leatherback._ALL_INDICES, self._target_index]
        self._position_error_vector = current_target_positions - self.leatherback.data.root_pos_w[:, :2]
        self._previous_position_error = self._position_error.clone()
        self._position_error = torch.norm(self._position_error_vector, dim=-1) # had placed dim=1

        # heading error
        heading = self.leatherback.data.heading_w
        target_heading_w = torch.atan2(
            self._target_positions[self.leatherback._ALL_INDICES, self._target_index, 1] - self.leatherback.data.root_link_pos_w[:, 1],
            self._target_positions[self.leatherback._ALL_INDICES, self._target_index, 0] - self.leatherback.data.root_link_pos_w[:, 0],
        )
        self.target_heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))

        obs = torch.cat(
            (
                self._position_error.unsqueeze(dim=1),
                torch.cos(self.target_heading_error).unsqueeze(dim=1),
                torch.sin(self.target_heading_error).unsqueeze(dim=1),
                self.leatherback.data.root_lin_vel_b[:, 0].unsqueeze(dim=1),
                self.leatherback.data.root_lin_vel_b[:, 1].unsqueeze(dim=1),
                self.leatherback.data.root_ang_vel_w[:, 2].unsqueeze(dim=1),
                self._throttle_state[:, 0].unsqueeze(dim=1),
                self._steering_state[:, 0].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        
        if torch.any(obs.isnan()):
            raise ValueError("Observations cannot be NAN") # When NaN, car flies...

        observations = {"policy": obs}
        return observations
    
    def _get_rewards(self) -> torch.Tensor:
        """
        Example reward that uses distance to a target plus a bonus for reaching it.
        """

                # Trying to Implement TorchScript

        # # position progress
        position_progress_rew = self._previous_position_error - self._position_error

        # Heading Distance - changing the numerator to positive make it drive backwards
        target_heading_rew = torch.exp(-torch.abs(self.target_heading_error) / self.heading_coefficient)

        # Checks if the goal is reached
        goal_reached = self._position_error < self.position_tolerance

        # if the goal is reached, the target index is updated
        self._target_index = self._target_index + goal_reached

        self.task_completed = self._target_index > (self._num_goals -1)

        self._target_index = self._target_index % self._num_goals

        composite_reward = (
            position_progress_rew*self.position_progress_weight +
            target_heading_rew*self.heading_progress_weight +
            goal_reached*self.goal_reached_bonus
        )

        # region debugging
        # Update Waypoints
        # marker0 to marker9 is RED   # 1 is red
        # marker 10 to marker19 is BLUE  # 0 is blue
        one_hot_encoded = torch.nn.functional.one_hot(self._target_index.long(), num_classes=self._num_goals) # one_hot - all zeros except the target_index
        marker_indices = one_hot_encoded.view(-1).tolist()
        self.waypoints.visualize(marker_indices=marker_indices)

        if torch.any(composite_reward.isnan()):
            raise ValueError("Rewards cannot be NAN")

        return composite_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        
        task_failed = self.episode_length_buf > self.max_episode_length

        # task completed is calculated in get_rewards before target_index is wrapped around
        return task_failed, self.task_completed

    def _reset_idx(self, env_ids: Sequence[int] | None):
        print("~~~~~~~~~~~~~~~~~~~RESET~~~~~~~~~~~~~~~~~~~~~")
        if env_ids is None:
            print("XXXXX NONE XXXXX")
            env_ids = self.leatherback._ALL_INDICES
        super()._reset_idx(env_ids)

        # num_reset = len(env_ids)
        num_reset = len(env_ids)

        # region Reset
        # reset from config
        default_state = self.leatherback.data.default_root_state[env_ids]   # first there are pos, next 4 quats, next 3 vel,next 3 ang vel, 
        leatherback_pose = default_state[:, :7]                             # proper way of getting default pose from config file
        leatherback_velocities = default_state[:, 7:]                       # proper way of getting default velocities from config file
        joint_positions = self.leatherback.data.default_joint_pos[env_ids]  # proper way to get joint positions from config file
        joint_velocities = self.leatherback.data.default_joint_vel[env_ids] # proper way to get joint velocities from config file

        leatherback_pose[:, :3] += self.scene.env_origins[env_ids] # Adds center of each env position in leatherback position

        # Randomize Steering position at start of track
        leatherback_pose[:, 0] -= self.env_spacing / 2
        leatherback_pose[:, 1] += 2.0 * torch.rand((num_reset), dtype=torch.float32, device=self.device) * self.course_width_coefficient

        # Randomize Starting Heading
        angles = torch.pi / 6.0 * torch.rand((num_reset), dtype=torch.float32, device=self.device)

        # Isaac Sim Quaternions are w first (w, x, y, z) To rotate about the Z axis, we will modify the W and Z values
        leatherback_pose[:, 3] = torch.cos(angles * 0.5)
        leatherback_pose[:, 6] = torch.sin(angles * 0.5)

        self.leatherback.write_root_pose_to_sim(leatherback_pose, env_ids)
        self.leatherback.write_root_velocity_to_sim(leatherback_velocities, env_ids)
        self.leatherback.write_joint_state_to_sim(joint_positions, joint_velocities, None, env_ids)
        # end region reset robot
        
        # region Reset Actions
        # It trains without this
        # self._throttle_state[env_ids] = 0.0
        # self._throttle_action[env_ids] = 0.0 # AttributeError: 'LeatherbackEnv' object has no attribute '_throttle_action'
        # end region Reset Actions

        # region reset goals
        self._target_positions[env_ids, :, :] = 0.0
        self._markers_pos[env_ids, :, :] = 0.0

        spacing = 2 / self._num_goals
        target_positions = torch.arange(-0.8, 1.1, spacing, device=self.device) * self.env_spacing / self.course_length_coefficient
        self._target_positions[env_ids, :len(target_positions), 0] = target_positions
        self._target_positions[env_ids, :, 1] = torch.rand((num_reset, self._num_goals), dtype=torch.float32, device=self.device) + self.course_length_coefficient
        self._target_positions[env_ids, :] += self.scene.env_origins[env_ids, :2].unsqueeze(1)

        self._target_index[env_ids] = 0

        # Update the visual markers
        self._markers_pos[env_ids, :, :2] = self._target_positions[env_ids]
        visualize_pos = self._markers_pos.view(-1, 3)
        self.waypoints.visualize(translations=visualize_pos)
        # end Region Reset Goals


        ### --------------- Region Reset: Cone Collections ----------------- ###
        num_objects = len(CONE_COLLECTION_CFG.rigid_objects)
        # self.object_state = self.cones.data.default_object_state.clone()
        self.object_state = self.cones.data.default_object_state.clone()
        object_ids = torch.arange(num_objects, device=self.device)  # Each object has a unique index
        # self.object_state[env_ids, :, :] = self.cones.data.default_object_state[env_ids, :, :].clone()

        # Offset cone positions per environment
        self.object_state[env_ids, :, :3] = self.scene.env_origins[env_ids].unsqueeze(1).expand(num_reset, num_objects, 3)
        """ use = since it's to individual env,  += will cause shift to other locations??"""
        # self.object_state[env_ids] = self.cones.data.default_object_state[env_ids].clone()
        # self.cones.write_object_link_pose_to_sim(self.object_state[env_ids, :, :7])  # Position + Quaternion
        # self.cones.write_object_com_velocity_to_sim(self.object_state[env_ids, :, 7:])  # Linear + Angular velocity



        # # Add slight randomization in X and Y directions
        self.object_state[env_ids, :, 0] += torch.rand((num_reset, num_objects), dtype=torch.float32, device=self.device) * 2.0 - 1.0  # X-axis jitter
        self.object_state[env_ids, :, 1] += 3 #torch.rand((num_reset, num_objects), dtype=torch.float32, device=self.device) #* 1.5 - 0.75

        # # Write updated positions and velocities to simulation
        # self.cones.write_object_link_pose_to_sim(self.object_state[..., :7])  # Position + Quaternion
        # print(f"{self.object_state[env_ids, :, :7]}, {env_ids}, {object_ids}")
        print(f"env_ids before function call: {env_ids}, type: {type(env_ids)}, device: {env_ids.device if isinstance(env_ids, torch.Tensor) else 'CPU'}")
        self.cones.write_object_link_pose_to_sim(self.object_state[env_ids, :, :7], env_ids, object_ids)  # Position + Quaternion
        # self.cones.write_object_com_velocity_to_sim(self.object_state[..., 7:])  # Linear + Angular velocity
        # self.cones.data.default_object_state.copy_(self.object_state)
        # print(f"Expected num_objects: 5, Actual num_objects: {self.object_state.shape[1]}")
        # print(f"env_ids: {env_ids}")  # Should always be within range(0, 32)
        # print(f"Reset triggered for: {env_ids.shape[0]} environments")

        # print(f"env_ids.shape: {env_ids.shape}")
        # print(f"num_reset (computed): {num_reset}")
        # print(f"num_objects: {num_objects}")
        # print(f"self.scene.env_origins[env_ids].shape: {self.scene.env_origins[env_ids].shape}")
        # print(f"Expected shape: ({num_reset}, {num_objects}, 3)")
        # print(f"Actual shape: {self.scene.env_origins[env_ids].unsqueeze(1).expand(num_reset, num_objects, 3).shape}")
        ### End Region Reset ###




        # region make sure the position error and position dist are up to date after the reset
        # reset positions error
        current_target_positions = self._target_positions[self.leatherback._ALL_INDICES, self._target_index]
        self._position_error_vector = current_target_positions[:, :2] - self.leatherback.data.root_pos_w[:, :2]
        self._position_error = torch.norm(self._position_error_vector, dim=-1)
        self._previous_position_error = self._position_error.clone()

        # reset heading error
        heading = self.leatherback.data.heading_w[:]
        target_heading_w = torch.atan2( 
            self._target_positions[:, 0, 1] - self.leatherback.data.root_pos_w[:, 1],
            self._target_positions[:, 0, 0] - self.leatherback.data.root_pos_w[:, 0],
        )
        self._heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))
        self._previous_heading_error = self._heading_error.clone()
        # end region