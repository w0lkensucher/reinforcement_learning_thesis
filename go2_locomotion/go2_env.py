"""OBSOLETE NOW, USING NEW STRUCTURE"""
import torch
import math
import genesis as gs
import numpy as np
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class Go2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, wind_force=False, uneven_terrain=False):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        # ADD THESE LINES for petting capability
        self.petting_enabled = env_cfg.get('enable_petting', False)
        if self.petting_enabled:
            # Petting detection parameters
            self.head_touched = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            self.gesture_timer = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
            self.gesture_duration = 100  # 2 seconds at 50Hz
            self.gesture_cooldown = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
            self.cooldown_duration = 250  # 5 seconds cooldown
            
            # Detection parameters
            self.head_touch_radius = env_cfg.get('head_touch_radius', 0.2)  # 20cm detection range
            self.gentle_speed_threshold = env_cfg.get('gentle_speed_threshold', 0.3)  # Must be moving slowly
            self.head_height_offset = env_cfg.get('head_height_offset', 0.25)  # Head 25cm above base

            # Height-based detection parameters
            self.use_height_detection = env_cfg.get('use_height_detection', True)
            self.gentle_press_range = env_cfg.get('gentle_press_range', [0.02, 0.08])  # 2-8cm push down
            self.gentle_vel_threshold = env_cfg.get('gentle_vel_threshold', 0.1)  # Downward velocity

            # Add gesture observations to obs buffer
            self.num_obs += 2  # gesture_active and touch_detected
            self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float)

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        self.episode_sums = dict()
        for name in self.reward_scales.keys():
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        self.episode_sums['reward'] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                # for this locomotion policy there are usually no more than 30 collision pairs
                # set a low value can save memory
                max_collision_pairs=30,
            ),
            show_viewer=show_viewer,
        )

        # add wind force field; make it conditional later if needed
        if wind_force:
            ff = gs.force_fields.Wind(direction=(1, 0, 0), strength=5.0, radius=10.0, center=(0, 0, 0))
            self.scene.add_force_field(ff)

        # add plane
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )
        
        self.obstacle_entities = []
        self.obstacle_positions = []

        if env_cfg.get("use_obstacles", False):
            self._create_obstacles()

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]]

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions = dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), gs.device)

    def step(self, actions):
        # Check for head petting
        if self.petting_enabled:
            self._check_head_petting()
            
            # Override actions if gesture is active
            active_gesture = self.gesture_timer > 0
            if active_gesture.any():
                gesture_actions = self._get_petting_gesture_actions()
                # Apply gesture only to environments where timer is active
                actions = torch.where(active_gesture.unsqueeze(1), gesture_actions, actions)
        
        # Update timers
        self.gesture_timer = torch.clamp(self.gesture_timer - 1, 0, self.gesture_duration)
        self.gesture_cooldown = torch.clamp(self.gesture_cooldown - 1, 0, self.cooldown_duration)
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat),
            rpy=True,
            degrees=True,
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )
        self._resample_commands(envs_idx)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        self.episode_sums['reward'] += self.rew_buf

        low_obs, high_obs = self._detect_nearby_obstacles()
        # compute observations
        if self.petting_enabled:
            gesture_active = (self.gesture_timer > 0).float()
            touch_detected = self.head_touched.float()
            
            self.obs_buf = torch.cat([
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
                low_obs.unsqueeze(1),   # 1: signal for low obstacle
                high_obs.unsqueeze(1),  # 1: signal for high obstacle
                gesture_active.unsqueeze(1),  # 1: gesture active signal
                touch_detected.unsqueeze(1),  # 1: touch detected signal
            ], axis=-1)
        else:
            self.obs_buf = torch.cat(
                [
                    self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                    self.projected_gravity,  # 3
                    self.commands * self.commands_scale,  # 3
                    (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                    self.dof_vel * self.obs_scales["dof_vel"],  # 12
                    self.actions,  # 12
                    low_obs.unsqueeze(1),   # 1: signal for low obstacle
                    high_obs.unsqueeze(1),  # 1: signal for high obstacle
                ],
                axis=-1,
            )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

        #if self.env_cfg.get('randomize_obstacles_per_episode', False):
        #    self._randomize_obstacles_positions()
        
        # self._resample_commands(envs_idx)

    def force_randomize_obstacles(self):
        self._randomize_obstacles_positions()

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])
    
    def _create_obstacles(self):
        """Create obstacles in the environment"""
        import numpy as np

        try:
            terrain_size = self.env_cfg['terrain_size']
            density = self.env_cfg['obstacle_density']
            clear_radius = self.env_cfg['clear_radius']

            area = terrain_size[0] * terrain_size[1]
            num_obstacles = int(area * density)
            
            print(f"Creating {num_obstacles} obstacles...")

            self.obstacle_positions = []
            self.obstacle_entities = []

            for i in range(num_obstacles):
                attempts = 0
                while attempts < 50:
                    x = np.random.uniform(-terrain_size[0]/2, terrain_size[0]/2)
                    y = np.random.uniform(-terrain_size[1]/2, terrain_size[1]/2)
                    height = np.random.uniform(*self.env_cfg['obstacle_height_range'])

                    robot_spawn = self.env_cfg['base_init_pos'][:2]
                    dist_to_spawn = np.sqrt((x - robot_spawn[0])**2 + (y - robot_spawn[1])**2)
                    if dist_to_spawn < clear_radius:
                        attempts += 1
                        continue

                    too_close = False
                    for pos in self.obstacle_positions:
                        dist = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
                        if dist < self.env_cfg['obstacle_spacing_min']:
                            too_close = True
                            break

                    if not too_close:
                        self.obstacle_positions.append([x, y, height])
                        entity = self._add_single_obstacle(x, y, height)
                        if entity is not None:    
                            self.obstacle_entities.append(entity)
                        break

                    attempts += 1

            print(f"Successfully created {len(self.obstacle_positions)} obstacles")
            
        except Exception as e:
            print(f"Warning: Failed to create obstacles: {e}")
            print("Continuing without obstacles...")
    
    def _add_single_obstacle(self, x, y, height):
        """Add a single obstacle at position (x, y)"""
        import numpy as np

        try:
            obstacle_type = np.random.choice(self.env_cfg['obstacle_types'])
            width = np.random.uniform(*self.env_cfg['obstacle_width_range'])

            if obstacle_type == 'box':
                geom = gs.morphs.Box(
                    pos=(x, y, height / 2),
                    size=(width, width, height)
                )
            elif obstacle_type == 'cylinder':
                geom = gs.morphs.Cylinder(
                    pos=(x, y, height / 2),
                    radius=width/2,
                    height=height
                )

            # Material is passed to add_entity, not to the morph
            material = gs.materials.Rigid(friction=0.8)
            entity = self.scene.add_entity(geom, material=material)
            return entity
        except Exception as e:
            print(f"Warning: Failed to create obstacle at ({x:.2f}, {y:.2f}): {e}")

    def _reward_forward_progress(self):
        """Reward for forward progress in the x direction"""
        return self.base_lin_vel[:, 0]
    
    def _detect_nearby_obstacles(self):
        """Detect obstacles near the robot and classify as jumpable or avoidable"""
        if not self.env_cfg.get('use_obstacles', False) or len(self.obstacle_positions) == 0:
            return torch.zeros(self.num_envs, device=self.device), torch.zeros(self.num_envs, device=self.device)
        
        robot_pos = self.base_pos[:, :2]  # x, y position
        detection_radius = 1.5  # Distance ahead to look for obstacles
        
        # For each environment, find nearest obstacle ahead
        nearest_low_obstacle = torch.zeros(self.num_envs, device=self.device)
        nearest_high_obstacle = torch.zeros(self.num_envs, device=self.device)
        
        # Get robot's forward direction
        robot_heading = self.base_euler[:, 2]  # Yaw angle
        forward_dir = torch.stack([torch.cos(robot_heading), torch.sin(robot_heading)], dim=1)
        
        for i, obs_data in enumerate(self.obstacle_positions):
            obs_pos = torch.tensor(obs_data[:2], device=self.device)  # x, y
            obs_height = obs_data[2] if len(obs_data) > 2 else 0.1  # height or default
            
            # Vector from robot to obstacle
            to_obstacle = obs_pos.unsqueeze(0) - robot_pos  # Shape: (num_envs, 2)
            distance = torch.norm(to_obstacle, dim=1)
            
            # Check if obstacle is ahead (dot product with forward direction)
            is_ahead = torch.sum(to_obstacle * forward_dir, dim=1) > 0
            is_close = (distance < detection_radius) & is_ahead
            
            # Classify obstacle by height
            jump_threshold = self.reward_cfg.get('jump_height_threshold', 0.08)
            
            if obs_height < jump_threshold:
                # Low obstacle - should jump
                obstacle_signal = 1.0 / (distance + 0.1)  # Stronger signal for closer obstacles
                nearest_low_obstacle = torch.where(is_close, 
                                                torch.max(nearest_low_obstacle, obstacle_signal),
                                                nearest_low_obstacle)
            else:
                # High obstacle - should avoid
                obstacle_signal = 1.0 / (distance + 0.1)
                nearest_high_obstacle = torch.where(is_close,
                                                torch.max(nearest_high_obstacle, obstacle_signal),
                                                nearest_high_obstacle)
        
        return nearest_low_obstacle, nearest_high_obstacle

    def _reward_jump_reward(self):
        """Reward for jumping over low obstacles, high obstacles are ignored"""
        low_obstacles, high_obstacles = self._detect_nearby_obstacles()
        
        # Get current base height and vertical velocity
        current_height = self.base_pos[:, 2]
        base_vel_z = self.base_lin_vel[:, 2]
        
        # Target jump height for low obstacles
        jump_target = self.reward_cfg.get('jump_reward_height', 0.15)
        base_target = self.reward_cfg['base_height_target']
        
        # Reward jumping when low obstacles are detected
        jump_height_above_normal = current_height - base_target
        is_jumping = (jump_height_above_normal > 0.05) | (base_vel_z > 0.2)  # 5cm above normal
        
        # Reward formula: jump when low obstacles detected, stay normal otherwise
        jump_reward = torch.where(
            high_obstacles > 0.1,  # High obstacle detected
            -1.0 * is_jumping, # Penalty for jumping over high obstacles,
            torch.where(
                low_obstacles > 0.1,  # Low obstacle detected
                torch.where(
                    is_jumping,
                    torch.clamp(jump_height_above_normal / jump_target, 0, 1),  # Reward proportional to jump height
                    -0.5 * low_obstacles  # Penalty for not jumping when should
                ),
                torch.where(
                    is_jumping,
                    -0.3 * jump_height_above_normal,  # Small penalty for unnecessary jumping
                    torch.zeros_like(current_height)   # No reward/penalty for normal walking
                )
            )
        )
        
        return jump_reward

    def _reward_jump_timing(self):
        """Reward for proper jump timing"""
        low_obstacles, _ = self._detect_nearby_obstacles()
        
        # Get vertical velocity
        base_vel_z = self.base_lin_vel[:, 2]
        
        # Reward upward velocity when approaching low obstacles
        timing_reward = torch.where(
            low_obstacles > 0.3,  # Close to low obstacle
            torch.clamp(base_vel_z, 0, 1),  # Reward upward velocity
            torch.where(
                low_obstacles > 0.1,  # Moderate distance to low obstacle
                torch.clamp(base_vel_z * 0.5, 0, 0.5),  # Smaller reward
                torch.zeros_like(base_vel_z)  # No timing reward
            )
        )
        
        return timing_reward

    def _reward_landing_stability(self):
        """Reward for stable landings after jumps"""
        current_height = self.base_pos[:, 2]
        base_vel_z = self.base_lin_vel[:, 2]
        base_target = self.reward_cfg['base_height_target']
        
        # Detect landing phase (coming down from jump)
        is_landing = (current_height > base_target + 0.03) & (base_vel_z < -0.1)
        
        # Reward stability during landing (low angular velocities)
        ang_vel_magnitude = torch.norm(self.base_ang_vel, dim=1)
        stability = torch.exp(-ang_vel_magnitude)  # Higher reward for lower angular velocity
        
        landing_reward = torch.where(
            is_landing,
            stability * 0.5,  # Reward stability during landing
            torch.zeros_like(current_height)
        )
        
        return landing_reward

    def _reward_obstacle_avoidance(self):
        """Unified obstacle avoidance: avoid high obstacles, don't jump at them"""
        if not self.env_cfg.get('use_obstacles', False):
            return torch.zeros(self.num_envs, device=self.device)
        
        # Initialize obstacle_positions if it doesn't exist
        if not hasattr(self, 'obstacle_positions'):
            self.obstacle_positions = []
        
        if len(self.obstacle_positions) == 0:
            return torch.zeros(self.num_envs, device=self.device)
        
        low_obstacles, high_obstacles = self._detect_nearby_obstacles()
        
        # Part 1: Reward staying away from high obstacles
        distance_reward = torch.clamp(1.0 - high_obstacles, 0, 1)
        
        # Part 2: Reward lateral movement when high obstacles detected
        lateral_vel = self.base_lin_vel[:, 1]  # y-velocity
        lateral_reward = torch.where(
            high_obstacles > 0.1,
            torch.abs(lateral_vel) * 0.5,
            torch.zeros_like(high_obstacles)
        )
        
        # Part 3: Penalize jumping at high obstacles
        current_height = self.base_pos[:, 2]
        base_vel_z = self.base_lin_vel[:, 2]
        base_target = self.reward_cfg.get('base_height_target', 0.3)
        jump_height_above_normal = current_height - base_target
        is_jumping = (jump_height_above_normal > 0.05) | (base_vel_z > 0.2)
        
        jump_penalty = torch.where(
            (high_obstacles > 0.1) & is_jumping,
            -2.0 * high_obstacles,  # Strong penalty for jumping at high obstacles
            torch.zeros_like(high_obstacles)
        )
        
        return distance_reward + lateral_reward + jump_penalty

    def _randomize_obstacles_positions(self):
        """Move existing obstacles to new random positions"""
        try:
            if len(self.obstacle_entities) == 0:
                return
                
            terrain_size = self.env_cfg['terrain_size']
            clear_radius = self.env_cfg['clear_radius']
            robot_spawn = self.env_cfg['base_init_pos'][:2]
            
            print(f"Randomizing positions of {len(self.obstacle_entities)} obstacles...")
            
            new_positions = []
            
            for i, entity in enumerate(self.obstacle_entities):
                attempts = 0
                while attempts < 50:
                    # Generate new random position
                    x = np.random.uniform(-terrain_size[0]/2, terrain_size[0]/2)
                    y = np.random.uniform(-terrain_size[1]/2, terrain_size[1]/2)
                    
                    # Check distance from robot spawn
                    dist_to_spawn = np.sqrt((x - robot_spawn[0])**2 + (y - robot_spawn[1])**2)
                    if dist_to_spawn < clear_radius:
                        attempts += 1
                        continue
                    
                    # Check distance from other obstacles
                    too_close = False
                    for pos in new_positions:
                        dist = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
                        if dist < self.env_cfg['obstacle_spacing_min']:
                            too_close = True
                            break
                    
                    if not too_close:
                        # Randomize height too
                        height = np.random.uniform(*self.env_cfg['obstacle_height_range'])
                        
                        # Move obstacle to new position
                        new_pos = torch.tensor([x, y, height/2], device=self.device)
                        entity.set_pos(
                            new_pos.unsqueeze(0).repeat(self.num_envs, 1),
                            envs_idx=torch.arange(self.num_envs, device=self.device)
                        )
                        
                        new_positions.append([x, y, height])
                        break
                        
                    attempts += 1
                
                # If couldn't find good position, move far away
                if attempts >= 50:
                    far_away = torch.tensor([1000.0, 1000.0, -100.0], device=self.device)
                    entity.set_pos(
                        far_away.unsqueeze(0).repeat(self.num_envs, 1),
                        envs_idx=torch.arange(self.num_envs, device=self.device)
                    )
            
            # Update tracking
            self.obstacle_positions = new_positions
            print(f"Successfully randomized {len(new_positions)} obstacle positions")

        except Exception as e:
            print(f"Warning: Failed to move obstacles: {e}")

    def _reward_stable_walk(self):
        """Reward stability but only when moving forward"""
        # Only reward stability when robot is moving forward
        forward_velocity = self.base_lin_vel[:, 0]
        
        # If not moving forward, give no stability reward
        moving_mask = forward_velocity > 0.1  # Only when moving > 0.1 m/s
        
        # Penalize vertical velocity (galloping) only when moving
        vertical_penalty = torch.square(self.base_lin_vel[:, 2])
        height_penalty = torch.square(self.base_pos[:, 2] - self.reward_cfg['base_height_target'])
        ang_vel_penalty = torch.sum(torch.square(self.base_ang_vel), dim=1)
        
        stability_sigma = self.reward_cfg.get('stability_sigma', 0.1)
        stability_score = torch.exp(-(vertical_penalty + height_penalty + ang_vel_penalty) / stability_sigma)
        
        # Only give stability reward when moving
        return stability_score * moving_mask.float()

    def _reward_straight_walk(self):
        """Reward for walking straight (penalize sideways movement)"""
        lateral_penalty = torch.square(self.base_lin_vel[:, 1])
        straight_sigma = self.reward_cfg.get('straight_sigma', 0.1)
        return torch.exp(-lateral_penalty / straight_sigma)
    
    def _reward_forward_movement(self):
        """Heavily reward forward movement, penalize standing still"""
        forward_vel = self.base_lin_vel[:, 0]
        
        # Heavy penalty for standing still
        standing_penalty = torch.where(forward_vel < 0.1, -2.0, 0.0)
        
        # Reward forward movement
        forward_reward = torch.clamp(forward_vel, 0, 2.0)  # Cap at 2 m/s
        
        return forward_reward + standing_penalty
    
    def _detect_head_pressure(self):
        """Method 2: Height-Based Detection - Detect gentle downward pressure on head"""
        if not self.petting_enabled or not self.use_height_detection:
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # Normal target height
        target_height = self.reward_cfg.get("base_height_target", 0.35)
        current_height = self.base_pos[:, 2]
        
        # Calculate height reduction from normal
        height_reduction = target_height - current_height
        
        # Get vertical velocity (negative = being pushed down)
        vertical_velocity = self.base_lin_vel[:, 2]
        gentle_downward_vel = -vertical_velocity  # Positive = being pushed down
        
        # Conditions for gentle head pressure:
        # 1. Height slightly reduced (2-8cm below normal)
        gentle_height_reduction = (height_reduction >= self.gentle_press_range[0]) & (height_reduction <= self.gentle_press_range[1])
        
        # 2. Gentle downward velocity (being pressed down)
        gentle_downward_pressure = gentle_downward_vel > self.gentle_vel_threshold
        
        # 3. Robot relatively stationary (not jumping/falling)
        robot_speed = torch.norm(self.base_lin_vel[:, :2], dim=1)
        is_stationary = robot_speed < self.gentle_speed_threshold
        
        # 4. Not in cooldown
        not_in_cooldown = self.gesture_cooldown == 0
        
        # 5. Robot relatively stable (not tilted)
        roll_pitch_magnitude = torch.norm(self.base_euler[:, :2], dim=1)
        is_stable = roll_pitch_magnitude < 10.0  # Less than 10 degrees tilt
        
        pressure_touch = (gentle_height_reduction & 
                        gentle_downward_pressure & 
                        is_stationary & 
                        not_in_cooldown & 
                        is_stable)
        
        return pressure_touch
    
    def _check_head_petting(self):
        """Combined petting detection using both methods"""
        if not self.petting_enabled:
            return
        
        pressure_touches = self._detect_head_pressure()
        
        self.head_touched = pressure_touches

        # Start gesture timer and cooldown for touched environments
        touched_envs = self.head_touched
        self.gesture_timer = torch.where(touched_envs, 
                                    torch.full_like(self.gesture_timer, self.gesture_duration),
                                    self.gesture_timer)
        self.gesture_cooldown = torch.where(touched_envs,
                                        torch.full_like(self.gesture_cooldown, self.cooldown_duration),
                                        self.gesture_cooldown)

    def _get_petting_gesture_actions(self):
        """Generate friendly petting response gesture (tail wag + head movement)"""
        gesture_actions = torch.zeros_like(self.actions)
        
        # Create gentle wave pattern
        wave_phase = (self.gesture_timer.float() / 20.0) % (2 * 3.14159)
        wave_amplitude = 0.3
        
        # Gentle "happy" front leg movement (like excited stepping)
        gesture_actions[:, 3] = wave_amplitude * 0.5 * torch.sin(wave_phase)           # FL_hip
        gesture_actions[:, 4] = 0.8 + wave_amplitude * 0.3 * torch.sin(wave_phase)    # FL_thigh
        gesture_actions[:, 5] = -1.5 + wave_amplitude * 0.3 * torch.cos(wave_phase)   # FL_calf
        
        gesture_actions[:, 0] = wave_amplitude * 0.5 * torch.sin(wave_phase + 1.57)   # FR_hip
        gesture_actions[:, 1] = 0.8 + wave_amplitude * 0.3 * torch.sin(wave_phase + 1.57)  # FR_thigh
        gesture_actions[:, 2] = -1.5 + wave_amplitude * 0.3 * torch.cos(wave_phase + 1.57) # FR_calf
        
        # "Tail wagging" with rear legs (gentle side-to-side)
        tail_wag = wave_amplitude * 0.4 * torch.sin(wave_phase * 2)  # Faster tail movement
        gesture_actions[:, 6] = tail_wag    # RL_hip
        gesture_actions[:, 7] = 1.0         # RL_thigh (stable)
        gesture_actions[:, 8] = -1.5        # RL_calf (stable)
        
        gesture_actions[:, 9] = -tail_wag   # RR_hip (opposite direction)
        gesture_actions[:, 10] = 1.0        # RR_thigh (stable)
        gesture_actions[:, 11] = -1.5       # RR_calf (stable)
        
        return gesture_actions
    
    def _reward_petting_response(self):
        """Reward for appropriate gesture response to petting"""
        if not self.petting_enabled:
            return torch.zeros(self.num_envs, device=self.device)
        
        gesture_reward = torch.where(
            self.gesture_timer > 0,
            torch.full_like(self.head_touched, 0.2, dtype=torch.float),
            torch.zeros_like(self.head_touched, dtype=torch.float)
        )
        return gesture_reward

    def _reward_petting_stability(self):
        """Reward for maintaining stability during petting gestures"""
        if not self.petting_enabled or not (self.gesture_timer > 0).any():
            return torch.zeros(self.num_envs, device=self.device)
        
        # Reward stability during gesture
        ang_vel_magnitude = torch.norm(self.base_ang_vel, dim=1)
        height_deviation = torch.abs(self.base_pos[:, 2] - self.reward_cfg.get('base_height_target', 0.35))
        
        stability = torch.exp(-(ang_vel_magnitude + height_deviation * 5))
        
        stability_reward = torch.where(
            self.gesture_timer > 0,
            stability * 0.1,
            torch.zeros_like(stability)
        )
        
        return stability_reward