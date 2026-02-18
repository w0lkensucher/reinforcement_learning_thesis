import torch
import genesis as gs
import numpy as np
from go2_env_base import Go2BaseEnv

class Go2NavigationEnv(Go2BaseEnv):
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, wind_force=False, uneven_terrain=False, show_viewer=False, dynamic_obstacles=False):
        super().__init__(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg)
        
        self.wind_force = wind_force
        self.uneven_terrain = uneven_terrain
        self.dynamic_obstacles = dynamic_obstacles
        self.obstacle_entities = []
        self.obstacle_positions = []

        self.terrain_entities = []
        self.terrain_height_map = None

        self._setup_scene(show_viewer)

        # Additional initialization for navigation-specific features can go here
        if uneven_terrain:
            self._create_uneven_terrain()

        # add wind force field; change shape and direction?
        if wind_force:
            ff = gs.force_fields.Wind(direction=(1, 0, 0), strength=5.0, radius=10.0, center=(0, 0, 0))
            self.scene.add_force_field(ff)

        self._setup_robot()
        self._create_obstacles()
        self._setup_buffers()
        
        # optional camera
        if self.env_cfg.get("visualize_camera", True):
            self.cam = self.scene.add_camera(
                res=(1920, 1080),
                pos=(4.0, 0.0, 4.0),
                lookat=(0, 0, 1.0),
                fov=30,
                GUI=True
            )

        self._build_scene_and_setup()


    # Setup extensions
    def _setup_robot(self):
        """Setup robot with navigation-specific extensions"""
        super()._setup_robot()
        # Additional robot setup for navigation can go here
        if self.uneven_terrain:
            spawn_x, spawn_y = self.base_init_pos[0].item(), self.base_init_pos[1].item()
            terrain_height = self._get_terrain_height_at_position(spawn_x, spawn_y)
            self.base_init_pos[2] = terrain_height + 0.42  # Adjust spawn height based on terrain

            self.robot.set_pos(self.base_init_pos.cpu().numpy())


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


    # Terrain Creation
    def _create_uneven_terrain(self):
        """Create uneven terrain with various surface types"""
        terrain_type = self.env_cfg.get('terrain_type', 'heightmap')
        terrain_size = self.env_cfg.get('terrain_size', [20.0, 20.0])
        
        if terrain_type == 'heightmap':
            self._create_heightmap_terrain(terrain_size)
        elif terrain_type == 'slopes':
            self._create_slope_terrain(terrain_size)
        elif terrain_type == 'stairs':
            self._create_stairs_terrain(terrain_size)
        elif terrain_type == 'random_boxes':
            self._create_random_box_terrain(terrain_size)
        else:
            print(f"Unknown terrain type: {terrain_type}, using flat terrain")
            self._create_flat_terrain()


    def _create_heightmap_terrain(self, terrain_size):
        """Create terrain from height map (hills and valleys)"""
        # Terrain parameters
        resolution = self.env_cfg.get('terrain_resolution', 0.1)  # 10cm resolution
        height_scale = self.env_cfg.get('terrain_height_scale', 0.3)  # Max 30cm elevation
        noise_frequency = self.env_cfg.get('terrain_noise_frequency', 0.1)
        
        # Generate height map
        x_samples = int(terrain_size[0] / resolution)
        y_samples = int(terrain_size[1] / resolution)
        
        x = np.linspace(-terrain_size[0]/2, terrain_size[0]/2, x_samples)
        y = np.linspace(-terrain_size[1]/2, terrain_size[1]/2, y_samples)
        X, Y = np.meshgrid(x, y)
        
        # Generate Perlin-like noise for natural terrain
        heights = np.zeros_like(X)
        
        # Multiple octaves for realistic terrain
        for octave in range(3):
            freq = noise_frequency * (2 ** octave)
            amplitude = height_scale / (2 ** octave)
            
            heights += amplitude * np.sin(freq * X) * np.cos(freq * Y)
            heights += amplitude * 0.5 * np.sin(freq * 2 * X + 1.5) * np.cos(freq * 1.3 * Y + 0.8)
        
        # Smooth around robot spawn area
        robot_spawn = self.env_cfg.get('base_init_pos', [0.0, 0.0, 0.42])
        spawn_radius = 1.5  # 1.5m radius around spawn
        
        for i in range(x_samples):
            for j in range(y_samples):
                dist_to_spawn = np.sqrt((X[j, i] - robot_spawn[0])**2 + (Y[j, i] - robot_spawn[1])**2)
                if dist_to_spawn < spawn_radius:
                    # Smooth transition to flat around spawn
                    smooth_factor = dist_to_spawn / spawn_radius
                    heights[j, i] *= smooth_factor
        
        self.terrain_height_map = heights
        
        # Create heightfield in Genesis
        try:
            heightfield = gs.morphs.Heightfield(
                height_map=heights,
                x_range=(-terrain_size[0]/2, terrain_size[0]/2),
                y_range=(-terrain_size[1]/2, terrain_size[1]/2),
                pos=(0, 0, 0)
            )
            material = gs.materials.Rigid(friction=0.8)
            terrain_entity = self.scene.add_entity(heightfield, material=material)
            self.terrain_entities.append(terrain_entity)
            
            print(f"Created heightmap terrain: {x_samples}x{y_samples}, height range: {heights.min():.3f} to {heights.max():.3f}")
            
        except Exception as e:
            print(f"Failed to create heightmap terrain: {e}")
            print("Falling back to flat terrain")
            self._create_flat_terrain()


    def _create_slope_terrain(self, terrain_size):
        """Create terrain with slopes and ramps"""
        num_slopes = self.env_cfg.get('num_slopes', 5)
        max_slope_angle = self.env_cfg.get('max_slope_angle', 15)  # degrees
        slope_length = self.env_cfg.get('slope_length', 3.0)  # meters
        
        for i in range(num_slopes):
            # Random position (avoid robot spawn area)
            while True:
                x = np.random.uniform(-terrain_size[0]/2 + 2, terrain_size[0]/2 - 2)
                y = np.random.uniform(-terrain_size[1]/2 + 2, terrain_size[1]/2 - 2)
                robot_spawn = self.env_cfg.get('base_init_pos', [0.0, 0.0, 0.42])
                if np.sqrt(x**2 + y**2) > 2.0:  # At least 2m from spawn
                    break
            
            # Random slope parameters
            angle = np.random.uniform(-max_slope_angle, max_slope_angle) * np.pi / 180
            orientation = np.random.uniform(0, 2*np.pi)
            width = np.random.uniform(1.0, 2.0)
            
            height = slope_length * np.tan(angle)
            
            # Create slope as angled box
            slope_geom = gs.morphs.Box(
                pos=(x, y, height/2),
                size=(slope_length, width, 0.1),
                quat=gs.transforms.euler_to_quat([0, angle, orientation])
            )
            
            material = gs.materials.Rigid(friction=0.8)
            slope_entity = self.scene.add_entity(slope_geom, material=material)
            self.terrain_entities.append(slope_entity)

        # Add base plane
        self._create_flat_terrain()
        print(f"Created {num_slopes} slope terrain features")


    def _create_stairs_terrain(self, terrain_size):
        """Create terrain with stairs and steps"""
        num_staircases = self.env_cfg.get('num_staircases', 3)
        step_height = self.env_cfg.get('step_height', 0.1)  # 10cm steps
        step_depth = self.env_cfg.get('step_depth', 0.3)   # 30cm deep
        steps_per_staircase = self.env_cfg.get('steps_per_staircase', 5)
        
        for i in range(num_staircases):
            # Random position (avoid robot spawn)
            while True:
                start_x = np.random.uniform(-terrain_size[0]/2 + 3, terrain_size[0]/2 - 3)
                start_y = np.random.uniform(-terrain_size[1]/2 + 2, terrain_size[1]/2 - 2)
                if np.sqrt(start_x**2 + start_y**2) > 2.5:
                    break
            
            direction = np.random.choice(['up', 'down'])
            orientation = np.random.uniform(0, 2*np.pi)
            
            for step in range(steps_per_staircase):
                step_x = start_x + step * step_depth * np.cos(orientation)
                step_y = start_y + step * step_depth * np.sin(orientation)
                
                if direction == 'up':
                    step_z = step * step_height
                else:
                    step_z = (steps_per_staircase - step - 1) * step_height
                
                # Create step
                step_geom = gs.morphs.Box(
                    pos=(step_x, step_y, step_z + step_height/2),
                    size=(step_depth * 1.2, 0.8, step_height),
                    quat=gs.transforms.euler_to_quat([0, 0, orientation])
                )
                
                material = gs.materials.Rigid(friction=0.9)  # Higher friction for steps
                step_entity = self.scene.add_entity(step_geom, material=material)
                self.terrain_entities.append(step_entity)

        # Add base plane
        self._create_flat_terrain()
        print(f"Created {num_staircases} staircases with {steps_per_staircase} steps each")


    def _create_random_box_terrain(self, terrain_size):
        """Create terrain with random scattered boxes (rocks/debris)"""
        num_boxes = self.env_cfg.get('num_terrain_boxes', 20)
        box_size_range = self.env_cfg.get('box_size_range', [0.1, 0.4])
        box_height_range = self.env_cfg.get('box_height_range', [0.05, 0.2])
        
        for i in range(num_boxes):
            # Random position (avoid robot spawn)
            while True:
                x = np.random.uniform(-terrain_size[0]/2 + 1, terrain_size[0]/2 - 1)
                y = np.random.uniform(-terrain_size[1]/2 + 1, terrain_size[1]/2 - 1)
                if np.sqrt(x**2 + y**2) > 1.5:
                    break
            
            # Random box dimensions
            width = np.random.uniform(*box_size_range)
            length = np.random.uniform(*box_size_range)
            height = np.random.uniform(*box_height_range)
            
            # Random orientation
            orientation = np.random.uniform(0, 2*np.pi)
            
            box_geom = gs.morphs.Box(
                pos=(x, y, height/2),
                size=(length, width, height),
                quat=gs.transforms.euler_to_quat([0, 0, orientation])
            )
            
            material = gs.materials.Rigid(friction=0.7)
            box_entity = self.scene.add_entity(box_geom, material=material)
            self.terrain_entities.append(box_entity)

        # Add base plane
        self._create_flat_terrain()
        print(f"Created {num_boxes} random terrain boxes")


    def _get_terrain_height_at_position(self, x, y):
        """Get terrain height at specific position (for terrain-aware rewards)"""
        if self.terrain_height_map is None:
            return 0.0
        
        terrain_size = self.env_cfg.get('terrain_size', [20.0, 20.0])
        resolution = self.env_cfg.get('terrain_resolution', 0.1)
        
        # Convert world coordinates to height map indices
        x_idx = int((x + terrain_size[0]/2) / resolution)
        y_idx = int((y + terrain_size[1]/2) / resolution)
        
        # Bounds checking
        if (0 <= x_idx < self.terrain_height_map.shape[1] and 
            0 <= y_idx < self.terrain_height_map.shape[0]):
            return float(self.terrain_height_map[y_idx, x_idx])
        
        return 0.0
    

    # obstacle creation
    def _create_obstacles(self):
        """Create obstacles in the environment"""
        if not self.env_cfg.get('use_obstacles', False):
            print("use_obstacles is False")
            return
        try:
            if not self.dynamic_obstacles:
                terrain_size = self.env_cfg['terrain_size']
                density = self.env_cfg['obstacle_density']
                clear_radius = self.env_cfg['clear_radius']

                area = terrain_size[0] * terrain_size[1]
                num_obstacles = int(area * density)
                
                print(f"Creating {num_obstacles} obstacles...")

                for _ in range(num_obstacles):
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
                            jump_threshold = self.reward_cfg.get('jump_height_threshold', 0.08)
                            if height < jump_threshold:
                                obstacle_type = 'low'
                            else:
                                obstacle_type = 'high'

                            self.obstacle_positions.append([x, y, height, obstacle_type])
                            entity = self._add_single_obstacle(x, y, height)
                            if entity is not None:    
                                self.obstacle_entities.append(entity)
                            break

                        attempts += 1

            else:
                print("Dynamic obstacles enabled")
                # Place obstacles in front of robot spawn
                base_init_pos = self.env_cfg.get('base_init_pos', [0.0, 0.0, 0.42])
                base_yaw = self.env_cfg.get('base_init_yaw', 0.0)
                dists = [2.5, 4.5, 6.5]  # meters ahead
                for _, dist in enumerate(dists):
                    width = np.mean(self.env_cfg["obstacle_width_range"])
                    height = np.mean(self.env_cfg["obstacle_height_range"])
                    x = base_init_pos[0] + dist * np.cos(base_yaw)
                    y = base_init_pos[1] + dist * np.sin(base_yaw)
                    geom = gs.morphs.Box(
                        pos=(x, y, height / 2),
                        size=(width, width, height)
                    )
                    # High rho makes obstacles heavy and immovable during collisions
                    material = gs.materials.Rigid(friction=0.8, rho=10000.0)
                    entity = self.scene.add_entity(geom, material=material)
                    self.obstacle_entities.append(entity)
                    self.obstacle_positions.append([x, y, height, "box"])

            print(f"Successfully created {len(self.obstacle_positions)} obstacles")
        except Exception as e:
            print(f"Warning: Failed to create obstacles: {e}")
            print("Continuing without obstacles...")


    def _add_single_obstacle(self, x, y, height):
        """Add a single obstacle at position (x, y)"""
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
            # High rho makes obstacles heavy and immovable during collisions
            material = gs.materials.Rigid(friction=0.8, rho=10000.0)
            entity = self.scene.add_entity(geom, material=material)
            return entity
        except Exception as e:
            print(f"Warning: Failed to create obstacle at ({x:.2f}, {y:.2f}): {e}")


    # Observation Helpers
    def _compute_observations(self):
        """Compute observations including navigation-specific data"""
        if self.env_cfg.get('use_obstacles', False):
            low_obs, high_obs, _, _ = self._detect_nearby_obstacles()
        else:
            low_obs = torch.zeros(self.num_envs, device=self.device)
            high_obs = torch.zeros(self.num_envs, device=self.device)

        self.obs_buf = torch.cat([
        self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
        self.projected_gravity,  # 3
        self.commands * self.commands_scale,  # 3
        (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
        self.dof_vel * self.obs_scales["dof_vel"],  # 12
        self.actions,  # 12
        low_obs.unsqueeze(1),   # 1: signal for low obstacle
        high_obs.unsqueeze(1),  # 1: signal for high obstacle
    ], axis=-1)
        

    def get_observations(self):
        """Get navigation observations"""
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras


    def _detect_nearby_obstacles(self):
        """Detect obstacles near the robot and classify as jumpable or avoidable"""
        if not self.env_cfg.get('use_obstacles', False) or len(self.obstacle_positions) == 0:
                zeros = torch.zeros(self.num_envs, device=self.device)
                infs = torch.full((self.num_envs,), float('inf'), device=self.device)
                return zeros, zeros, infs, infs 
        
        robot_pos = self.base_pos[:, :2]  # x, y position
        detection_radius = 1.5  # Distance ahead to look for obstacles
        
        # For each environment, find nearest obstacle ahead
        nearest_low_signal = torch.zeros(self.num_envs, device=self.device)
        nearest_high_signal = torch.zeros(self.num_envs, device=self.device)
        closest_low_distance = torch.full((self.num_envs,), float('inf'), device=self.device)
        closest_high_distance = torch.full((self.num_envs,), float('inf'), device=self.device)

        # Get robot's forward direction
        robot_heading = self.base_radians[:, 2]  # Yaw angle
        forward_dir = torch.stack([torch.cos(robot_heading), torch.sin(robot_heading)], dim=1)
        
        for obs_data in self.obstacle_positions:
            obs_pos = torch.tensor(obs_data[:2], device=self.device, dtype=torch.float32)  # x, y
            obs_type = obs_data[3] if len(obs_data) > 3 else 'high'  # type or default

            # Vector from robot to obstacle
            to_obstacle = obs_pos.unsqueeze(0) - robot_pos  # Shape: (num_envs, 2)
            distance = torch.norm(to_obstacle, dim=1)

            # Check if obstacle is ahead (dot product with forward direction)
            is_ahead = torch.sum(to_obstacle * forward_dir, dim=1) > 0
            is_close = (distance < detection_radius) & is_ahead
            
            obstacle_signal = 1.0 / (distance + 0.1)
            # Classify obstacle by height
            if obs_type == 'low':
                # Low obstacle - should jump over
                update_mask = is_close & (distance < closest_low_distance)
                nearest_low_signal = torch.where(update_mask, obstacle_signal, nearest_low_signal)
                closest_low_distance = torch.where(update_mask, distance, closest_low_distance)
            else:  # obs_type == 'high'
                # High obstacle - should go around
                update_mask = is_close & (distance < closest_high_distance)
                nearest_high_signal = torch.where(update_mask, obstacle_signal, nearest_high_signal)
                closest_high_distance = torch.where(update_mask, distance, closest_high_distance)
        
        return nearest_low_signal, nearest_high_signal, closest_low_distance, closest_high_distance


    def _reward_height(self):
        """Petting-specific height control - allow sitting, lying"""
        base_height = self.base_pos[:, 2]

        # Normal standing posture
        tolerance = 0.1
        target_height = self.reward_cfg.get('base_height_target', 0.42)
        
        height_error = torch.abs(base_height - target_height)
        return torch.exp(-10 * torch.clamp(height_error - tolerance, min=0.0))
    
    
    def _reward_symmetry(self):
        """Reward for left-right symmetry in leg movement (encourages coordinated gait)"""
        # Actual joint order: [FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf]
        # Indices for each joint type
        FR = [0, 1, 2]
        FL = [3, 4, 5]
        RR = [6, 7, 8]
        RL = [9, 10, 11]
        dof_pos = self.dof_pos
        # Hip symmetry (front and rear)
        hip_sym = torch.abs(dof_pos[:, FL[0]] - dof_pos[:, FR[0]]) + torch.abs(dof_pos[:, RL[0]] - dof_pos[:, RR[0]])
        # Thigh symmetry
        thigh_sym = torch.abs(dof_pos[:, FL[1]] - dof_pos[:, FR[1]]) + torch.abs(dof_pos[:, RL[1]] - dof_pos[:, RR[1]])
        # Calf symmetry
        calf_sym = torch.abs(dof_pos[:, FL[2]] - dof_pos[:, FR[2]]) + torch.abs(dof_pos[:, RL[2]] - dof_pos[:, RR[2]])
        # Combine and negate (lower difference = higher reward)
        symmetry_penalty = hip_sym + thigh_sym + calf_sym
        return symmetry_penalty

    # Locomotion Rewards
    def _reward_forward_movement(self):
        """Heavily reward forward movement, penalize standing still"""
        forward_vel = self.base_lin_vel[:, 0]
        
        # Heavy penalty for standing still
        standing_penalty = torch.where(forward_vel < 0.1, -1.0, 0.0)
        
        # Reward forward movement
        forward_reward = torch.clamp(forward_vel, 0, 2.0)  # Cap at 2 m/s
        
        return forward_reward + standing_penalty
    

    def _reward_straight_walk_when_clear(self):
        """Reward walking straight when no obstacles are nearby"""
        low_obs, high_obs, low_dist, high_dist = self._detect_nearby_obstacles()
        
        # Find closest obstacle of any type
        closest_distance = torch.min(low_dist, high_dist)
        jumpable_low = (low_dist < 1.0) & (low_obs > 0.0)

        # Distance-based straight walking encouragement
        safe_distance = 2.0      # Full straight walking reward beyond this
        warning_distance = 1.0   # Start reducing reward below this
        danger_distance = 0.5    # No straight walking reward below this

        # Calculate straight walking factor (0 to 1)
        straight_factor = torch.where(
            (closest_distance >= safe_distance) | jumpable_low,
            1.0,  # Full reward when safe
            torch.where(
                closest_distance >= warning_distance,
                (closest_distance - warning_distance) / (safe_distance - warning_distance),  # Linear scaling
                torch.where(
                    closest_distance >= danger_distance,
                    0.1,  # Small reward in warning zone
                    0.0   # No reward in danger zone
                )
            )
        )
        
        # Reward components
        lateral_velocity = torch.abs(self.base_lin_vel[:, 1])
        angular_velocity = torch.abs(self.base_ang_vel[:, 2])
        forward_velocity = torch.clamp(self.base_lin_vel[:, 0], 0, 2.0)
        
        # Apply distance-based scaling
        straight_reward = (
            forward_velocity * 0.5 * straight_factor -      # Reward forward movement when safe
            lateral_velocity * 2.0 * straight_factor -      # Penalize lateral movement when safe  
            angular_velocity * 1.5 * straight_factor        # Penalize turning when safe
        )
        
        return straight_reward
    

    def _reward_jump_clearance(self):
        """Reward for actively jumping over obstacles - with bootstrapping for easier discovery"""
        if len(self.obstacle_positions) == 0:
            return torch.zeros(self.num_envs, device=self.device)
        
        robot_pos = self.base_pos[:, :2]
        current_height = self.base_pos[:, 2]
        vertical_velocity = self.base_lin_vel[:, 2]
        normal_height = 0.42
        
        # Calculate height above normal
        height_above_normal = current_height - normal_height
        
        jump_reward = torch.zeros(self.num_envs, device=self.device)
        
        for obs_data in self.obstacle_positions:
            obs_pos = torch.tensor(obs_data[:2], device=self.device, dtype=torch.float32)
            obs_height = obs_data[2]
            
            # Distance to obstacle
            to_obstacle = obs_pos.unsqueeze(0) - robot_pos
            distance = torch.norm(to_obstacle, dim=1)
            
            # WIDER detection zone for easier discovery (1.5m → 2.5m)
            near_obstacle = distance < 2.5
            approaching = distance < 2.0  # Within 2m
            
            # Multi-tiered reward system for bootstrapping:
            
            # Tier 1: Reward upward velocity when approaching obstacle (easiest to discover)
            upward_velocity_reward = torch.clamp(vertical_velocity, 0, 1.0) * 0.3  # Max 0.3
            tier1 = torch.where(approaching & (vertical_velocity > 0.1), upward_velocity_reward, torch.zeros_like(upward_velocity_reward))
            
            # Tier 2: Reward being airborne near obstacle (LOWERED threshold: 5cm → 2cm)
            is_airborne = height_above_normal > 0.02  # Much easier to trigger
            airborne_reward = torch.clamp(height_above_normal * 5.0, 0, 1.0)  # Scale height to reward
            tier2 = torch.where(near_obstacle & is_airborne, airborne_reward, torch.zeros_like(airborne_reward))
            
            # Tier 3: Bonus for clearing the obstacle height
            clearance_height = torch.clamp(height_above_normal - obs_height, 0, 0.2)
            clearance_reward = (clearance_height / 0.2) * 2.0  # Max 2.0 (increased from 1.0)
            tier3 = torch.where(near_obstacle & (height_above_normal > obs_height), clearance_reward, torch.zeros_like(clearance_reward))
            
            # Combine all tiers
            jump_reward += tier1 + tier2 + tier3
        
        return jump_reward
    
    def _reward_landing_stability(self):
        """Landing stability reward - easier to trigger during and after jumping"""
        # Return zero if no obstacles
        if len(self.obstacle_positions) == 0:
            return torch.zeros(self.num_envs, device=self.device)
        
        # Detect when robot is airborne or recently landed
        normal_height = None
        
        if self.uneven_terrain:
            # Get terrain-aware normal height for each environment
            normal_heights = []
            for env_idx in range(self.num_envs):
                robot_x = self.base_pos[env_idx, 0].item()
                robot_y = self.base_pos[env_idx, 1].item()
                terrain_height = self._get_terrain_height_at_position(robot_x, robot_y)
                normal_heights.append(terrain_height + 0.42)
            normal_height = torch.tensor(normal_heights, device=self.device)
        else:
            normal_height = 0.42
        
        current_height = self.base_pos[:, 2]
        vertical_velocity = self.base_lin_vel[:, 2]
        
        # EXPANDED landing detection (easier to trigger):
        # Phase 1: Airborne and descending (LOWERED threshold: 5cm → 2cm)
        is_descending = (current_height > normal_height + 0.02) & (vertical_velocity < 0)
        
        # Phase 2: Recently landed (near ground with low vertical velocity)
        is_grounded = (current_height < normal_height + 0.08) & (torch.abs(vertical_velocity) < 0.3)
        
        # Combined: reward during descent AND after landing
        in_landing_phase = is_descending | is_grounded
        
        # Check if robot is near any obstacle (WIDENED: 1.0m → 2.5m)
        robot_pos = self.base_pos[:, :2]
        near_obstacle = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        for obs_data in self.obstacle_positions:
            obs_pos = torch.tensor(obs_data[:2], device=self.device, dtype=torch.float32)
            
            # Distance to obstacle
            to_obstacle = obs_pos.unsqueeze(0) - robot_pos
            distance = torch.norm(to_obstacle, dim=1)
            
            # WIDENED proximity range (1.0m → 2.5m) - catches landing after clearing
            near_obstacle |= (distance < 2.5)
        
        # Reward stability when in landing phase near obstacles
        is_jump_context = in_landing_phase & near_obstacle
        
        # Reward low angular velocity (stability) and low vertical oscillation
        ang_vel_penalty = torch.norm(self.base_ang_vel, dim=1)
        stability_score = torch.exp(-ang_vel_penalty * 2.0)  # More sensitive to rotation
        
        # Scale reward: 0.3 during descent, 0.5 when grounded (encourage settling)
        reward_scale = torch.where(is_grounded, 0.5, 0.3)
        
        return torch.where(is_jump_context, stability_score * reward_scale, torch.zeros_like(current_height))


    def _reward_obstacle_avoidance(self):
        """Reward for proper obstacle handling"""
        if len(self.obstacle_positions) == 0:
            return torch.zeros(self.num_envs, device=self.device)
        
        robot_pos = self.base_pos[:, :2]
        collision_penalty = torch.zeros(self.num_envs, device=self.device)
        
        for obs_data in self.obstacle_positions:
            obs_pos = torch.tensor(obs_data[:2], device=self.device, dtype=torch.float32)
            obs_type = obs_data[3]
            
            # Distance to obstacle
            to_obstacle = obs_pos.unsqueeze(0) - robot_pos
            distance = torch.norm(to_obstacle, dim=1)
            
            # Different safety margins for different obstacle types
            if obs_type == 'low':
                safety_margin = 0.3  # Can get closer to low obstacles (jump over)
                collision_threshold = 0.1
            else:  # 'high'
                safety_margin = 0.8  # Must maintain distance from high obstacles
                collision_threshold = 0.3
            
            # Penalties based on obstacle type
            collision_mask = distance < collision_threshold
            close_mask = (distance < safety_margin) & ~collision_mask
            
            collision_penalty += torch.where(collision_mask, -1.0, 0.0)
            
            collision_penalty += torch.where(close_mask, -0.1, 0.0)
        
        return collision_penalty
    

    def _check_termination(self):
        """Override base termination to include collision detection"""
        # Call base class termination checks (pitch, roll, height, time)
        super()._check_termination()
        
        # Add collision termination for navigation (if enabled)
        if self.env_cfg.get('terminate_on_collision', False) and len(self.obstacle_positions) > 0:
            robot_pos = self.base_pos[:, :2]
            
            for obs_data in self.obstacle_positions:
                obs_pos = torch.tensor(obs_data[:2], device=self.device, dtype=torch.float32)
                
                # Distance to obstacle
                to_obstacle = obs_pos.unsqueeze(0) - robot_pos
                distance = torch.norm(to_obstacle, dim=1)
                
                # Collision threshold - terminate if robot touches obstacle
                collision_threshold = self.env_cfg.get('collision_threshold', 0.25)
                collision_mask = distance < collision_threshold
                
                # Set reset buffer for collided environments
                self.reset_buf |= collision_mask
    

    def _reposition_dynamic_obstacles(self):
        """Reposition dynamic obstacles in front of robot when passed during training"""
        if len(self.obstacle_entities) < 3:
            return
            
        for env_idx in range(self.num_envs):
            base_pos = self.base_pos[env_idx].cpu().numpy()
            base_yaw = self.base_radians[env_idx, 2].cpu().item()
            
            # Use velocity direction for trajectory prediction if robot is moving
            base_vel = self.base_lin_vel[env_idx, :2].cpu().numpy()  # X, Y velocity
            vel_magnitude = np.linalg.norm(base_vel)
            
            # If robot is moving, use velocity direction; otherwise use facing direction
            if vel_magnitude > 0.1:  # Moving threshold
                trajectory_direction = base_vel / vel_magnitude
            else:
                trajectory_direction = np.array([np.cos(base_yaw), np.sin(base_yaw)])
            
            # Helper to check if robot has passed an obstacle
            def has_passed(obs_pos):
                rel = np.array([obs_pos[0] - base_pos[0], obs_pos[1] - base_pos[1]])
                return np.dot(rel, trajectory_direction) < -0.5  # Negative means behind robot
            
            # Distances for obstacles
            dists = [2.5, 4.5, 6.5]
            clear_radius = self.env_cfg.get('clear_radius', 1.0)
            
            # First check if the last obstacle (index 2) has been passed
            last_obs_qpos = self.obstacle_entities[2].get_qpos()
            if hasattr(last_obs_qpos, 'cpu'):
                last_obs_qpos = last_obs_qpos.cpu().numpy()
            
            if last_obs_qpos.ndim > 1 and len(last_obs_qpos) > env_idx:
                last_obs_pos = last_obs_qpos[env_idx].flatten()[:2]
            else:
                last_obs_pos = last_obs_qpos.flatten()[:2]
            
            # Only reposition if last obstacle has been passed
            if not has_passed(last_obs_pos):
                continue
            
            # Now find and reposition the first obstacle that's passed
            for i, dist in enumerate(dists):
                if i >= len(self.obstacle_entities):
                    break
                
                # Get obstacle position
                obs_qpos = self.obstacle_entities[i].get_qpos()
                if hasattr(obs_qpos, 'cpu'):
                    obs_qpos = obs_qpos.cpu().numpy()
                
                # Handle multi-env case - get position for this environment
                if obs_qpos.ndim > 1 and len(obs_qpos) > env_idx:
                    obs_pos = obs_qpos[env_idx].flatten()[:2]
                else:
                    obs_pos = obs_qpos.flatten()[:2]
                
                if has_passed(obs_pos):
                    # Place obstacle ahead along predicted trajectory
                    x = base_pos[0] + dist * trajectory_direction[0]
                    y = base_pos[1] + dist * trajectory_direction[1]
                    z = base_pos[2] + 0.1
                    
                    dist_to_robot = np.sqrt((x - base_pos[0])**2 + (y - base_pos[1])**2)
                    if dist_to_robot < clear_radius:
                        # Place farther if too close
                        offset = clear_radius + 0.5
                        x = base_pos[0] + (dist + offset) * trajectory_direction[0]
                        y = base_pos[1] + (dist + offset) * trajectory_direction[1]
                    
                    # Set new position for this specific environment
                    new_pos = torch.tensor([x, y, z], device=self.device, dtype=torch.float32)
                    self.obstacle_entities[i].set_pos(
                        new_pos.unsqueeze(0),
                        envs_idx=torch.tensor([env_idx], device=self.device)
                    )
                    
                    # Update tracking for this environment
                    height = self.env_cfg.get('obstacle_height_range', [0.05, 0.15])
                    height = np.mean(height)
                    self.obstacle_positions[i] = [x, y, height, "box"]
                    break  # Only reposition the first passed obstacle per step


    # Computation Helpers
    def _compute_rewards(self):
        """Navigation-specific reward computation"""
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        self.episode_sums['reward'] += self.rew_buf

    def step(self, actions):
        """Navigation-specific step logic"""
        self._execute_actions(actions)

        self._update_robot_state()

        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )
        self._resample_commands(envs_idx)

        self._check_termination()

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape(-1))

        # Dynamic obstacle repositioning during training
        if self.dynamic_obstacles:
            self._reposition_dynamic_obstacles()

        self._compute_rewards()

        self._compute_observations()

        self.last_actions[:] = actions
        self.last_dof_vel[:] = self.dof_vel

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=torch.float)
        self.extras["time_outs"][time_out_idx] = 1.0

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras