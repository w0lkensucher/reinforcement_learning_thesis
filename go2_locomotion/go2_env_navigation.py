import torch
import genesis as gs
import numpy as np
from go2_env_base import Go2BaseEnv

class Go2NavigationEnv(Go2BaseEnv):
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, wind_force=False, uneven_terrain=False, show_viewer=False):
        super().__init__(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg)
        
        self.wind_force = wind_force
        self.uneven_terrain = uneven_terrain
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
                # res=(960, 540),
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
            return
        try:
            terrain_size = self.env_cfg['terrain_size']
            density = self.env_cfg['obstacle_density']
            clear_radius = self.env_cfg['clear_radius']

            area = terrain_size[0] * terrain_size[1]
            num_obstacles = int(area * density)
            
            print(f"Creating {num_obstacles} obstacles...")

            self.obstacle_positions = []
            self.obstacle_entities = []

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
            material = gs.materials.Rigid(friction=0.8)
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


    # Locomotion Rewards
    def _reward_tracking_lin_vel(self):
        """Track commanded linear velocity with direct error penalty"""
        # Calculate velocity error
        vel_error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        error_magnitude = torch.norm(vel_error, dim=1)
        
        # Direct penalty
        tolerance = 0.2  # 0.2 m/s tolerance
        penalty = torch.where(error_magnitude > tolerance,
                            -(error_magnitude - tolerance) * 3.0,  # Linear penalty beyond tolerance
                            torch.zeros_like(error_magnitude))     # No penalty within tolerance
        
        return penalty


    def _reward_tracking_ang_vel(self):
        """Track commanded angular velocity with direct error penalty"""
        # Calculate angular velocity error
        ang_vel_error = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2])
        
        # Direct penalty
        tolerance = 0.3  # 0.3 rad/s tolerance
        penalty = torch.where(ang_vel_error > tolerance,
                            -(ang_vel_error - tolerance) * 2.0,    # Linear penalty beyond tolerance
                            torch.zeros_like(ang_vel_error))      # No penalty within tolerance
        
        return penalty
    

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

    def _reward_adaptive_base_height(self):
        """Maintain appropriate base height with terrain and obstacle awareness"""
        base_height = self.base_pos[:, 2]
        
        # if self.uneven_terrain:
        #     # Adaptive height based on local terrain
        #     target_heights = []
        #     for env_idx in range(self.num_envs):
        #         robot_x = self.base_pos[env_idx, 0].item()
        #         robot_y = self.base_pos[env_idx, 1].item()
        #         terrain_height = self._get_terrain_height_at_position(robot_x, robot_y)
        #         target_heights.append(terrain_height + 0.42)  # 42cm above terrain
        #     target_height = torch.tensor(target_heights, device=self.device)
        # else:
            # Fixed height for flat terrain

        target_height = self.reward_cfg["base_height_target"]
        
        # # Base height error
        # height_error = torch.abs(base_height - target_height)
        
        # # Allow more height variation during jumping over low obstacles
        # low_obs, _ = self._detect_nearby_obstacles()[:2]
        
        # # More lenient height control when jumping
        # is_jumping = (self.base_lin_vel[:, 2] > 0.1) & (low_obs > 0.1)
        # tolerance = torch.where(is_jumping, 0.3, 0.1)  # 30cm tolerance when jumping, 10cm normally
        
        # # Scaled penalty - less penalty within tolerance
        # height_penalty = torch.where(height_error < tolerance, 
        #                             height_error * 0.5,  # Gentle penalty within tolerance
        #                             height_error * 2.0)  # Stronger penalty outside tolerance
        
        # bonus = (height_error < 0.02).float() * 0.1  # Small bonus for being very close to target height
        
        # return -height_penalty + bonus

        return torch.square(base_height - target_height)
    

    def _reward_landing_stability(self):
        """Landing stability reward"""
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
        
        # Simple landing detection: above normal height with downward velocity
        is_landing = (current_height > normal_height + 0.05) & (vertical_velocity < 0)
        
        # Reward low angular velocity during landing
        ang_vel_penalty = torch.norm(self.base_ang_vel, dim=1)
        stability_reward = torch.exp(-ang_vel_penalty)
        
        return torch.where(is_landing, stability_reward * 0.3, torch.zeros_like(current_height))


    def _reward_jumping_behavior(self):
        """Reward for appropriate jumping over low obstacles"""
        if len(self.obstacle_positions) == 0:
            return torch.zeros(self.num_envs, device=self.device)
        
        robot_pos = self.base_pos
        jumping_reward = torch.zeros(self.num_envs, device=self.device)
        
        # Check if robot is airborne (z-velocity > 0 and z-position > normal)
        is_jumping = (self.base_lin_vel[:, 2] > 0.1) & (robot_pos[:, 2] > 0.5)
        
        if is_jumping.any():
            # Check if there's a low obstacle nearby that justifies jumping
            for obs_data in self.obstacle_positions:
                if obs_data[3] == 'low':  # Only for low obstacles
                    obs_pos = torch.tensor(obs_data[:2], device=self.device, dtype=torch.float32)
                    
                    to_obstacle = obs_pos.unsqueeze(0) - robot_pos[:, :2]
                    distance = torch.norm(to_obstacle, dim=1)
                    
                    # Reward jumping over low obstacles
                    near_low_obstacle = distance < 1.0
                    jumping_reward += torch.where(is_jumping & near_low_obstacle, 5.0, 0.0)
        
        return jumping_reward


    def _reward_orientation_stability(self):
        """Penalize excessive roll and pitch to prevent falling"""
        # Use radians for calculation
        roll = torch.abs(self.base_radians[:, 0])    # Roll angle
        pitch = torch.abs(self.base_radians[:, 1])   # Pitch angle
        
        # Get termination thresholds (need to convert from degrees to radians)
        max_roll = self.env_cfg["termination_if_roll_greater_than"] * torch.pi / 180.0
        max_pitch = self.env_cfg["termination_if_pitch_greater_than"] * torch.pi / 180.0
        
        # Progressive penalty as robot approaches falling
        roll_penalty = torch.where(roll > max_roll * 0.5,  # Start penalty at 50% of termination threshold
                                -(roll / max_roll) * 1.5,  # Scale penalty by how close to falling
                                torch.zeros_like(roll))
        
        pitch_penalty = torch.where(pitch > max_pitch * 0.5,
                                -(pitch / max_pitch) * 1.5,
                                torch.zeros_like(pitch))
        
        return roll_penalty + pitch_penalty


    def _reward_angular_velocity_stability(self):
        """Penalize excessive angular velocities that lead to falling"""
        # Penalize high angular velocities in roll and pitch
        roll_vel = torch.abs(self.base_ang_vel[:, 0])
        pitch_vel = torch.abs(self.base_ang_vel[:, 1])
        
        # Moderate angular velocity is OK, but excessive is dangerous
        max_safe_ang_vel = 2.0  # rad/s
        
        roll_vel_penalty = torch.where(roll_vel > max_safe_ang_vel,
                                    torch.square(roll_vel - max_safe_ang_vel),
                                    torch.zeros_like(roll_vel))
        
        pitch_vel_penalty = torch.where(pitch_vel > max_safe_ang_vel,
                                    torch.square(pitch_vel - max_safe_ang_vel),
                                    torch.zeros_like(pitch_vel))
        
        return roll_vel_penalty + pitch_vel_penalty


    def _reward_upright_posture(self):
        """Small reward for maintaining upright posture"""
        # Small positive reward for maintaining good orientation
        roll = torch.abs(self.base_radians[:, 0])
        pitch = torch.abs(self.base_radians[:, 1])
        
        # Convert termination thresholds to radians
        max_roll = self.env_cfg["termination_if_roll_greater_than"] * torch.pi / 180.0
        max_pitch = self.env_cfg["termination_if_pitch_greater_than"] * torch.pi / 180.0
        
        # Exponential reward for staying upright
        roll_reward = torch.exp(-roll * 3.0 / max_roll)
        pitch_reward = torch.exp(-pitch * 3.0 / max_pitch)
        
        return (roll_reward + pitch_reward) * 0.1  # Small positive reward


    # Navigation Rewards
    def _reward_ground_clearance(self):
        """Ensure minimum ground clearance for safe navigation"""
        min_clearance = 0.25  # 25cm minimum
        current_height = self.base_pos[:, 2]
        clearance_violation = torch.clamp(min_clearance - current_height, min=0)
        return -clearance_violation * 10.0  # Strong penalty for dragging
    

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

        self._compute_rewards()

        self._compute_observations()

        self.last_actions[:] = actions
        self.last_dof_vel[:] = self.dof_vel

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=torch.float)
        self.extras["time_outs"][time_out_idx] = 1.0

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras