import torch
import math
import genesis as gs
import numpy as np
from go2_env_base import Go2BaseEnv

class Go2NavigationEnv(Go2BaseEnv):
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, wind_force=False, uneven_terrain=False):
        super().__init__(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg)
        
        self.wind_force = wind_force
        self.uneven_terrain = uneven_terrain
        self.obstacle_entities = []
        self.obstacle_positions = []

        self.terrain_entities = []
        self.terrain_height_map = None

        # Additional initialization for navigation-specific features can go here
        if uneven_terrain:
            self._create_uneven_terrain()

        # add wind force field; change shape and direction?
        if wind_force:
            ff = gs.force_fields.Wind(direction=(1, 0, 0), strength=5.0, radius=10.0, center=(0, 0, 0))
            self.scene.add_force_field(ff)

        self._setup_robot()
        

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


    # Locomotion Rewards
    def _reward_tracking_lin_vel(self):
        """Track commanded linear velocity - shared across all locomotion tasks"""
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])


    def _reward_tracking_ang_vel(self):
        """Track commanded angular velocity - shared across all locomotion tasks"""
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])