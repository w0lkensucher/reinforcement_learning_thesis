import torch
import math
import genesis as gs
import numpy as np
from go2_env_base import Go2BaseEnv

class Go2NavigationEnv(Go2BaseEnv):
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, wind_force=False, uneven_terrain=False):
        super().__init__(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg)
        
        # Additional initialization for navigation-specific features can go here
        if uneven_terrain:
            self._create_uneven_terrain()

        # add wind force field; make it conditional later if needed
        if wind_force:
            ff = gs.force_fields.Wind(direction=(1, 0, 0), strength=5.0, radius=10.0, center=(0, 0, 0))
            self.scene.add_force_field(ff)


    # Navigation-specific methods can be added here

    
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