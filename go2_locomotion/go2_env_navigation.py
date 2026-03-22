import torch
import genesis as gs
import numpy as np
import csv
import os
from go2_env_base import Go2BaseEnv

# Shared goal radius for all goal-reaching checks (reward and termination)
GOAL_TERMINATION_RADIUS = 0.7  # meters

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

        # Track strict curriculum success per environment (one credit per episode).
        self.jump_success_armed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.jump_success_counted = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # Track attempts to reach next curriculum level
        self.jump_attempts_at_level = 0
        self.jump_attempts_per_env = torch.zeros(self.num_envs, device=self.device)  # Steps per environment
        self.curriculum_advance_metrics = {}  # Store metrics for external logging
        
        # Stage 3 avoidance tracking
        self.avoidance_goals_reached = 0
        self.avoidance_episodes_total = 0
        self.avoidance_collisions = 0
        self.avoidance_avg_distance = 0.0
        self.avoidance_episode_costs = []  # Track cost per episode
        self.episode_had_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # Per-env collision flag
        
        # CSV logging for curriculum progress
        self.curriculum_csv_file = None
        self.curriculum_csv_writer = None
        self.avoidance_csv_file = None
        self.avoidance_csv_writer = None
        self._init_curriculum_logger()
        
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
    

    def _init_curriculum_logger(self):
        """Initialize CSV loggers for curriculum advancement and stage 3 avoidance tracking"""
        # Get log directory from env_cfg (passed from training script)
        log_dir = self.env_cfg.get('log_dir')
        
        # Fallback to environment variable or default
        if not log_dir:
            log_dir = os.environ.get('RL_LOG_DIR', './logs')
        
        # Ensure directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Create curriculum metrics CSV path (stage 4)
        csv_path = os.path.join(log_dir, 'curriculum_progress.csv')
        
        try:
            self.curriculum_csv_file = open(csv_path, 'w', newline='')
            self.curriculum_csv_writer = csv.DictWriter(
                self.curriculum_csv_file,
                fieldnames=['iteration', 'level', 'prev_height_cm', 'new_height_cm', 'total_attempts', 'avg_attempts_per_env', 'successful_jumps']
            )
            self.curriculum_csv_writer.writeheader()
            self.curriculum_csv_file.flush()
            print(f"📊 Curriculum logger initialized: {csv_path}")
        except Exception as e:
            print(f"Warning: Could not initialize curriculum logger: {e}")
            self.curriculum_csv_writer = None
        
        # Create avoidance metrics CSV path (stage 3)
        avoidance_csv_path = os.path.join(log_dir, 'avoidance_progress.csv')
        try:
            self.avoidance_csv_file = open(avoidance_csv_path, 'w', newline='')
            self.avoidance_csv_writer = csv.DictWriter(
                self.avoidance_csv_file,
                fieldnames=['iteration', 'total_episodes', 'goals_reached', 'reach_success_rate', 'collisions', 'collision_rate', 'avg_distance_to_goal']
            )
            self.avoidance_csv_writer.writeheader()
            self.avoidance_csv_file.flush()
            print(f"📊 Avoidance logger initialized: {avoidance_csv_path}")
        except Exception as e:
            print(f"Warning: Could not initialize avoidance logger: {e}")
            self.avoidance_csv_writer = None
    
    def _log_curriculum_advancement(self):
        """Log curriculum advancement to CSV"""
        if self.curriculum_csv_writer is None or not self.curriculum_advance_metrics:
            return
        
        try:
            metrics = self.curriculum_advance_metrics
            self.curriculum_csv_writer.writerow({
                'iteration': getattr(self, 'global_iteration', 0),
                'level': metrics.get('curriculum_level', 0),
                'prev_height_cm': metrics.get('prev_height_cm', 0),
                'new_height_cm': metrics.get('new_height_cm', 0),
                'total_attempts': metrics.get('total_attempts', 0),
                'avg_attempts_per_env': f"{metrics.get('avg_attempts_per_env', 0):.1f}",
                'successful_jumps': metrics.get('successful_jumps', 0),
            })
            self.curriculum_csv_file.flush()
        except Exception as e:
            print(f"Warning: Could not write curriculum metrics: {e}")

    def _log_avoidance_metrics(self, iteration):
        """Log stage 3 avoidance metrics to CSV periodically"""
        if self.avoidance_csv_writer is None or self.avoidance_episodes_total == 0:
            return
        
        try:
            success_rate = (self.avoidance_goals_reached / self.avoidance_episodes_total) * 100 if self.avoidance_episodes_total > 0 else 0
            collision_rate = (self.avoidance_collisions / self.avoidance_episodes_total) * 100 if self.avoidance_episodes_total > 0 else 0
            
            self.avoidance_csv_writer.writerow({
                'iteration': iteration,
                'total_episodes': self.avoidance_episodes_total,
                'goals_reached': self.avoidance_goals_reached,
                'reach_success_rate': f"{success_rate:.1f}%",
                'collisions': self.avoidance_collisions,
                'collision_rate': f"{collision_rate:.1f}%",
                'avg_distance_to_goal': f"{self.avoidance_avg_distance:.3f}",
            })
            self.avoidance_csv_file.flush()
            print(f"📊 Avoidance metrics: {success_rate:.1f}% reach rate | {collision_rate:.1f}% collision rate | Avg dist: {self.avoidance_avg_distance:.3f}m")
        except Exception as e:
            print(f"Warning: Could not write avoidance metrics: {e}")

    def __del__(self):
        """Clean up CSV files on deletion"""
        if hasattr(self, 'curriculum_csv_file') and self.curriculum_csv_file:
            try:
                self.curriculum_csv_file.close()
            except:
                pass
        if hasattr(self, 'avoidance_csv_file') and self.avoidance_csv_file:
            try:
                self.avoidance_csv_file.close()
            except:
                pass
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
                base_init_pos = self.env_cfg.get('base_init_pos', [0.0, 0.0, 0.42])
                base_yaw = self.env_cfg.get('base_init_yaw', 0.0)

                obstacle_width = self.env_cfg.get('dynamic_obstacle_width', 0.4)
                obstacle_distance = self.env_cfg.get('dynamic_obstacle_distance', 2.0)

                # ── Height curriculum ────────────────────────────────────────────────
                # Each level gets its own Box entity of the correct size, created before
                # scene.build(). Only the active level sits in front of the robot; all
                # others are parked at (1000, 1000, h/2) where they rest on the ground
                # harmlessly. Advancing the curriculum swaps positions via set_pos().
                start_h  = self.env_cfg.get('jump_curriculum_start_height', 0.05)
                target_h = self.env_cfg.get('dynamic_obstacle_height', 0.15)
                step_h   = self.env_cfg.get('jump_curriculum_step', 0.025)

                levels = []
                h = start_h
                while h < target_h - 1e-6:
                    levels.append(round(h, 4))
                    h += step_h
                levels.append(target_h)   # always include exact target

                self.curriculum_heights   = levels
                self.curriculum_level     = 0
                self.jump_curriculum_height = levels[0]
                self.jump_success_count   = 0
                self.jump_success_threshold = self.env_cfg.get('jump_success_threshold', 200)
                self.curriculum_entities  = []   # one entity per level

                jump_threshold = self.reward_cfg.get('jump_height_threshold', 0.2)

                x1 = base_init_pos[0] + obstacle_distance * np.cos(base_yaw)
                y1 = base_init_pos[1] + obstacle_distance * np.sin(base_yaw)

                for i, lh in enumerate(levels):
                    active = (i == 0)
                    px = x1 if active else 1000.0
                    py = y1 if active else 1000.0
                    color = (1.0, 1.0, 0.0, 1.0) if lh < jump_threshold else (1.0, 0.5, 0.0, 1.0)
                    obs_type = "low" if lh < jump_threshold else "high"

                    geom = gs.morphs.Box(
                        pos=(px, py, lh / 2),
                        size=(obstacle_width, obstacle_width, lh)
                    )
                    surface  = gs.surfaces.Default(color=color)
                    material = gs.materials.Rigid(friction=0.8, rho=10000.0)
                    entity   = self.scene.add_entity(geom, material=material, surface=surface)
                    self.curriculum_entities.append(entity)

                    if active:
                        self.obstacle_entities.append(entity)
                        self.obstacle_positions.append([x1, y1, lh, obs_type])

                print(f"Curriculum heights: {[f'{lh*100:.1f}cm' for lh in levels]}")

                # ── Target cylinder (green goal object) ──────────────────────────────
                dist2   = obstacle_distance + 5.0
                width2  = 0.25
                height2 = 0.15
                x2 = base_init_pos[0] + dist2 * np.cos(base_yaw)
                y2 = base_init_pos[1] + dist2 * np.sin(base_yaw)
                geom2 = gs.morphs.Cylinder(
                    pos=(x2, y2, height2 / 2),
                    radius=width2 / 2,
                    height=height2
                )
                surface2  = gs.surfaces.Default(color=(0.0, 1.0, 0.0, 1.0))
                material2 = gs.materials.Rigid(friction=0.8, rho=10000.0)
                entity2   = self.scene.add_entity(geom2, material=material2, surface=surface2)
                self.obstacle_entities.append(entity2)
                self.obstacle_positions.append([x2, y2, height2, "cylinder"])

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

        # Goal direction observation (only in dynamic goal-reaching mode)
        if self.dynamic_obstacles and len(self.obstacle_positions) == 2:
            target_pos = torch.tensor(self.obstacle_positions[1][:2], device=self.device, dtype=torch.float32)
            blocker_pos = torch.tensor(self.obstacle_positions[0][:2], device=self.device, dtype=torch.float32)
            robot_pos = self.base_pos[:, :2]
            robot_yaw = self.base_radians[:, 2]
            cos_yaw = torch.cos(robot_yaw)
            sin_yaw = torch.sin(robot_yaw)

            # Goal relative observations
            to_goal = target_pos.unsqueeze(0) - robot_pos  # (N, 2)
            goal_dist = torch.norm(to_goal, dim=1, keepdim=True).clamp(min=0.1)
            to_goal_dir = to_goal / goal_dist
            goal_cos = to_goal_dir[:, 0] * cos_yaw + to_goal_dir[:, 1] * sin_yaw
            goal_sin = -to_goal_dir[:, 0] * sin_yaw + to_goal_dir[:, 1] * cos_yaw
            goal_dist_norm = (goal_dist.squeeze(1) / 15.0).clamp(0.0, 1.0)

            # Blocker relative observations (same encoding as goal)
            to_blocker = blocker_pos.unsqueeze(0) - robot_pos  # (N, 2)
            blocker_dist = torch.norm(to_blocker, dim=1, keepdim=True).clamp(min=0.1)
            to_blocker_dir = to_blocker / blocker_dist
            blocker_cos = to_blocker_dir[:, 0] * cos_yaw + to_blocker_dir[:, 1] * sin_yaw  # 1=facing blocker
            blocker_sin = -to_blocker_dir[:, 0] * sin_yaw + to_blocker_dir[:, 1] * cos_yaw  # +ve=blocker left, -ve=right
            blocker_dist_norm = (blocker_dist.squeeze(1) / 8.0).clamp(0.0, 1.0)  # normalize by max ~8m
        else:
            goal_cos = torch.zeros(self.num_envs, device=self.device)
            goal_sin = torch.zeros(self.num_envs, device=self.device)
            goal_dist_norm = torch.zeros(self.num_envs, device=self.device)
            blocker_cos = torch.zeros(self.num_envs, device=self.device)
            blocker_sin = torch.zeros(self.num_envs, device=self.device)
            blocker_dist_norm = torch.zeros(self.num_envs, device=self.device)

        self.obs_buf = torch.cat([
            self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
            self.projected_gravity,  # 3
            self.commands * self.commands_scale,  # 3
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
            self.dof_vel * self.obs_scales["dof_vel"],  # 12
            self.actions,  # 12
            low_obs.unsqueeze(1),          # 1: signal for low obstacle
            high_obs.unsqueeze(1),         # 1: signal for high obstacle
            goal_cos.unsqueeze(1),         # 1: cos of heading error to goal
            goal_sin.unsqueeze(1),         # 1: sin of heading error to goal
            goal_dist_norm.unsqueeze(1),   # 1: normalized distance to goal
            blocker_cos.unsqueeze(1),      # 1: cos of heading error to blocker
            blocker_sin.unsqueeze(1),      # 1: sin of heading error to blocker (+ve=left, -ve=right)
            blocker_dist_norm.unsqueeze(1),# 1: normalized distance to blocker
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


    def _reward_keep_moving(self):
        """Penalize being stationary using total XY speed - for navigation stages where
        lateral avoidance maneuvers are needed and forward-only check would be unfair."""
        xy_speed = torch.norm(self.base_lin_vel[:, :2], dim=1)
        standing_penalty = torch.where(xy_speed < 0.1, -1.0, 0.0)
        movement_reward = torch.clamp(xy_speed, 0, 2.0)
        return movement_reward + standing_penalty


    def _reward_blocker_clearance(self):
        """Reward for navigating past the blocker obstacle in X.
        Fills the reward-dead zone during the lateral avoidance maneuver:
        fires continuously once the robot's X exceeds the blocker's X,
        giving a gradient to push through the avoidance rather than retreat."""
        if not self.dynamic_obstacles or len(self.obstacle_positions) != 2:
            return torch.zeros(self.num_envs, device=self.device)

        # Blocker is obstacle 0, always placed along +X (base_init_yaw=0)
        blocker_x = self.obstacle_positions[0][0]  # X coordinate of blocker
        robot_x = self.base_pos[:, 0]              # Robot X position

        # Reward proportional to how far past the blocker the robot is, capped at 2m
        clearance = torch.clamp(robot_x - blocker_x, min=0.0, max=2.0)
        return clearance


    def _reward_centerline_near_blocker(self):
        """Penalize side-stepping near the blocker to discourage go-around solutions."""
        if not self.dynamic_obstacles or len(self.obstacle_positions) < 1:
            return torch.zeros(self.num_envs, device=self.device)

        blocker_x = self.obstacle_positions[0][0]
        blocker_y = self.obstacle_positions[0][1]
        robot_x = self.base_pos[:, 0]
        robot_y = self.base_pos[:, 1]

        x_window = self.env_cfg.get('jump_centerline_penalty_window', 1.8)
        y_tolerance = self.env_cfg.get('jump_centerline_tolerance', 0.35)
        x_success_margin = self.env_cfg.get('jump_success_x_margin', 0.25)

        x_dist = torch.abs(robot_x - blocker_x)
        y_error = torch.abs(robot_y - blocker_y)

        # High penalty only around the blocker and before true crossing.
        proximity = torch.clamp(1.0 - x_dist / max(x_window, 1e-6), min=0.0, max=1.0)
        before_cross = (robot_x <= (blocker_x + x_success_margin)).float()
        lateral_overrun = torch.clamp(y_error - y_tolerance, min=0.0, max=2.0)

        return lateral_overrun * proximity * before_cross


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
        
        height_above_normal = current_height - normal_height
        jump_reward = torch.zeros(self.num_envs, device=self.device)
        
        for obs_data in self.obstacle_positions:
            obs_pos = torch.tensor(obs_data[:2], device=self.device, dtype=torch.float32)
            obs_height = obs_data[2]
            
            # Distance and direction to obstacle
            to_obstacle = obs_pos.unsqueeze(0) - robot_pos
            distance = torch.norm(to_obstacle, dim=1)
            
            near_obstacle = distance < 2.5
            approaching = distance < 2.0
            
            # Directional check - only computed once, used where relevant
            to_obstacle_dot = (to_obstacle * self.base_lin_vel[:, :2]).sum(dim=1)
            moving_toward = to_obstacle_dot > 0

            at_wall = distance < 0.8

            # Tier 0: reward body ELEVATION near obstacle.
            # Previously rewarded crouching (going DOWN) — this was backwards.
            # The gradient we need is: being higher near an obstacle = good.
            # No moving_toward: forward vel is ~0 when physically blocked.
            elevation = torch.clamp(height_above_normal, 0.0, 0.25)  # 0 at normal, max at 25cm above
            tier0 = torch.where(
                near_obstacle,
                elevation * 2.0,  # max 0.5 per step
                torch.zeros_like(elevation)
            )

            # Tier 0b: constant floor bonus at the wall so reward never goes negative at contact
            tier0_wall = torch.where(at_wall, torch.full_like(distance, 0.05), torch.zeros_like(distance))

            # Tier 1: Any upward velocity near obstacle.
            # REMOVED moving_toward: physical blockage kills forward vel → moving_toward=False
            # → tier1 was zero at the exact moment legs were pushing off the box.
            # Lowered threshold 0.1 → 0.03: small upward velocity is still a learning signal.
            upward_velocity_reward = torch.clamp(vertical_velocity, 0, 1.0) * 0.5
            tier1 = torch.where(
                near_obstacle & (vertical_velocity > 0.03),
                upward_velocity_reward,
                torch.zeros_like(upward_velocity_reward)
            )

            # Tier 2: Airborne near obstacle.
            # REMOVED moving_toward: same reason as Tier 1 — body can be rising straight up
            # with near-zero forward vel and this tier was producing 0 reward.
            is_airborne = height_above_normal > 0.06
            airborne_reward = torch.clamp(height_above_normal * 5.0, 0, 1.0)
            tier2 = torch.where(
                near_obstacle & is_airborne,
                airborne_reward,
                torch.zeros_like(airborne_reward)
            )

            # Tier 3: Clearing obstacle height (unchanged)
            clearance_height = torch.clamp(current_height - (normal_height + obs_height * 0.7), 0, 0.2)
            tier3 = torch.where(
                near_obstacle & (clearance_height > 0),
                torch.clamp((clearance_height / 0.2) * 2.0, 0, 2.0),
                torch.zeros(self.num_envs, device=self.device)
            )

            jump_reward += tier0 + tier0_wall + tier1 + tier2 + tier3
        
        return jump_reward

    def _reward_jump_approach(self):
        """Phase 1 — potential-based approach shaping (Ng et al. 1999).
        Φ(s) = clamp((base_z - z_stand) / obs_h, 0, 1) × exp(-dist / 1.5)
        Rewarding height × proximity simultaneously creates a dense gradient from
        any distance without gating on velocity, which was allowing gait-oscillation
        bouncing to satisfy the old v_z condition without committing to a real jump.
        The exponential proximity factor peaks at the obstacle and fades smoothly,
        so the policy learns to approach AND elevate concurrently."""
        if not self.dynamic_obstacles or len(self.obstacle_positions) < 1:
            return torch.zeros(self.num_envs, device=self.device)

        robot_pos = self.base_pos[:, :2]
        base_z    = self.base_pos[:, 2]
        z_stand   = 0.42

        obs_data  = self.obstacle_positions[0]
        obs_pos   = torch.tensor(obs_data[:2], device=self.device, dtype=torch.float32)
        obs_h     = getattr(self, 'jump_curriculum_height', float(obs_data[2]))
        blocker_x = float(obs_data[0])

        # Stop paying jump shaping once the blocker is crossed to avoid farming
        # repeated hops after the core maneuver is complete.
        x_margin = self.env_cfg.get('jump_success_x_margin', 0.25)
        before_cross = self.base_pos[:, 0] <= (blocker_x + x_margin)

        distance  = torch.norm(obs_pos.unsqueeze(0) - robot_pos, dim=1)

        # Height potential: 0 at normal standing, 1 at jump target CoM height
        height_frac = torch.clamp((base_z - z_stand) / max(float(obs_h), 0.01), 0.0, 1.0)
        # Exponential proximity: 1.0 at the obstacle, ~0.37 at 1.5 m, ~0.04 at 5 m
        prox = torch.exp(-distance / 1.5)

        return torch.where(before_cross, height_frac * prox, torch.zeros(self.num_envs, device=self.device))

    def _reward_jump_takeoff(self):
        """Phase 2 — explosive upward-velocity at the obstacle wall (Zhuang et al. 2023).
        Two key changes from the original:
          (1) Gate tightened from 1.5 m → 0.8 m: robot must be physically at the wall.
              The old 1.5 m gate fired during the entire approach, letting gait bouncing
              accumulate takeoff reward without ever attempting a real jump.
          (2) Minimum v_z threshold of 0.3 m/s filters out the ~0.1 m/s oscillations
              that arise from normal trotting, so only genuine upward impulses score.
          (3) Reward scaled as v_z² (power, not velocity) to disproportionately prefer
              explosive single-impulse jumps over sustained small-velocity bouncing."""
        if not self.dynamic_obstacles or len(self.obstacle_positions) < 1:
            return torch.zeros(self.num_envs, device=self.device)

        robot_pos = self.base_pos[:, :2]
        vz        = self.base_lin_vel[:, 2]

        obs_data  = self.obstacle_positions[0]
        obs_pos   = torch.tensor(obs_data[:2], device=self.device, dtype=torch.float32)
        blocker_x = float(obs_data[0])
        blocker_y = float(obs_data[1])
        distance  = torch.norm(obs_pos.unsqueeze(0) - robot_pos, dim=1)

        at_wall   = distance < 0.8   # physically at the obstacle face
        explosive = vz > 0.3         # filters ~0.1 m/s gait-oscillation noise
        x_margin = self.env_cfg.get('jump_success_x_margin', 0.25)
        y_tol = self.env_cfg.get('jump_centerline_tolerance', 0.35)
        before_cross = self.base_pos[:, 0] <= (blocker_x + x_margin)
        centered = torch.abs(self.base_pos[:, 1] - blocker_y) <= y_tol

        return torch.where(at_wall & explosive & before_cross & centered,
                           torch.clamp(vz * vz, 0.0, 4.0),
                           torch.zeros(self.num_envs, device=self.device))

    def _reward_jump_flight(self):
        """Phase 3 — body clearance above obstacle top (Extreme Parkour, Zhuang et al. 2024).
        Two changes from the original:
          (1) Clearance threshold lowered from obs_h → obs_h * 0.5.  A quadruped clears
              a box when its feet reach obs_h from the ground; the CoM only rises by
              ~50 % of obs_h during such a manoeuvre (feet tuck while body rises).
              The old threshold (CoM rise = 100 % obs_h) was unreachable at 15 cm,
              which is why jump_flight was always 0.0 in the advanced_curr run.
          (2) Spatial gate tightened from 2.5 m → 1.5 m so the reward only fires when
              the robot is directly at or over the obstacle, not during the long approach."""
        if not self.dynamic_obstacles or len(self.obstacle_positions) < 1:
            return torch.zeros(self.num_envs, device=self.device)

        robot_pos = self.base_pos[:, :2]
        base_z    = self.base_pos[:, 2]
        vx        = self.base_lin_vel[:, 0]

        obs_data  = self.obstacle_positions[0]
        obs_pos   = torch.tensor(obs_data[:2], device=self.device, dtype=torch.float32)
        obs_h     = getattr(self, 'jump_curriculum_height', float(obs_data[2]))
        blocker_x = float(obs_data[0])
        blocker_y = float(obs_data[1])

        distance = torch.norm(obs_pos.unsqueeze(0) - robot_pos, dim=1)

        # Tightened proximity: robot must be directly over/at the obstacle
        over_obstacle = distance < 1.5
        # Threshold: CoM rises ~50 % of obstacle height when feet clear it
        clearance_threshold = 0.42 + obs_h * 0.5
        above_target = base_z > clearance_threshold
        x_margin = self.env_cfg.get('jump_success_x_margin', 0.25)
        y_tol = self.env_cfg.get('jump_centerline_tolerance', 0.35)
        before_cross = self.base_pos[:, 0] <= (blocker_x + x_margin)
        centered = torch.abs(self.base_pos[:, 1] - blocker_y) <= y_tol

        height_bonus = torch.clamp(base_z - clearance_threshold, 0.0, 0.3)
        fwd_bonus    = torch.clamp(1.0 + vx, 0.5, 2.5)

        return torch.where(over_obstacle & above_target & before_cross & centered,
                           height_bonus * fwd_bonus,
                           torch.zeros(self.num_envs, device=self.device))

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
        
        # Landing detection: descending always counts, grounded phase only counts
        # after a real airborne event was observed near the blocker.
        # Phase 1: Airborne and descending (LOWERED threshold: 5cm → 2cm)
        is_descending = (current_height > normal_height + 0.02) & (vertical_velocity < 0)
        
        # Phase 2: Recently landed (near ground with low vertical velocity)
        is_grounded = (current_height < normal_height + 0.08) & (torch.abs(vertical_velocity) < 0.3)

        if self.dynamic_obstacles and hasattr(self, 'jump_success_armed'):
            grounded_after_airborne = is_grounded & self.jump_success_armed
        else:
            grounded_after_airborne = is_grounded
        
        # Combined: reward during descent AND after landing
        in_landing_phase = is_descending | grounded_after_airborne
        
        # Check if robot is near any obstacle
        robot_pos = self.base_pos[:, :2]
        near_obstacle = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        landing_window = self.env_cfg.get('jump_landing_window', 1.4)
        
        for obs_data in self.obstacle_positions:
            obs_pos = torch.tensor(obs_data[:2], device=self.device, dtype=torch.float32)
            
            # Distance to obstacle
            to_obstacle = obs_pos.unsqueeze(0) - robot_pos
            distance = torch.norm(to_obstacle, dim=1)
            
            near_obstacle |= (distance < landing_window)
        
        # Reward stability when in landing phase near obstacles
        is_jump_context = in_landing_phase & near_obstacle
        
        # Reward low angular velocity (stability) and low vertical oscillation
        ang_vel_penalty = torch.norm(self.base_ang_vel, dim=1)
        stability_score = torch.exp(-ang_vel_penalty * 2.0)  # More sensitive to rotation
        
        # Scale reward: 0.3 during descent, 0.5 when grounded (encourage settling)
        reward_scale = torch.where(is_grounded, 0.5, 0.3)
        
        return torch.where(is_jump_context, stability_score * reward_scale, torch.zeros_like(current_height))


    def _reward_obstacle_avoidance(self):
        """Reward for proper obstacle handling (excludes target obstacle in goal-reaching mode)"""
        if len(self.obstacle_positions) == 0:
            return torch.zeros(self.num_envs, device=self.device)
        
        robot_pos = self.base_pos[:, :2]
        collision_penalty = torch.zeros(self.num_envs, device=self.device)
        
        # Skip the last obstacle only in dynamic goal-reaching mode (2 obstacles: blocker + target)
        if self.dynamic_obstacles and len(self.obstacle_positions) == 2:
            obstacles_to_check = self.obstacle_positions[:-1]  # Only check the blocker
        else:
            obstacles_to_check = self.obstacle_positions  # Check all obstacles
        
        for obs_data in obstacles_to_check:
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
            
            # Track collisions for stage 3 (set per-env flag, don't count each frame)
            self.episode_had_collision = torch.logical_or(self.episode_had_collision, collision_mask)
            
            collision_penalty += torch.where(collision_mask, -1.0, 0.0)
            
            collision_penalty += torch.where(close_mask, -0.1, 0.0)
        
        return collision_penalty
    

    def _reward_goal_heading(self):
        """Reward for facing toward the goal - fires at all times, encourages re-orientation after passing the blocker"""
        if not self.dynamic_obstacles or len(self.obstacle_positions) != 2:
            return torch.zeros(self.num_envs, device=self.device)

        target_pos = torch.tensor(self.obstacle_positions[1][:2], device=self.device, dtype=torch.float32)
        robot_pos = self.base_pos[:, :2]
        robot_yaw = self.base_radians[:, 2]

        to_goal = target_pos.unsqueeze(0) - robot_pos  # (N, 2)
        goal_dist = torch.norm(to_goal, dim=1, keepdim=True).clamp(min=0.1)
        to_goal_dir = to_goal / goal_dist

        # cos of angle between robot heading and goal direction: 1=facing goal, -1=facing away
        cos_yaw = torch.cos(robot_yaw)
        sin_yaw = torch.sin(robot_yaw)
        goal_cos = to_goal_dir[:, 0] * cos_yaw + to_goal_dir[:, 1] * sin_yaw

        return goal_cos  # range [-1, 1]


    def _reward_target_proximity(self):
        """Reward being close to the target object (second obstacle) - only in goal-reaching mode"""
        # Only active in dynamic obstacle mode with exactly 2 obstacles (blocker + target)
        if not self.dynamic_obstacles or len(self.obstacle_positions) != 2:
            return torch.zeros(self.num_envs, device=self.device)
        
        # Target is the second obstacle (cylinder behind the first)
        target_pos = torch.tensor(self.obstacle_positions[1][:2], device=self.device, dtype=torch.float32)
        robot_pos = self.base_pos[:, :2]
        distance = torch.norm(robot_pos - target_pos, dim=1)

        # Share termination radius to keep all goal checks aligned
        zero_radius = GOAL_TERMINATION_RADIUS

        # Long-range pull so far states still get directionally useful signal.
        linear_reward = torch.clamp(1.0 - distance / 12.0, 0.0, 1.0)

        # Mid-range pull around 1-3m.
        exp_reward = torch.exp(-distance / 1.1)

        # Precision shaping in the final approach window (0.7m..2.0m).
        # Cubic ramp makes the reward much more sensitive near the goal boundary.
        precision_window = 2.0
        precision_progress = torch.clamp(
            (precision_window - distance) / max(precision_window - zero_radius, 1e-6),
            0.0,
            1.0,
        )
        precision_reward = precision_progress * precision_progress * precision_progress

        reward = 0.2 * linear_reward + 0.35 * exp_reward + 0.45 * precision_reward

        # Smooth edge at zero_radius to avoid a hard discontinuity in the signal.
        edge_width = 0.08
        edge_taper = torch.clamp((distance - zero_radius) / edge_width, 0.0, 1.0)
        reward = reward * edge_taper
        return reward


    def _reward_target_reached(self):
        """Bonus reward for reaching the target object - only in goal-reaching mode"""
        # Only active in dynamic obstacle mode with exactly 2 obstacles (blocker + target)
        if not self.dynamic_obstacles or len(self.obstacle_positions) != 2:
            return torch.zeros(self.num_envs, device=self.device)
        
        target_pos = torch.tensor(self.obstacle_positions[1][:2], device=self.device, dtype=torch.float32)
        robot_pos = self.base_pos[:, :2]
        distance = torch.norm(robot_pos - target_pos, dim=1)
        
        # Award reaching the goal at shared termination radius
        reward = torch.zeros_like(distance)
        reward = torch.where(distance < GOAL_TERMINATION_RADIUS, 1.0, reward)
        return reward


    def _reward_target_progress(self):
        """Reward for moving toward the target (reduces distance) - prevents standing still"""
        # Only active in dynamic obstacle mode with exactly 2 obstacles (blocker + target)
        if not self.dynamic_obstacles or len(self.obstacle_positions) != 2:
            return torch.zeros(self.num_envs, device=self.device)
        
        target_pos = torch.tensor(self.obstacle_positions[1][:2], device=self.device, dtype=torch.float32)
        robot_pos = self.base_pos[:, :2]
        current_distance = torch.norm(robot_pos - target_pos, dim=1)

        # Zero out reward within shared termination radius (avoid duplicate reward near goal)
        zero_radius = GOAL_TERMINATION_RADIUS
        mask = current_distance >= zero_radius
        
        # Initialize tracking on first call
        if not hasattr(self, 'last_target_distance'):
            self.last_target_distance = current_distance.clone()
            return torch.zeros(self.num_envs, device=self.device)  # No reward on first step
        
        # Compute progress only along the direction TO the target (not raw Euclidean)
        # This prevents reward hacking by moving sideways to reduce diagonal distance
        to_target = target_pos.unsqueeze(0) - robot_pos  # (num_envs, 2)
        to_target_dist = torch.norm(to_target, dim=1, keepdim=True).clamp(min=1e-6)
        to_target_dir = to_target / to_target_dist  # unit vector toward target

        # Project robot velocity onto target direction - reward only genuine approach
        robot_vel_xy = self.base_lin_vel[:, :2]  # (num_envs, 2)
        approach_vel = torch.sum(robot_vel_xy * to_target_dir, dim=1)  # scalar per env

        # Update distance tracking
        self.last_target_distance = current_distance.clone()
        
        # Reward = velocity component toward target (positive = approaching).
        # Clamped to min=0 so physics bounce-back at the obstacle wall doesn't give
        # a large negative penalty that makes "stop before obstacle" the optimal strategy.
        reward = torch.clamp(approach_vel, min=0.0, max=2.0)
        reward = reward * mask.float()  # Zero out reward within zero_radius
        return reward
    

    def _reward_fast_completion(self):
        """Reward for faster completion: higher reward for fewer steps to reach the goal."""
        # Only reward on termination due to goal reached
        if not self.dynamic_obstacles or len(self.obstacle_positions) != 2:
            return torch.zeros(self.num_envs, device=self.device)
        
        # Compute fresh goal distance instead of reading stale obs_buf
        target_pos = torch.tensor(self.obstacle_positions[1][:2], device=self.device, dtype=torch.float32)
        robot_pos = self.base_pos[:, :2]
        distance = torch.norm(robot_pos - target_pos, dim=1)
        goal_reached = distance < GOAL_TERMINATION_RADIUS

        # Calculate completion bonus: higher for fewer steps
        max_steps = self.max_episode_length
        steps_taken = self.episode_length_buf
        completion_bonus = (max_steps - steps_taken) / max_steps  # Range: 0 (slow) to 1 (fast)

        # Only reward on termination
        return completion_bonus * goal_reached.float()


    def _check_termination(self):
        """Override base termination to include collision detection and goal reaching"""
        # Call base class termination checks (pitch, roll, height, time)
        super()._check_termination()

        # Terminate if robot reaches goal or walks significantly past it
        if self.dynamic_obstacles and len(self.obstacle_positions) == 2:
            # Compute fresh goal distance (do NOT read stale obs_buf which is from previous step)
            target_pos = torch.tensor(self.obstacle_positions[1][:2], device=self.device, dtype=torch.float32)
            robot_pos = self.base_pos[:, :2]
            distance = torch.norm(robot_pos - target_pos, dim=1)
            
            # Track average distance for stage 3
            self.avoidance_avg_distance = distance.mean().item()
            
            # Terminate when goal is reached using shared termination radius
            goal_reached = distance < GOAL_TERMINATION_RADIUS
            
            # Track goal reaches for stage 3 avoidance logging
            num_goals_reached = goal_reached.sum().item()
            self.avoidance_goals_reached += num_goals_reached
            
            self.reset_buf |= goal_reached
            
            # Also terminate for significant overshoot (walked well past goal)
            overshoot_threshold = 3.0
            overshot = robot_pos[:, 0] > (target_pos[0] + overshoot_threshold)
            self.reset_buf |= overshot

        # Add collision termination for navigation (if enabled)
        if self.env_cfg.get('terminate_on_collision', False):
            contacts = self.contact_sensor.read()
            # blocker_dist_norm = distance_to_blocker / 8.0 (see obs_buf construction)
            # collision_threshold = self.env_cfg.get('collision_threshold', 0.25)
            # blocker_norm_thresh = collision_threshold / 8.0
            # collision_mask = blocker_dist_norm < blocker_norm_thresh
            self.reset_buf |= contacts.squeeze(-1)
    

    def reset_idx(self, envs_idx):
        """Override to reset navigation-specific tracking"""
        # Track episode counters for stage 3 avoidance
        self.avoidance_episodes_total += len(envs_idx)
        
        # Count episodes with collisions (true collision rate: 0-100%)
        collisions_in_reset = self.episode_had_collision[envs_idx].sum().item()
        self.avoidance_collisions += collisions_in_reset
        
        # Call base class reset
        super().reset_idx(envs_idx)
        
        # Reset collision flag for these environments
        self.episode_had_collision[envs_idx] = False

        # Reset strict curriculum success tracking.
        self.jump_success_armed[envs_idx] = False
        self.jump_success_counted[envs_idx] = False
        
        # Reset progress tracking for goal-reaching
        if hasattr(self, 'last_target_distance'):
            if self.dynamic_obstacles and len(self.obstacle_positions) == 2:
                target_pos = torch.tensor(self.obstacle_positions[1][:2], device=self.device, dtype=torch.float32)
                robot_pos = self.base_pos[:, :2]
                self.last_target_distance[envs_idx] = torch.norm(robot_pos[envs_idx] - target_pos, dim=1)
            else:
                self.last_target_distance[envs_idx] = 0.0


    def _advance_curriculum(self):
        """Swap active blocker to the next curriculum height level.
        Parks the old box at (1000, 1000) and teleports the new one in front of the robot.
        Boxes are pre-built at the correct size so no resizing is needed."""
        if not hasattr(self, 'curriculum_heights') or self.curriculum_level >= len(self.curriculum_heights) - 1:
            return

        old_level = self.curriculum_level
        new_level = old_level + 1
        old_h = self.curriculum_heights[old_level]
        new_h = self.curriculum_heights[new_level]

        blocker_x = self.obstacle_positions[0][0]
        blocker_y = self.obstacle_positions[0][1]
        all_envs  = torch.arange(self.num_envs, device=self.device)

        # Park old entity far away (on the ground at its own height)
        park_pos = torch.tensor([1000.0, 1000.0, old_h / 2], device=self.device)
        self.curriculum_entities[old_level].set_pos(
            park_pos.unsqueeze(0).repeat(self.num_envs, 1), envs_idx=all_envs
        )

        # Activate new entity at blocker position
        active_pos = torch.tensor([blocker_x, blocker_y, new_h / 2], device=self.device)
        self.curriculum_entities[new_level].set_pos(
            active_pos.unsqueeze(0).repeat(self.num_envs, 1), envs_idx=all_envs
        )

        jump_threshold = self.reward_cfg.get('jump_height_threshold', 0.2)
        obs_type = "low" if new_h < jump_threshold else "high"
        self.obstacle_positions[0] = [blocker_x, blocker_y, new_h, obs_type]
        self.obstacle_entities[0]  = self.curriculum_entities[new_level]
        self.jump_curriculum_height = new_h
        self.curriculum_level       = new_level
        self.jump_success_count     = 0

        # Calculate metrics for this advancement
        avg_attempts = (self.jump_attempts_at_level / self.num_envs) if self.num_envs > 0 else 0
        self.curriculum_advance_metrics = {
            "curriculum_level": new_level,
            "prev_height_cm": old_h * 100,
            "new_height_cm": new_h * 100,
            "total_attempts": self.jump_attempts_at_level,
            "avg_attempts_per_env": avg_attempts,
            "successful_jumps": self.jump_success_threshold,
        }
        
        print(f"🎓 Curriculum advanced: {old_h*100:.1f}cm → {new_h*100:.1f}cm "
              f"(level {new_level}/{len(self.curriculum_heights)-1}) "
              f"| Attempts: {self.jump_attempts_at_level} total ({avg_attempts:.1f} per env)")
        
        self._log_curriculum_advancement()  # Log to CSV
        
        self.jump_attempts_at_level = 0
        self.jump_attempts_per_env.zero_()


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

        # Height-curriculum success detection with strict anti-bypass criteria.
        if (self.dynamic_obstacles
                and hasattr(self, 'curriculum_heights')
                and len(self.obstacle_positions) >= 1):
            blocker_x = self.obstacle_positions[0][0]
            blocker_y = self.obstacle_positions[0][1]
            blocker_h = getattr(self, 'jump_curriculum_height', float(self.obstacle_positions[0][2]))

            width = self.env_cfg.get('dynamic_obstacle_width', 0.4)
            half_width = width * 0.5
            x_margin = self.env_cfg.get('jump_success_x_margin', 0.25)
            y_tol = self.env_cfg.get('jump_centerline_tolerance', 0.35)
            clearance_factor = self.env_cfg.get('jump_success_clearance_factor', 0.35)
            z_margin = self.env_cfg.get('jump_success_height_margin', 0.03)

            robot_x = self.base_pos[:, 0]
            robot_y = self.base_pos[:, 1]
            robot_z = self.base_pos[:, 2]

            near_blocker = torch.abs(robot_x - blocker_x) < (half_width + 0.35)
            centered = torch.abs(robot_y - blocker_y) <= y_tol

            # Arm success if the robot gets airborne above a minimum clearance while near blocker.
            min_jump_z = 0.42 + blocker_h * clearance_factor + z_margin
            airborne_near_blocker = near_blocker & centered & (robot_z > min_jump_z)
            self.jump_success_armed |= airborne_near_blocker

            crossed = robot_x > (blocker_x + half_width + x_margin)
            valid_success = crossed & centered & self.jump_success_armed & (~self.jump_success_counted)

            successful_crossings = int(valid_success.sum().item())
            
            # Count every step as an attempt for all environments
            self.jump_attempts_at_level += self.num_envs
            self.jump_attempts_per_env += 1
            
            if successful_crossings > 0:
                self.jump_success_counted |= valid_success
                self.jump_success_count += successful_crossings
                if self.jump_success_count >= self.jump_success_threshold:
                    self._advance_curriculum()

        self._compute_rewards()

        self._compute_observations()

        self.last_actions[:] = actions
        self.last_dof_vel[:] = self.dof_vel

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=torch.float)
        self.extras["time_outs"][time_out_idx] = 1.0

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras