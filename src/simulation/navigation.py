"""
Navigation Environment module for Cynosure embodied pet simulation.

Provides a procedurally generated 10x10x10 grid environment with obstacles,
LiDAR ray casting, and collision detection for robot navigation training.
"""

import numpy as np
from typing import Tuple, List


class NavigationEnvironment:
    """
    Procedural 10x10x10 grid navigation environment with obstacles.

    Provides:
    - Procedurally generated obstacle placement
    - LiDAR ray casting (8 rays, 360 degree coverage)
    - Collision detection (obstacles and boundaries)
    - Deterministic environment generation via seed

    Attributes:
        grid_size: Size of the cubic grid (default 10)
        num_obstacles: Number of circular obstacles to place
        seed: Random seed for reproducible generation
    """

    def __init__(self, grid_size: float = 10, num_obstacles: int = 3, seed=None):
        """
        Initialize the navigation environment.

        Args:
            grid_size: Size of cubic grid in meters (default 10)
            num_obstacles: Number of cylindrical obstacles (default 3)
            seed: Random seed for reproducibility (default None)
        """
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.seed_value = seed

        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Obstacle storage: list of (center_x, center_y, radius)
        self.obstacles: List[Tuple[float, float, float]] = []

        # Agent start and goal positions
        self.agent_start: Tuple[float, float, float] = (0, 0, 0)
        self.agent_goal: Tuple[float, float, float] = (0, 0, 0)

        # Generate initial obstacles and positions
        self._generate_obstacles()

    def _generate_obstacles(self) -> None:
        """
        Generate obstacles procedurally in the grid.

        Obstacles are circles (cylinders) with random positions and fixed radius.
        Ensures no overlap between obstacles by checking distance.
        """
        self.obstacles = []
        min_distance_between_centers = 2.0
        obstacle_radius = 0.3

        while len(self.obstacles) < self.num_obstacles:
            # Generate random position within grid bounds (with margin)
            margin = obstacle_radius + 0.5
            x = np.random.uniform(margin, self.grid_size - margin)
            y = np.random.uniform(margin, self.grid_size - margin)

            # Check for overlap with existing obstacles
            overlaps = False
            for ox, oy, orad in self.obstacles:
                distance = np.sqrt((x - ox) ** 2 + (y - oy) ** 2)
                if distance < min_distance_between_centers:
                    overlaps = True
                    break

            if not overlaps:
                self.obstacles.append((x, y, obstacle_radius))

    def reset(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Reset environment and return start and goal positions.

        Places agent at a random location and goal at another location
        such that they are at least 2.0 m apart.

        Returns:
            Tuple of (start_position, goal_position) where each is (x, y, z)
        """
        # Place start position
        start_x = np.random.uniform(0.5, self.grid_size - 0.5)
        start_y = np.random.uniform(0.5, self.grid_size - 0.5)
        start_z = 0.1  # Slightly above ground

        # Place goal position at least 2.0 m away from start
        goal_valid = False
        while not goal_valid:
            goal_x = np.random.uniform(0.5, self.grid_size - 0.5)
            goal_y = np.random.uniform(0.5, self.grid_size - 0.5)
            goal_z = 0.1

            distance = np.sqrt((goal_x - start_x) ** 2 + (goal_y - start_y) ** 2)
            if distance > 2.0:
                goal_valid = True

        self.agent_start = (start_x, start_y, start_z)
        self.agent_goal = (goal_x, goal_y, goal_z)

        return self.agent_start, self.agent_goal

    def get_lidar(self, agent_pos: Tuple[float, float, float], num_rays: int = 8) -> np.ndarray:
        """
        Cast LiDAR rays from agent position and return distances to obstacles.

        Rays are evenly distributed around the agent in the horizontal plane.
        Uses ray-circle intersection to find closest obstacle for each ray.

        Args:
            agent_pos: Agent position as (x, y, z)
            num_rays: Number of rays to cast (default 8)

        Returns:
            numpy array of shape (num_rays,) with distances to nearest obstacle.
            Distances are bounded by [0, grid_size].
        """
        ax, ay, az = agent_pos
        distances = np.zeros(num_rays)

        # Generate ray directions: evenly spaced around 360 degrees
        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)

        for i, angle in enumerate(angles):
            # Ray direction (unit vector in horizontal plane)
            dx = np.cos(angle)
            dy = np.sin(angle)

            # Find closest obstacle for this ray
            min_distance = self.grid_size  # Max distance is grid size

            # Check each obstacle
            for ox, oy, orad in self.obstacles:
                # Ray-circle intersection
                # Ray: P(t) = (ax + t*dx, ay + t*dy)
                # Circle: (x - ox)^2 + (y - oy)^2 = orad^2
                # Substitute: (ax + t*dx - ox)^2 + (ay + t*dy - oy)^2 = orad^2

                # Let: px = ax - ox, py = ay - oy
                px = ax - ox
                py = ay - oy

                # (px + t*dx)^2 + (py + t*dy)^2 = orad^2
                # px^2 + 2*px*t*dx + t^2*dx^2 + py^2 + 2*py*t*dy + t^2*dy^2 = orad^2
                # t^2*(dx^2 + dy^2) + 2*t*(px*dx + py*dy) + (px^2 + py^2 - orad^2) = 0
                # Since dx^2 + dy^2 = 1 (unit vector):
                # t^2 + 2*t*(px*dx + py*dy) + (px^2 + py^2 - orad^2) = 0

                a_coeff = 1.0  # dx^2 + dy^2
                b_coeff = 2.0 * (px * dx + py * dy)
                c_coeff = px * px + py * py - orad * orad

                discriminant = b_coeff * b_coeff - 4 * a_coeff * c_coeff

                if discriminant >= 0:
                    # Ray intersects circle
                    sqrt_disc = np.sqrt(discriminant)
                    t1 = (-b_coeff - sqrt_disc) / (2 * a_coeff)
                    t2 = (-b_coeff + sqrt_disc) / (2 * a_coeff)

                    # Take the smallest positive t (first intersection)
                    valid_distances = [t for t in [t1, t2] if t > 0.01]  # Small epsilon to avoid self-intersection
                    if valid_distances:
                        closest_t = min(valid_distances)
                        if closest_t < min_distance:
                            min_distance = closest_t

            # Check boundary collision (rays extend to grid boundaries)
            # Ray extends in direction (dx, dy) from (ax, ay)
            # Boundaries: x in [0, grid_size], y in [0, grid_size]

            boundary_distance = self.grid_size  # Default to max

            # Check x boundaries
            if dx != 0:
                for boundary_x in [0, self.grid_size]:
                    t = (boundary_x - ax) / dx
                    if t > 0.01:
                        y_at_t = ay + t * dy
                        if 0 <= y_at_t <= self.grid_size:
                            boundary_distance = min(boundary_distance, t)

            # Check y boundaries
            if dy != 0:
                for boundary_y in [0, self.grid_size]:
                    t = (boundary_y - ay) / dy
                    if t > 0.01:
                        x_at_t = ax + t * dx
                        if 0 <= x_at_t <= self.grid_size:
                            boundary_distance = min(boundary_distance, t)

            distances[i] = min(min_distance, boundary_distance)

        return distances.astype(np.float32)

    def check_collision(self, agent_pos: Tuple[float, float, float], radius: float = 0.1) -> bool:
        """
        Check if agent collides with obstacles or boundaries.

        Agent is modeled as a sphere with given radius.
        Collides if:
        - Distance to any obstacle < agent_radius + obstacle_radius
        - Agent position outside grid boundaries [0, grid_size] in x,y

        Args:
            agent_pos: Agent position as (x, y, z)
            radius: Agent radius (default 0.1)

        Returns:
            True if collision detected, False otherwise
        """
        ax, ay, az = agent_pos

        # Check boundary collision
        if ax - radius < 0 or ax + radius > self.grid_size:
            return True
        if ay - radius < 0 or ay + radius > self.grid_size:
            return True

        # Check collision with obstacles
        for ox, oy, orad in self.obstacles:
            distance_to_center = np.sqrt((ax - ox) ** 2 + (ay - oy) ** 2)
            min_distance = radius + orad

            if distance_to_center < min_distance:
                return True

        return False
