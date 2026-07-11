import pybullet as p
import pybullet_data
import numpy as np
import torch
import math


class PhysicsWorld:
    """
    Pybullet-based physics simulation for RPi5-like differential-drive robot.

    Handles:
    - Robot body (cylinder + casters, mass ~500g)
    - Arena (ground plane, obstacles)
    - Sensor simulation (position, velocity, collisions)
    - Per-tick physics integration (240 Hz, 10 substeps)
    """

    def __init__(self, num_agents=5, grid_size=50, use_gui=False, device='cpu'):
        """
        Args:
            num_agents: Number of robots in the arena
            grid_size: Arena size (grid_size x grid_size x grid_size)
            use_gui: If True, render with Pybullet GUI (slow)
            device: 'cpu' or 'cuda' for sensor computations
        """
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.device = device

        # Pybullet setup
        if use_gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)

        # Load/create arena
        self.ground_id = None
        self.obstacles = []
        self.agent_bodies = []
        self.agent_radius = 0.1  # 10cm radius
        self.agent_mass = 0.5  # 500g

        self._setup_arena()
        self._create_agents()

        # Physics parameters
        self.physics_freq = 240  # Hz
        self.sim_time_step = 1.0 / self.physics_freq
        self.time = 0.0

    def _setup_arena(self):
        """Create ground plane and obstacles."""
        # Ground plane
        ground_shape = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=self.client)
        self.ground_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=ground_shape,
            basePosition=[self.grid_size/2, self.grid_size/2, 0],
            physicsClientId=self.client
        )

        # Add a few obstacles (boxes) in the arena
        for i in range(3):
            x = 10 + i * 15
            y = 25
            z = 0.5
            obstacle_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1, 1, 0.5], physicsClientId=self.client)
            obstacle_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=obstacle_shape,
                basePosition=[x, y, z],
                physicsClientId=self.client
            )
            self.obstacles.append(obstacle_id)

        # Add resource nodes (small spheres that can be collected)
        for i in range(5):
            x = np.random.uniform(5, self.grid_size - 5)
            y = np.random.uniform(5, self.grid_size - 5)
            z = 0.2
            resource_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1, physicsClientId=self.client)
            resource_id = p.createMultiBody(
                baseMass=0.05,
                baseCollisionShapeIndex=resource_shape,
                basePosition=[x, y, z],
                physicsClientId=self.client
            )
            self.obstacles.append(resource_id)  # Reuse list for simplicity

    def _create_agents(self):
        """Create robot bodies for each agent."""
        for i in range(self.num_agents):
            x = 5 + i * 8
            y = 5
            z = 0.2

            # Create a simple cylinder as robot body
            agent_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=self.agent_radius, height=0.2, physicsClientId=self.client)
            agent_id = p.createMultiBody(
                baseMass=self.agent_mass,
                baseCollisionShapeIndex=agent_shape,
                basePosition=[x, y, z],
                physicsClientId=self.client
            )

            # Set friction
            p.changeDynamics(agent_id, -1, lateralFriction=0.5, physicsClientId=self.client)

            self.agent_bodies.append(agent_id)

    def step(self, actions):
        """
        Step simulation. Actions are 7-element discrete values:
        0: STOP, 1: MOVE_FORWARD, 2: MOVE_BACKWARD, 3: TURN_LEFT, 4: TURN_RIGHT, 5: ATTACK, 6: IDLE
        """
        for agent_id, action in actions.items():
            if agent_id >= len(self.agent_bodies):
                continue

            body_id = self.agent_bodies[agent_id]

            # Map action to differential drive commands
            left_speed, right_speed = self._action_to_drive(action)

            # Apply wheel forces (simplified tank drive)
            # Pybullet doesn't have native differential drive, so we use applyExternalForce
            forward_vec = np.array([np.cos(self._get_yaw(body_id)), np.sin(self._get_yaw(body_id)), 0])
            right_vec = np.array([-np.sin(self._get_yaw(body_id)), np.cos(self._get_yaw(body_id)), 0])

            left_force = forward_vec * left_speed + right_vec * (left_speed * 0.1)  # Differential
            right_force = forward_vec * right_speed - right_vec * (right_speed * 0.1)

            p.applyExternalForce(body_id, -1, left_force, [0, 0, 0.1], p.WORLD_FRAME, physicsClientId=self.client)

        # Step physics (10 substeps for stability)
        for _ in range(10):
            p.stepSimulation(physicsClientId=self.client)

        self.time += self.sim_time_step

        # Collect sensor readings
        sensor_readings = {}
        for agent_id in range(self.num_agents):
            sensor_readings[agent_id] = self.get_sensor_readings(agent_id)

        return sensor_readings

    def _action_to_drive(self, action):
        """Map 7-element action to (left_speed, right_speed) in range [-1, 1]."""
        max_speed = 2.0  # m/s
        if action == 0:  # STOP
            return 0, 0
        elif action == 1:  # MOVE_FORWARD
            return max_speed, max_speed
        elif action == 2:  # MOVE_BACKWARD
            return -max_speed, -max_speed
        elif action == 3:  # TURN_LEFT
            return -max_speed * 0.5, max_speed * 0.5
        elif action == 4:  # TURN_RIGHT
            return max_speed * 0.5, -max_speed * 0.5
        else:  # ATTACK, IDLE: no movement
            return 0, 0

    def _get_yaw(self, body_id):
        """Get yaw angle (rotation around Z) from body orientation."""
        _, rot = p.getBasePositionAndOrientation(body_id, physicsClientId=self.client)
        # rot is a quaternion (x, y, z, w)
        x, y, z, w = rot
        yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return yaw

    def get_sensor_readings(self, agent_id):
        """Return sensor dict for agent."""
        body_id = self.agent_bodies[agent_id]
        pos, rot = p.getBasePositionAndOrientation(body_id, physicsClientId=self.client)
        vel, ang_vel = p.getBaseVelocity(body_id, physicsClientId=self.client)

        # Check for collisions with obstacles
        collisions = p.getContactPoints(bodyA=body_id, physicsClientId=self.client)

        # Simulate vision: raycast forward, detect obstacles
        yaw = self._get_yaw(body_id)
        vision_objects = self._simulate_vision(body_id, yaw)

        # Simulate lidar: 360-degree rangefinder
        lidar_readings = self._simulate_lidar(body_id, yaw)

        # IMU: accelerometer, gyroscope, magnetometer
        imu_readings = {
            'accel': vel,  # Simplified: linear velocity as acceleration
            'gyro': ang_vel,
            'mag': (1.0, 0.0, 0.0),  # Dummy magnetometer (points to +X)
        }

        # Motor feedback
        motor_feedback = {'left_speed': 0.0, 'right_speed': 0.0}  # Would be set from drive commands

        return {
            'vision': vision_objects,
            'lidar': lidar_readings,
            'imu': imu_readings,
            'motor_feedback': motor_feedback,
            'ram': 50.0,  # Dummy resource level
            'position': pos,
            'time': int(self.time * 1000),  # ms
        }

    def _simulate_vision(self, agent_id, yaw):
        """Raycast to detect objects ahead."""
        pos, _ = p.getBasePositionAndOrientation(agent_id, physicsClientId=self.client)
        forward = np.array([np.cos(yaw), np.sin(yaw), 0])

        objects = []
        for angle_offset in [-0.3, -0.15, 0, 0.15, 0.3]:  # 5 rays
            ray_dir = np.array([
                np.cos(yaw + angle_offset),
                np.sin(yaw + angle_offset),
                0
            ])
            ray_end = np.array(pos) + ray_dir * 10  # 10m range

            result = p.rayTest(pos, ray_end, physicsClientId=self.client)[0]
            if result[0] != -1:  # Hit something
                hit_obj = result[0]
                hit_pos = result[3]
                dist = np.linalg.norm(np.array(hit_pos) - np.array(pos))
                objects.append((hit_pos[0], hit_pos[1], hit_pos[2], 1, 0.9))  # class=1, conf=0.9

        return objects

    def _simulate_lidar(self, agent_id, yaw):
        """360-degree rangefinder."""
        pos, _ = p.getBasePositionAndOrientation(agent_id, physicsClientId=self.client)

        readings = []
        for angle in np.linspace(0, 2*np.pi, 360, endpoint=False):
            ray_dir = np.array([np.cos(angle), np.sin(angle), 0])
            ray_end = np.array(pos) + ray_dir * 50  # 50m range

            result = p.rayTest(pos, ray_end, physicsClientId=self.client)[0]
            if result[0] != -1:
                hit_pos = result[3]
                dist = np.linalg.norm(np.array(hit_pos) - np.array(pos))
            else:
                dist = 50.0  # Max range

            readings.append((angle, dist))

        return readings

    def reset(self, agent_positions=None):
        """Reset all agents to starting positions."""
        raise NotImplementedError("Implement in next step")

    def close(self):
        """Disconnect from Pybullet."""
        p.disconnect(self.client)
