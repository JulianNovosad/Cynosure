"""
Sensor simulation module: converts arena state into realistic sensor inputs.
Produces camera object detections, lidar ranges, and IMU data that feed into semantic encoding.
"""

import torch
import numpy as np
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DetectedObject:
    """Simulated object detection output (like MobileNetSSD)."""
    label: str  # 'agent', 'ram', 'threat', 'wall'
    confidence: float  # 0-1 detection confidence
    x: float  # normalized position in frame 0-1
    y: float  # normalized position in frame 0-1
    width: float  # bbox width as fraction of frame
    height: float  # bbox height as fraction of frame
    distance: float  # estimated distance in meters


class CameraSimulator:
    """Simulates RGB camera with object detection (like MobileNetSSD)."""

    def __init__(self, resolution: Tuple[int, int] = (320, 240), fov_degrees: float = 60.0):
        self.resolution = resolution
        self.fov_rad = math.radians(fov_degrees)
        self.max_detect_range = 15.0  # meters

    def detect_objects(
        self,
        agent_pos: Tuple[float, float, float],
        agent_dir: float,
        agents: List,
        ram_nodes: List,
        safe_zone_center: Tuple[float, float, float],
        safe_zone_radius: float,
    ) -> List[DetectedObject]:
        """
        Simulate camera detecting objects in the agent's field of view.
        Returns list of DetectedObject with confidence scores and positions.
        """
        detections = []
        ax, ay, az = agent_pos

        # Detect other agents
        for other in agents:
            if other.id == agent_pos.id if hasattr(agent_pos, 'id') else False:
                continue
            ox, oy, oz = other.position
            dx, dy, dz = ox - ax, oy - ay, oz - az
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)

            if dist < self.max_detect_range and dist > 0.1:
                # Check if in FOV
                angle_to_object = math.atan2(dx, dy)
                angle_diff = abs(angle_to_object - agent_dir)
                if angle_diff > math.pi:
                    angle_diff = 2 * math.pi - angle_diff

                if angle_diff < self.fov_rad / 2:
                    # Convert to camera frame
                    x_norm = 0.5 + (angle_diff / self.fov_rad)
                    y_norm = 0.5 - (dz / self.max_detect_range) * 0.3

                    # Confidence decreases with distance
                    conf = max(0.5, 1.0 - (dist / self.max_detect_range))

                    threat_label = "agent_threat" if other.ram > 100 else "agent"
                    detections.append(DetectedObject(
                        label=threat_label,
                        confidence=conf,
                        x=max(0, min(1, x_norm)),
                        y=max(0, min(1, y_norm)),
                        width=0.1,
                        height=0.1,
                        distance=dist,
                    ))

        # Detect RAM nodes
        for ram in ram_nodes:
            rx, ry, rz = ram.position
            dx, dy, dz = rx - ax, ry - ay, rz - az
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)

            if dist < self.max_detect_range and dist > 0.1:
                angle_to_object = math.atan2(dx, dy)
                angle_diff = abs(angle_to_object - agent_dir)
                if angle_diff > math.pi:
                    angle_diff = 2 * math.pi - angle_diff

                if angle_diff < self.fov_rad / 2:
                    x_norm = 0.5 + (angle_diff / self.fov_rad)
                    y_norm = 0.5 - (dz / self.max_detect_range) * 0.3
                    conf = max(0.6, 1.0 - (dist / self.max_detect_range) * 0.3)

                    detections.append(DetectedObject(
                        label="ram",
                        confidence=conf,
                        x=max(0, min(1, x_norm)),
                        y=max(0, min(1, y_norm)),
                        width=0.08,
                        height=0.08,
                        distance=dist,
                    ))

        # Detect safe zone boundary as threat
        sz_dx = safe_zone_center[0] - ax
        sz_dy = safe_zone_center[1] - ay
        sz_dz = safe_zone_center[2] - az
        dist_to_zone = math.sqrt(sz_dx*sz_dx + sz_dy*sz_dy + sz_dz*sz_dz)

        if dist_to_zone > safe_zone_radius - 2.0:
            threat_conf = max(0.5, 1.0 - (dist_to_zone - safe_zone_radius) / 5.0)
            detections.append(DetectedObject(
                label="boundary_threat",
                confidence=threat_conf,
                x=0.5,
                y=0.7,
                width=0.3,
                height=0.2,
                distance=dist_to_zone - safe_zone_radius,
            ))

        return detections


class LidarSimulator:
    """Simulates 2D lidar with range measurements."""

    def __init__(self, num_rays: int = 32, max_range: float = 10.0):
        self.num_rays = num_rays
        self.max_range = max_range
        self.angles = [2 * math.pi * i / num_rays for i in range(num_rays)]

    def get_ranges(
        self,
        agent_pos: Tuple[float, float, float],
        agent_dir: float,
        obstacles: List[Tuple[float, float, float]],
        safe_zone_center: Tuple[float, float, float],
        safe_zone_radius: float,
    ) -> torch.Tensor:
        """
        Return tensor of range measurements (1D, num_rays).
        Includes agents, RAM nodes, walls, and safe zone boundary.
        """
        ranges = torch.full((self.num_rays,), self.max_range, dtype=torch.float32)
        ax, ay, az = agent_pos

        for i, angle_offset in enumerate(self.angles):
            ray_angle = agent_dir + angle_offset
            ray_x = math.cos(ray_angle)
            ray_y = math.sin(ray_angle)

            # Check obstacles
            for ox, oy, oz in obstacles:
                dx = ox - ax
                dy = oy - ay
                # Project onto ray
                proj = dx * ray_x + dy * ray_y
                if proj > 0:
                    cross = abs(dx * ray_y - dy * ray_x)
                    if cross < 0.3:  # object width
                        dist = proj - 0.3
                        if 0 < dist < ranges[i]:
                            ranges[i] = dist

            # Check safe zone boundary
            sz_dx = safe_zone_center[0] - ax
            sz_dy = safe_zone_center[1] - ay
            proj = sz_dx * ray_x + sz_dy * ray_y
            if proj > 0:
                cross = abs(sz_dx * ray_y - sz_dy * ray_x)
                boundary_dist = math.sqrt(proj*proj + cross*cross) - safe_zone_radius
                if -2.0 < boundary_dist < ranges[i]:
                    ranges[i] = max(0.1, boundary_dist)

        return ranges / self.max_range  # Normalize to 0-1


class IMUSensor:
    """Simulates 6-DOF IMU: 3D accelerometer + 3D gyroscope."""

    def __init__(self, noise_std: float = 0.01):
        self.noise_std = noise_std

    def get_reading(
        self,
        velocity: Tuple[float, float, float],
        angular_velocity: Tuple[float, float, float],
        gravity_accel: float = 9.81,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (accel, gyro) tensors, each 3D.
        Accel includes gravity component.
        """
        accel = torch.tensor(
            [velocity[0], velocity[1], velocity[2] + gravity_accel],
            dtype=torch.float32
        )
        gyro = torch.tensor(angular_velocity, dtype=torch.float32)

        # Add sensor noise
        accel += torch.randn_like(accel) * self.noise_std
        gyro += torch.randn_like(gyro) * self.noise_std

        return accel, gyro


class SensorFusion:
    """Combines all sensor inputs and produces the agent's sensory state vector."""

    def __init__(self, num_lidar_rays: int = 32):
        self.camera = CameraSimulator()
        self.lidar = LidarSimulator(num_rays=num_lidar_rays)
        self.imu = IMUSensor()

    def fuse_sensors(
        self,
        agent_pos: Tuple[float, float, float],
        agent_dir: float,
        velocity: Tuple[float, float, float],
        angular_velocity: Tuple[float, float, float],
        agents: List,
        ram_nodes: List,
        safe_zone_center: Tuple[float, float, float],
        safe_zone_radius: float,
    ) -> dict:
        """
        Fuse all sensors into a sensor state dictionary.
        Keys: 'detections', 'ranges', 'accel', 'gyro'
        """
        detections = self.camera.detect_objects(
            agent_pos, agent_dir, agents, ram_nodes,
            safe_zone_center, safe_zone_radius
        )

        obstacles = [ag.position for ag in agents] + [r.position for r in ram_nodes]
        ranges = self.lidar.get_ranges(
            agent_pos, agent_dir, obstacles,
            safe_zone_center, safe_zone_radius
        )

        accel, gyro = self.imu.get_reading(velocity, angular_velocity)

        return {
            'detections': detections,
            'ranges': ranges,
            'accel': accel,
            'gyro': gyro,
        }


class SensorSimulator:
    """Add realistic noise to physics engine sensor outputs."""

    def __init__(self, device='cpu'):
        self.device = device

    def add_vision_noise(self, objects, noise_std=0.1):
        """
        Add Gaussian noise to detected objects.

        Args:
            objects: list of (x, y, z, class_id, confidence)
            noise_std: std dev of Gaussian noise

        Returns:
            noisy_objects: list of tuples with noise added
        """
        noisy_objects = []
        for obj in objects:
            x, y, z, cls_id, conf = obj
            x_noisy = x + np.random.normal(0, noise_std)
            y_noisy = y + np.random.normal(0, noise_std)
            z_noisy = z + np.random.normal(0, noise_std)
            # Confidence slightly degrades
            conf_noisy = max(0.1, conf - np.random.uniform(0, 0.1))
            noisy_objects.append((x_noisy, y_noisy, z_noisy, cls_id, conf_noisy))
        return noisy_objects

    def add_lidar_noise(self, readings, noise_std=0.05):
        """Add noise to lidar distance readings."""
        noisy = []
        for angle, dist in readings:
            dist_noisy = max(0.0, dist + np.random.normal(0, noise_std * dist))
            noisy.append((angle, dist_noisy))
        return noisy

    def add_imu_noise(self, imu_dict):
        """Add drift and noise to IMU readings."""
        noisy_imu = {}
        for key in ['accel', 'gyro', 'mag']:
            val = np.array(imu_dict.get(key, (0, 0, 0)))
            noise = np.random.normal(0, 0.01, 3)
            noisy_imu[key] = tuple(val + noise)
        return noisy_imu

    def apply_noise(self, sensor_dict):
        """Apply all noise transformations."""
        return {
            'vision': self.add_vision_noise(sensor_dict['vision']),
            'lidar': self.add_lidar_noise(sensor_dict['lidar']),
            'imu': self.add_imu_noise(sensor_dict['imu']),
            'motor_feedback': sensor_dict.get('motor_feedback', {}),
            'ram': sensor_dict['ram'],
            'position': sensor_dict['position'],
            'time': sensor_dict['time'],
        }
