"""
Abstract I/O interface for simulation and hardware.

Defines the contract that both SimIO (Pybullet) and RPIIO (Raspberry Pi 5)
must implement, ensuring identical sensor/motor interfaces across both.
"""

from abc import ABC, abstractmethod


class IOInterface(ABC):
    """Abstract base class defining the I/O interface for all implementations."""

    @abstractmethod
    def read_sensors(self) -> dict:
        """
        Read sensor data from the system.

        Returns:
            dict: Sensor readings containing:
                - 'vision': list of detected objects (x, y, z, class_id, confidence)
                - 'lidar': list of (angle, distance) tuples
                - 'imu': dict with 'accel', 'gyro', 'mag' keys (3D tuples)
                - 'motor_feedback': dict with 'left_speed', 'right_speed'
                - 'ram': float, resource availability
                - 'position': (x, y, z) tuple
                - 'time': int, milliseconds
        """
        pass

    @abstractmethod
    def write_motors(self, action: int) -> None:
        """
        Send motor command to the system.

        Args:
            action: Discrete action in range [0, 6]:
                0: STOP
                1: MOVE_FORWARD
                2: MOVE_BACKWARD
                3: TURN_LEFT
                4: TURN_RIGHT
                5: ATTACK
                6: IDLE
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Cleanup and release resources."""
        pass
