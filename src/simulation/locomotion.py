"""Locomotion controller for differential-drive kinematics."""
import math
from typing import Tuple


class LocomotionController:
    """
    Differential-drive locomotion controller.

    Maps motor commands to velocities and integrates kinematics over time steps.
    """

    def __init__(
        self,
        wheelbase: float,
        wheel_radius: float,
        max_motor_speed: float,
        friction_coeff: float = 1.0,
    ):
        """
        Initialize the locomotion controller.

        Args:
            wheelbase: Distance between left and right wheels (meters)
            wheel_radius: Radius of the drive wheels (meters)
            max_motor_speed: Maximum motor speed in m/s
            friction_coeff: Friction coefficient for motor control (default 1.0)
        """
        self.wheelbase = wheelbase
        self.wheel_radius = wheel_radius
        self.max_motor_speed = max_motor_speed
        self.friction_coeff = friction_coeff

    def motors_to_velocity(self, left_motor: float, right_motor: float) -> Tuple[float, float]:
        """
        Convert motor commands to linear and angular velocities.

        Differential-drive kinematics:
        - Linear velocity = (left_wheel_velocity + right_wheel_velocity) / 2
        - Angular velocity = (right_wheel_velocity - left_wheel_velocity) / wheelbase

        Args:
            left_motor: Left motor command in [-1, 1]
            right_motor: Right motor command in [-1, 1]

        Returns:
            Tuple of (linear_vel, angular_vel)
        """
        # Map motor commands [-1, 1] to wheel velocities
        left_wheel_vel = left_motor * self.max_motor_speed * self.friction_coeff
        right_wheel_vel = right_motor * self.max_motor_speed * self.friction_coeff

        # Compute linear and angular velocities
        linear_vel = (left_wheel_vel + right_wheel_vel) / 2.0
        angular_vel = (right_wheel_vel - left_wheel_vel) / self.wheelbase

        return linear_vel, angular_vel

    def step(self, linear_vel: float, angular_vel: float, dt: float) -> Tuple[float, float, float]:
        """
        Integrate kinematics over one time step.

        For a differential-drive robot:
        - If angular_vel ≈ 0: straight line motion
        - If angular_vel ≠ 0: circular arc motion

        Args:
            linear_vel: Linear velocity (m/s)
            angular_vel: Angular velocity (rad/s)
            dt: Time step (s)

        Returns:
            Tuple of (dx, dy, dyaw) - displacement and rotation change
        """
        # Small threshold to treat near-zero angular velocity as straight motion
        threshold = 1e-6

        if abs(angular_vel) < threshold:
            # Straight line motion
            dx = linear_vel * dt
            dy = 0.0
            dyaw = 0.0
        else:
            # Circular arc motion
            radius = linear_vel / angular_vel
            dyaw = angular_vel * dt
            # For small angles, use exact circle arc formulas
            dx = radius * math.sin(dyaw)
            dy = radius * (1.0 - math.cos(dyaw))

        return dx, dy, dyaw
