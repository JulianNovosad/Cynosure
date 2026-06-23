"""Unit tests for LocomotionController."""
import pytest
import math
from src.simulation.locomotion import LocomotionController


class TestLocomotionController:
    """Test suite for differential-drive locomotion controller."""

    def test_differential_drive_forward(self):
        """Test forward motion with equal motors."""
        controller = LocomotionController(
            wheelbase=0.15, wheel_radius=0.05, max_motor_speed=1.0
        )
        lin_vel, ang_vel = controller.motors_to_velocity(1.0, 1.0)
        assert lin_vel > 0.0, "Linear velocity should be positive"
        assert abs(ang_vel) < 0.01, "Angular velocity should be ~0 for straight motion"

    def test_differential_drive_turn_left(self):
        """Test left turn with left motor slower."""
        controller = LocomotionController(
            wheelbase=0.15, wheel_radius=0.05, max_motor_speed=1.0
        )
        lin_vel, ang_vel = controller.motors_to_velocity(0.5, 1.0)
        assert lin_vel > 0.0, "Should still move forward"
        assert ang_vel > 0.0, "Should turn left (positive angular velocity)"

    def test_step_forward_motion(self):
        """Test step function moves agent forward."""
        controller = LocomotionController(
            wheelbase=0.15, wheel_radius=0.05, max_motor_speed=1.0
        )
        dx, dy, dyaw = controller.step(linear_vel=0.5, angular_vel=0.0, dt=0.1)
        assert abs(dx - 0.05) < 0.01, f"Expected dx≈0.05, got {dx}"
        assert abs(dy) < 0.01, f"Expected dy≈0, got {dy}"
        assert abs(dyaw) < 0.01, f"Expected dyaw≈0, got {dyaw}"
