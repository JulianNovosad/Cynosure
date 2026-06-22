import pytest
import numpy as np
from src.simulation.sensors import SensorSimulator


def test_vision_noise_shape_preserved():
    """Test that vision noise doesn't change object count."""
    sim = SensorSimulator()
    objects = [(1.0, 2.0, 3.0, 1, 0.9), (4.0, 5.0, 6.0, 2, 0.85)]

    noisy = sim.add_vision_noise(objects, noise_std=0.1)

    assert len(noisy) == len(objects)
    for noisy_obj in noisy:
        assert len(noisy_obj) == 5
        assert 0.0 <= noisy_obj[4] <= 1.0  # Confidence in [0, 1]


def test_lidar_noise_positive():
    """Test that lidar distances remain non-negative after noise."""
    sim = SensorSimulator()
    readings = [(float(i), 10.0) for i in range(360)]

    np.random.seed(42)
    noisy = sim.add_lidar_noise(readings, noise_std=0.5)

    for angle, dist in noisy:
        assert dist >= 0.0, f"Distance {dist} should be non-negative"


def test_apply_noise_preserves_structure():
    """Test that apply_noise returns valid sensor dict."""
    sim = SensorSimulator()
    sensor_dict = {
        'vision': [(1.0, 2.0, 3.0, 1, 0.9)],
        'lidar': [(0.0, 10.0), (np.pi, 15.0)],
        'imu': {'accel': (0.0, 0.0, 9.8), 'gyro': (0.0, 0.0, 0.0), 'mag': (1.0, 0.0, 0.0)},
        'motor_feedback': {'left_speed': 1.0, 'right_speed': 0.95},
        'ram': 50.0,
        'position': (10.0, 20.0, 0.5),
        'time': 100,
    }

    noisy = sim.apply_noise(sensor_dict)

    assert 'vision' in noisy
    assert 'lidar' in noisy
    assert 'imu' in noisy
    assert 'ram' in noisy
