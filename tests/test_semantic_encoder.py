import torch
import pytest
from src.lrsg.semantic_encoder import SemanticEncoder


def test_semantic_encoder_output_shape():
    """Test that encoder outputs 57D tensor."""
    encoder = SemanticEncoder(device='cpu')

    sensor_dict = {
        'vision': [(10.0, 5.0, 15.0, 1, 0.95), (20.0, 10.0, 30.0, 2, 0.87)],  # 2 objects
        'lidar': [(i * 1.0, 5.0 + i * 0.1) for i in range(360)],  # 360 rays
        'imu': {'accel': (0.1, -0.05, 9.8), 'gyro': (0.01, 0.02, 0.0), 'mag': (1.0, 0.5, -0.3)},
        'motor_feedback': {'left_speed': 1.0, 'right_speed': 0.95},
        'ram': 75.0,
        'position': (10.0, 5.0, 0.5),
        'time': 100,
    }

    output = encoder.forward(sensor_dict)

    assert isinstance(output, torch.Tensor), "Output must be torch.Tensor"
    assert output.shape == (57,), f"Expected shape (57,), got {output.shape}"
    assert output.dtype == torch.float32, f"Expected dtype float32, got {output.dtype}"
    assert torch.all(output >= -1.0) and torch.all(output <= 1.0), "Output must be in range [-1, 1]"


def test_semantic_encoder_deterministic():
    """Test that same input produces same output."""
    encoder = SemanticEncoder(device='cpu')
    encoder.eval()  # No dropout

    sensor_dict = {
        'vision': [(10.0, 5.0, 15.0, 1, 0.95)],
        'lidar': [(i * 1.0, 5.0) for i in range(360)],
        'imu': {'accel': (0.0, 0.0, 9.8), 'gyro': (0.0, 0.0, 0.0), 'mag': (1.0, 0.0, 0.0)},
        'motor_feedback': {'left_speed': 0.0, 'right_speed': 0.0},
        'ram': 50.0,
        'position': (0.0, 0.0, 0.0),
        'time': 0,
    }

    with torch.no_grad():
        out1 = encoder(sensor_dict)
        out2 = encoder(sensor_dict)

    assert torch.allclose(out1, out2, atol=1e-6), "Same input should produce same output"
