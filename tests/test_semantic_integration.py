import pytest
import torch
from src.lrsg.semantic_encoder import SemanticEncoder
from src.simulation.physics_world import PhysicsWorld
from src.simulation.sensors import SensorSimulator
from src.lrsg.graph import LRSGraph  # Existing
from src.config import LRSGConfig


def test_sensor_to_semantic_to_lrsg_pipeline():
    """Test full pipeline: physics sensors → semantic encoder → LRSG forward pass."""

    # Setup
    world = PhysicsWorld(num_agents=1, grid_size=50, use_gui=False)
    encoder = SemanticEncoder(device='cpu')
    sensor_sim = SensorSimulator(device='cpu')
    config = LRSGConfig()
    lrsg_graph = LRSGraph(config)

    # Step physics once to get sensor readings
    actions = {0: 1}  # Agent 0: MOVE_FORWARD
    raw_readings = world.step(actions)

    # Apply sensor noise
    noisy_readings = sensor_sim.apply_noise(raw_readings[0])

    # Encode to semantic state
    semantic_state = encoder.forward(noisy_readings)

    # Verify semantic state shape
    assert semantic_state.shape == (57,), f"Expected shape (57,), got {semantic_state.shape}"
    assert torch.all(semantic_state >= -1.0) and torch.all(semantic_state <= 1.0)

    # Pass through LRSG forward pass (this uses existing forward() logic)
    # The semantic state becomes part of context input to LRSG
    # (Actual integration with LRSG happens in Task 5)

    world.close()
