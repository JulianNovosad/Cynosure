"""
Full-pipeline integration test for Cynosure.

Tests the complete sensor-to-learning pipeline:
1. Sensors capture environment state
2. SemanticEncoder converts raw sensors to 57D semantic state
3. Semantic state integrates into LRSG context buffer
4. Agent learns via Hebbian plasticity without NaN explosions

This test validates end-to-end stability and correctness over 200 ticks.
"""

import torch
import pytest
from src.config import Config, SimulationConfig
from src.simulation.loop import Simulation


def test_full_pipeline_sensor_to_learning():
    """
    Comprehensive full-pipeline integration test.

    Setup:
    - 1 agent, physics enabled, semantic encoding enabled
    - Run 200 ticks with validation every 10 ticks

    Validates:
    - Semantic state shape (57,), dtype float32, range [-1, 1]
    - No NaNs in semantic state, context buffer, or agent RAM
    - Agent remains alive throughout
    - Learning occurs without explosions
    """

    # 1. SETUP: Create config with physics and semantic enabled
    config = Config()
    config.simulation.num_agents = 1
    config.simulation.use_physics = True
    config.simulation.use_semantic = True
    config.simulation.headless = True
    config.simulation.max_ticks = 200

    # 2. Create simulation
    sim = Simulation(config)
    assert sim.semantic_encoder is not None, "Semantic encoder should be initialized"
    assert sim.sensor_simulator is not None, "Sensor simulator should be initialized"

    agent = sim.agents[0]
    assert agent.alive, "Agent should start alive"

    # 3. EXECUTION: Run 200 ticks with validation every 10 ticks
    for tick in range(200):
        sim.tick()

        # Verify checkpoint validations every 10 ticks
        if (tick + 1) % 10 == 0:
            checkpoint_tick = tick + 1

            # Get current agent state
            assert agent.alive, f"Agent should be alive at tick {checkpoint_tick}"
            assert not torch.isnan(torch.tensor(agent.ram)), \
                f"Agent RAM should not be NaN at tick {checkpoint_tick}, got {agent.ram}"

            # Verify semantic state properties if it has been set
            if agent.semantic_state is not None:
                semantic_state = agent.semantic_state

                # Check shape
                assert semantic_state.shape == (57,), \
                    f"Semantic state shape should be (57,) at tick {checkpoint_tick}, got {semantic_state.shape}"

                # Check dtype
                assert semantic_state.dtype == torch.float32, \
                    f"Semantic state dtype should be float32 at tick {checkpoint_tick}, got {semantic_state.dtype}"

                # Check range [-1, 1]
                assert torch.all(semantic_state >= -1.0) and torch.all(semantic_state <= 1.0), \
                    f"Semantic state should be in [-1, 1] at tick {checkpoint_tick}, " \
                    f"got min={semantic_state.min()}, max={semantic_state.max()}"

                # Check for NaNs
                assert not torch.isnan(semantic_state).any(), \
                    f"Semantic state contains NaN at tick {checkpoint_tick}"

            # Verify context buffer has no NaNs (last 57 dims are semantic state)
            assert not torch.isnan(agent.context_buffer).any(), \
                f"Context buffer contains NaN at tick {checkpoint_tick}"

            # Verify brain state is healthy
            assert not torch.isnan(agent.brain.activations).any(), \
                f"Brain activations contain NaN at tick {checkpoint_tick}"
            assert not torch.isnan(agent.brain.out_weights).any(), \
                f"Brain weights contain NaN at tick {checkpoint_tick}"

    # 4. VERIFICATION AFTER LEARNING: Check final agent state

    # Final RAM should not be NaN
    final_ram = agent.ram
    assert not torch.isnan(torch.tensor(final_ram)), \
        f"Final RAM should not be NaN, got {final_ram}"

    # Context buffer should have no NaNs
    assert not torch.isnan(agent.context_buffer).any(), \
        "Final context buffer should have no NaNs"

    # Agent should still be alive (or at least, if dead, should be graceful)
    # We don't require agent to be alive at the end, but RAM should be valid
    if agent.alive:
        assert agent.ram > 0, "Alive agent should have positive RAM"

    # 5. CLEANUP
    sim.stop()

    print(f"✓ Full pipeline test PASSED")
    print(f"  - Completed 200 ticks")
    print(f"  - Agent final state: alive={agent.alive}, ram={agent.ram:.1f}")
    print(f"  - Validated semantic state at 20 checkpoints (every 10 ticks)")
    print(f"  - No NaNs detected in semantic, context, or brain state")


def test_full_pipeline_multi_agent():
    """
    Extended full-pipeline test with 2 agents to verify interaction stability.
    Runs for 100 ticks and validates multi-agent semantic encoding.
    """

    config = Config()
    config.simulation.num_agents = 2
    config.simulation.use_physics = True
    config.simulation.use_semantic = True
    config.simulation.headless = True
    config.simulation.max_ticks = 100

    sim = Simulation(config)

    alive_at_end = 0

    for tick in range(100):
        sim.tick()

        # Every 10 ticks, verify both agents
        if (tick + 1) % 10 == 0:
            for i, agent in enumerate(sim.agents):
                if agent.alive:
                    # Check context buffer
                    assert not torch.isnan(agent.context_buffer).any(), \
                        f"Agent {i} context buffer contains NaN at tick {tick + 1}"

                    # Check brain
                    assert not torch.isnan(agent.brain.activations).any(), \
                        f"Agent {i} brain activations contain NaN at tick {tick + 1}"

                    # Check RAM
                    assert not torch.isnan(torch.tensor(agent.ram)), \
                        f"Agent {i} RAM is NaN at tick {tick + 1}"

    alive_at_end = sum(1 for a in sim.agents if a.alive)
    assert alive_at_end >= 1, "At least one agent should survive 100 ticks"

    sim.stop()

    print(f"✓ Multi-agent pipeline test PASSED")
    print(f"  - Completed 100 ticks with 2 agents")
    print(f"  - Agents alive at end: {alive_at_end}")


def test_full_pipeline_without_semantic():
    """
    Backward compatibility test: full pipeline works without semantic encoding.
    Verifies that disabling semantic doesn't break learning.
    """

    config = Config()
    config.simulation.num_agents = 1
    config.simulation.use_physics = True
    config.simulation.use_semantic = False  # Disabled
    config.simulation.headless = True
    config.simulation.max_ticks = 100

    sim = Simulation(config)

    # Semantic encoder should not be initialized
    assert sim.semantic_encoder is None, "Semantic encoder should be None when disabled"

    agent = sim.agents[0]

    for tick in range(100):
        sim.tick()

        # Verify no NaNs even without semantic encoding
        if agent.alive:
            assert not torch.isnan(torch.tensor(agent.ram)), \
                f"Agent RAM should not be NaN at tick {tick + 1}"
            assert not torch.isnan(agent.brain.activations).any(), \
                f"Brain activations should not contain NaN at tick {tick + 1}"

    sim.stop()

    print(f"✓ Backward compatibility test PASSED")
    print(f"  - Completed 100 ticks without semantic encoding")
    print(f"  - Agent final state: alive={agent.alive}, ram={agent.ram:.1f}")


if __name__ == "__main__":
    test_full_pipeline_sensor_to_learning()
    test_full_pipeline_multi_agent()
    test_full_pipeline_without_semantic()
    print("\n✓ All full-pipeline tests PASSED")
