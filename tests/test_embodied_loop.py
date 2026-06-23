"""Unit tests for EmbodiedAgent class (physics + LRSG + reward integration)."""
import pytest
import numpy as np
import torch
from src.simulation.embodied_loop import EmbodiedAgent
from src.config import LRSGConfig, LocomotionConfig, EmbodiedConfig


class TestEmbodiedAgent:
    """Test suite for embodied agent integration."""

    def test_embodied_agent_initialization(self):
        """Test that EmbodiedAgent initializes without error."""
        lrsg_config = LRSGConfig(num_neurons=5000)
        loc_config = LocomotionConfig()
        emb_config = EmbodiedConfig(grid_size=10, num_obstacles=2)

        agent = EmbodiedAgent(
            lrsg_config=lrsg_config,
            locomotion_config=loc_config,
            embodied_config=emb_config,
            device='cpu'
        )

        assert agent is not None
        assert hasattr(agent, 'step'), "EmbodiedAgent should have step method"
        assert hasattr(agent, 'lrsg'), "EmbodiedAgent should have LRSG component"

    def test_embodied_agent_single_step(self):
        """Test that one step produces reward and state."""
        lrsg_config = LRSGConfig(num_neurons=5000)
        loc_config = LocomotionConfig()
        emb_config = EmbodiedConfig(grid_size=10, num_obstacles=1)

        agent = EmbodiedAgent(
            lrsg_config=lrsg_config,
            locomotion_config=loc_config,
            embodied_config=emb_config,
            device='cpu'
        )

        result = agent.step()

        assert 'reward' in result, "step() should return dict with 'reward'"
        assert 'distance_to_goal' in result
        assert 'is_collision' in result
        assert 'is_goal_reached' in result
        assert isinstance(result['reward'], (float, np.floating)), "Reward should be scalar"

    def test_embodied_agent_episode_run(self):
        """Test that agent can run a full episode."""
        lrsg_config = LRSGConfig(num_neurons=2000)
        loc_config = LocomotionConfig()
        emb_config = EmbodiedConfig(grid_size=10, num_obstacles=1, episode_length=20)

        agent = EmbodiedAgent(
            lrsg_config=lrsg_config,
            locomotion_config=loc_config,
            embodied_config=emb_config,
            device='cpu'
        )

        rewards = []
        for tick in range(20):
            result = agent.step()
            rewards.append(result['reward'])

        assert len(rewards) == 20, "Should complete 20 steps"
        assert all(isinstance(r, (float, np.floating)) for r in rewards), "All rewards should be scalars"
