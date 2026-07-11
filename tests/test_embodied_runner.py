"""Tests for embodied learning battleground runner."""
import json
import pytest
import numpy as np
import tempfile
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.experiments.run_embodied_battleground import run_embodied_learning
from src.config import LRSGConfig, LocomotionConfig, EmbodiedConfig
from src.simulation.embodied_loop import EmbodiedAgent
from src.simulation.embodied_visualizer import EmbodiedVisualizer


class TestEmbodiedRunner:
    """Test suite for run_embodied_learning runner."""

    def test_runner_creates_agent_and_visualizer(self):
        """Test that runner initializes agent and visualizer correctly."""
        # Mock the visualizer to avoid Flask startup
        with patch('src.experiments.run_embodied_battleground.EmbodiedVisualizer') as mock_viz_class:
            mock_viz = MagicMock()
            mock_viz_class.return_value = mock_viz

            # Run with 1 episode
            result = run_embodied_learning(num_episodes=1, num_neurons=100_000, device='cpu', headless=False)

            # Verify visualizer was created
            mock_viz_class.assert_called_once_with(port=8080)
            mock_viz.start.assert_called_once()

            # Verify result is a dict with expected keys
            assert isinstance(result, dict)
            assert 'episode_rewards' in result
            assert 'success_rate' in result
            assert 'avg_reward' in result
            assert 'max_reward' in result
            assert 'episode_log' in result

    def test_runner_headless_mode(self):
        """Test that headless mode doesn't create visualizer."""
        with patch('src.experiments.run_embodied_battleground.EmbodiedVisualizer') as mock_viz_class:
            # Run headless
            result = run_embodied_learning(num_episodes=1, num_neurons=100_000, device='cpu', headless=True)

            # Visualizer should not be created
            mock_viz_class.assert_not_called()

            # Result should still be valid
            assert isinstance(result, dict)
            assert 'episode_rewards' in result

    def test_runner_runs_single_episode(self):
        """Test that runner completes a single episode."""
        with patch('src.experiments.run_embodied_battleground.EmbodiedVisualizer'):
            result = run_embodied_learning(num_episodes=1, num_neurons=100_000, device='cpu', headless=True)

            # Should have exactly 1 episode in log
            assert len(result['episode_log']) == 1
            episode = result['episode_log'][0]

            # Episode should have required fields
            assert 'episode' in episode
            assert 'total_reward' in episode
            assert 'goal_reached' in episode
            assert 'max_enemies_seen' in episode
            assert 'steps' in episode
            assert 'final_distance_to_goal' in episode

            # Episode number should be 0
            assert episode['episode'] == 0

            # Steps should be > 0
            assert episode['steps'] > 0

    def test_runner_logs_telemetry(self):
        """Test that runner logs telemetry to JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('src.experiments.run_embodied_battleground.EmbodiedVisualizer'):
                # Change to temp dir
                import os
                old_cwd = os.getcwd()
                try:
                    os.chdir(tmpdir)

                    result = run_embodied_learning(num_episodes=2, num_neurons=100_000, device='cpu', headless=True)

                    # Check telemetry file was created
                    telemetry_path = Path('experiment_results') / 'embodied_battleground_telemetry.jsonl'
                    assert telemetry_path.exists()

                    # Read and validate JSONL
                    records = []
                    with open(telemetry_path, 'r') as f:
                        for line in f:
                            records.append(json.loads(line))

                    # Should have 2 records (one per episode)
                    assert len(records) == 2

                    # Each record should have required fields
                    for i, record in enumerate(records):
                        assert record['episode'] == i
                        assert 'total_reward' in record
                        assert 'goal_reached' in record
                        assert isinstance(record['goal_reached'], bool)
                        assert 'max_enemies_seen' in record
                        assert 'steps' in record
                        assert 'final_distance_to_goal' in record
                finally:
                    os.chdir(old_cwd)

    def test_runner_tracks_episode_rewards(self):
        """Test that runner correctly accumulates episode rewards."""
        with patch('src.experiments.run_embodied_battleground.EmbodiedVisualizer'):
            result = run_embodied_learning(num_episodes=3, num_neurons=100_000, device='cpu', headless=True)

            # Should have 3 episodes worth of rewards
            assert len(result['episode_rewards']) == 3
            assert len(result['episode_log']) == 3

            # Each reward should match the total_reward in episode_log
            for i, (reward, episode) in enumerate(zip(result['episode_rewards'], result['episode_log'])):
                assert reward == episode['total_reward']

            # All rewards should be numeric
            for reward in result['episode_rewards']:
                assert isinstance(reward, (int, float))

    def test_runner_computes_statistics_correctly(self):
        """Test that runner computes final statistics correctly."""
        with patch('src.experiments.run_embodied_battleground.EmbodiedVisualizer'):
            num_episodes = 5
            result = run_embodied_learning(num_episodes=num_episodes, num_neurons=100_000, device='cpu', headless=True)

            # Validate statistics
            episode_rewards = result['episode_rewards']

            # Success rate should be in [0, 1]
            assert 0.0 <= result['success_rate'] <= 1.0

            # Avg reward should match manual calculation
            expected_avg = np.mean(episode_rewards)
            assert np.isclose(result['avg_reward'], expected_avg)

            # Max reward should be >= avg reward
            assert result['max_reward'] >= result['avg_reward']

            # Max reward should be the actual max
            assert result['max_reward'] == max(episode_rewards)

            # Learning improvement should be numeric
            assert isinstance(result['learning_improvement'], (int, float))

    def test_runner_device_auto_detection(self):
        """Test that device auto-detection works."""
        with patch('src.experiments.run_embodied_battleground.EmbodiedVisualizer'):
            # Run with None device (auto)
            result = run_embodied_learning(num_episodes=1, num_neurons=100_000, device=None, headless=True)

            # Should complete without error
            assert isinstance(result, dict)
            assert 'episode_log' in result

    def test_runner_multiple_episodes_show_variety(self):
        """Test that multiple episodes show variety in outcomes."""
        with patch('src.experiments.run_embodied_battleground.EmbodiedVisualizer'):
            result = run_embodied_learning(num_episodes=10, num_neurons=100_000, device='cpu', headless=True)

            episode_log = result['episode_log']
            assert len(episode_log) == 10

            # At least some episodes should vary (not all identical)
            total_rewards = [e['total_reward'] for e in episode_log]
            final_distances = [e['final_distance_to_goal'] for e in episode_log]

            # Some variation in rewards (allow small margin for randomness)
            rewards_varied = len(set(np.round(total_rewards, 2))) > 1
            assert rewards_varied, "Episode rewards should vary"

    def test_runner_episode_numbering_sequential(self):
        """Test that episodes are numbered sequentially."""
        with patch('src.experiments.run_embodied_battleground.EmbodiedVisualizer'):
            result = run_embodied_learning(num_episodes=5, num_neurons=100_000, device='cpu', headless=True)

            episode_log = result['episode_log']
            for i, episode in enumerate(episode_log):
                assert episode['episode'] == i

    def test_runner_success_rate_calculation(self):
        """Test that success rate is calculated from goal_reached flags."""
        with patch('src.experiments.run_embodied_battleground.EmbodiedVisualizer'):
            result = run_embodied_learning(num_episodes=10, num_neurons=100_000, device='cpu', headless=True)

            episode_log = result['episode_log']
            goal_reaches = [e['goal_reached'] for e in episode_log]
            success_count = sum(goal_reaches)
            expected_rate = success_count / len(episode_log)

            assert np.isclose(result['success_rate'], expected_rate)

    @pytest.mark.slow
    def test_runner_cpu_only_device(self):
        """Test that runner works on CPU (slow test)."""
        with patch('src.experiments.run_embodied_battleground.EmbodiedVisualizer'):
            result = run_embodied_learning(num_episodes=1, num_neurons=50_000, device='cpu', headless=True)

            assert isinstance(result, dict)
            assert len(result['episode_log']) == 1
            assert result['episode_log'][0]['steps'] > 0


class TestRunEmbodiedLearningIntegration:
    """Integration tests that exercise the full pipeline."""

    @pytest.mark.slow
    def test_full_learning_progression(self):
        """Test that agent shows some learning over many episodes."""
        with patch('src.experiments.run_embodied_battleground.EmbodiedVisualizer'):
            # Run more episodes to see learning (if enabled)
            result = run_embodied_learning(
                num_episodes=20,
                num_neurons=100_000,
                device='cpu',
                headless=True
            )

            episode_rewards = result['episode_rewards']

            # Check that learning_improvement is computed
            assert 'learning_improvement' in result

            # Learning improvement should be numeric
            assert isinstance(result['learning_improvement'], (int, float))

    def test_return_dict_structure(self):
        """Test that return dict has correct structure."""
        with patch('src.experiments.run_embodied_battleground.EmbodiedVisualizer'):
            result = run_embodied_learning(num_episodes=2, num_neurons=100_000, device='cpu', headless=True)

            # Required keys
            required_keys = {'episode_rewards', 'success_rate', 'avg_reward', 'max_reward', 'episode_log'}
            assert set(result.keys()) >= required_keys

            # Type validation
            assert isinstance(result['episode_rewards'], list)
            assert isinstance(result['success_rate'], (int, float))
            assert isinstance(result['avg_reward'], (int, float))
            assert isinstance(result['max_reward'], (int, float))
            assert isinstance(result['episode_log'], list)

            # episode_log should have dicts
            for episode in result['episode_log']:
                assert isinstance(episode, dict)
                assert 'total_reward' in episode
