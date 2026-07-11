"""
Integration tests for embodied learning system (episode runner).

Tests the complete episode runner with visualization and telemetry.
"""
import pytest
import json
import os
import tempfile
from pathlib import Path

from src.config import LRSGConfig, LocomotionConfig, EmbodiedConfig
from src.simulation.embodied_loop import EmbodiedAgent
from src.simulation.embodied_visualizer import EmbodiedVisualizer


class TestEmbodiedRunnerInitialization:
    """Test suite for embodied runner initialization."""

    def test_embodied_agent_creation(self):
        """Test that EmbodiedAgent creates successfully."""
        lrsg_config = LRSGConfig(num_neurons=100_000)
        loc_config = LocomotionConfig()
        emb_config = EmbodiedConfig()

        agent = EmbodiedAgent(
            lrsg_config=lrsg_config,
            locomotion_config=loc_config,
            embodied_config=emb_config,
            device='cpu'
        )

        assert agent is not None
        assert agent.lrsg is not None
        assert agent.env is not None
        assert agent.agent_pos is not None
        assert agent.goal_pos is not None

    def test_visualizer_creation(self):
        """Test that EmbodiedVisualizer creates successfully."""
        visualizer = EmbodiedVisualizer(port=8081)

        assert visualizer is not None
        assert hasattr(visualizer, 'update_state')
        assert hasattr(visualizer, 'log_episode_reward')
        assert hasattr(visualizer, 'start')


class TestEmbodiedRunnerEpisode:
    """Test suite for single episode execution."""

    def test_single_episode_execution(self):
        """Test that a single episode runs and produces valid data."""
        lrsg_config = LRSGConfig(num_neurons=50_000)
        loc_config = LocomotionConfig()
        emb_config = EmbodiedConfig(episode_length=100)

        agent = EmbodiedAgent(
            lrsg_config=lrsg_config,
            locomotion_config=loc_config,
            embodied_config=emb_config,
            device='cpu'
        )

        # Run one episode
        episode_reward = 0.0
        goal_reached = False
        max_enemies = 0
        steps_taken = 0

        agent._reset_episode()
        while agent.tick_count < emb_config.episode_length:
            step_result = agent.step()
            episode_reward += step_result['reward']
            max_enemies = max(max_enemies, step_result['num_enemies'])
            steps_taken += 1

            if step_result['is_goal_reached']:
                goal_reached = True
                break

            if step_result['episode_done']:
                # Episode can end early (collision timeout), which resets
                # agent.tick_count to 0 internally — stop the loop here too.
                break

        # Verify episode produced data
        assert steps_taken > 0, "Episode should have at least one step"
        assert isinstance(episode_reward, float), "Reward should be float"
        assert isinstance(goal_reached, bool), "goal_reached should be bool"
        assert max_enemies >= 0, "max_enemies should be >= 0"

    def test_episode_step_returns_valid_data(self):
        """Test that step() returns all required fields."""
        lrsg_config = LRSGConfig(num_neurons=50_000)
        loc_config = LocomotionConfig()
        emb_config = EmbodiedConfig()

        agent = EmbodiedAgent(
            lrsg_config=lrsg_config,
            locomotion_config=loc_config,
            embodied_config=emb_config,
            device='cpu'
        )

        # Take a single step
        step_result = agent.step()

        # Verify all required fields are present
        required_fields = {
            'reward': float,
            'distance_to_goal': float,
            'is_collision': bool,
            'is_goal_reached': bool,
            'episode_done': bool,
            'action': int,
            'num_enemies': int,
        }

        for field, expected_type in required_fields.items():
            assert field in step_result, f"step_result missing field: {field}"
            assert isinstance(step_result[field], expected_type), \
                f"step_result[{field}] should be {expected_type.__name__}"


class TestEmbodiedRunnerFull:
    """Test suite for full episode runner execution."""

    def test_five_episode_run(self):
        """Test that a 5-episode run completes and produces telemetry."""
        lrsg_config = LRSGConfig(num_neurons=50_000)
        loc_config = LocomotionConfig()
        emb_config = EmbodiedConfig(episode_length=50)

        agent = EmbodiedAgent(
            lrsg_config=lrsg_config,
            locomotion_config=loc_config,
            embodied_config=emb_config,
            device='cpu'
        )

        # Run 5 episodes
        episode_log = []
        num_episodes = 5

        for episode in range(num_episodes):
            agent._reset_episode()

            episode_data = {
                'episode': episode,
                'total_reward': 0.0,
                'goal_reached': False,
                'max_enemies_seen': 0,
                'steps': 0,
                'final_distance_to_goal': 0.0,
            }

            while agent.tick_count < emb_config.episode_length:
                step_result = agent.step()
                episode_data['total_reward'] += step_result['reward']
                episode_data['max_enemies_seen'] = max(
                    episode_data['max_enemies_seen'],
                    step_result['num_enemies']
                )
                episode_data['steps'] += 1
                episode_data['final_distance_to_goal'] = step_result['distance_to_goal']

                if step_result['is_goal_reached']:
                    episode_data['goal_reached'] = True
                    break

                if step_result['episode_done']:
                    break

            episode_log.append(episode_data)

        # Verify telemetry
        assert len(episode_log) == num_episodes, \
            f"Should have {num_episodes} episodes logged"

        for i, episode_data in enumerate(episode_log):
            assert episode_data['episode'] == i
            assert isinstance(episode_data['total_reward'], float)
            assert isinstance(episode_data['goal_reached'], bool)
            assert isinstance(episode_data['max_enemies_seen'], int)
            assert isinstance(episode_data['steps'], int)
            assert isinstance(episode_data['final_distance_to_goal'], float)
            assert episode_data['steps'] > 0, "Episode should have at least one step"

    def test_telemetry_file_creation(self):
        """Test that telemetry file can be created and written."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal telemetry
            episode_log = [
                {
                    'episode': 0,
                    'total_reward': 45.23,
                    'goal_reached': False,
                    'max_enemies_seen': 2,
                    'steps': 50,
                    'final_distance_to_goal': 1.5,
                },
                {
                    'episode': 1,
                    'total_reward': 67.89,
                    'goal_reached': True,
                    'max_enemies_seen': 1,
                    'steps': 45,
                    'final_distance_to_goal': 0.2,
                },
            ]

            # Write telemetry
            telemetry_path = Path(tmpdir) / 'test_telemetry.jsonl'
            with open(telemetry_path, 'w') as f:
                for record in episode_log:
                    f.write(json.dumps(record) + '\n')

            # Verify file exists
            assert telemetry_path.exists(), "Telemetry file should exist"

            # Verify contents
            with open(telemetry_path, 'r') as f:
                lines = f.readlines()
            assert len(lines) == 2, "Should have 2 lines of telemetry"

            # Verify JSON parsing
            for line in lines:
                record = json.loads(line)
                assert 'episode' in record
                assert 'total_reward' in record
                assert 'goal_reached' in record

    def test_visualizer_update_state(self):
        """Test that visualizer can be updated with state."""
        import numpy as np

        visualizer = EmbodiedVisualizer(port=8082)

        # Create dummy state
        agent_pos = (5.0, 5.0)
        goal_pos = (8.0, 8.0)
        enemies = [
            {'position': [6.0, 6.0], 'health': 10, 'max_health': 20, 'strength': 1.0},
            {'position': [7.0, 5.0], 'health': 15, 'max_health': 20, 'strength': 1.0},
        ]
        obstacles = [(2.0, 2.0), (3.0, 3.0)]
        semantic_state = np.zeros(64, dtype=np.float32)

        # Update state
        visualizer.update_state(
            agent_pos=agent_pos,
            goal_pos=goal_pos,
            enemies=enemies,
            obstacles=obstacles,
            episode=0,
            tick=10,
            current_reward=1.5,
            accumulated_reward=15.0,
            semantic_state=semantic_state,
        )

        # Verify state was updated
        assert visualizer._state['agent_pos'] == list(agent_pos)
        assert visualizer._state['goal_pos'] == list(goal_pos)
        assert len(visualizer._state['enemies']) == 2
        assert len(visualizer._state['obstacles']) == 2
        assert visualizer._state['episode'] == 0
        assert visualizer._state['tick'] == 10

    def test_visualizer_log_episode_reward(self):
        """Test that visualizer logs episode rewards."""
        visualizer = EmbodiedVisualizer(port=8083)

        # Log some episode rewards
        for episode in range(3):
            reward = 50.0 + episode * 20.0
            visualizer.log_episode_reward(episode, reward)

        # Verify logged rewards
        assert len(visualizer._state['episode_rewards']) == 3
        assert visualizer._state['episode_rewards'][0] == [0, 50.0]
        assert visualizer._state['episode_rewards'][1] == [1, 70.0]
        assert visualizer._state['episode_rewards'][2] == [2, 90.0]
