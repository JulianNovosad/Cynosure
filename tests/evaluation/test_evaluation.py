import sys
import os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import time
from src.config import AgentConfig, LRSGConfig, ArenaConfig
from src.lrsg import LRSGraph
from src.arena import Environment, RAMNode
from src.agent import Agent
from src.harness.security import AnnouncerSanitizer
from src.evaluation import EvaluationSuite

_sanitizer = AnnouncerSanitizer()


def test_evaluation_suite_init():
    """Test that EvaluationSuite initializes correctly."""
    suite = EvaluationSuite()
    assert suite is not None
    assert hasattr(suite, 'eval_locomotion_stability')
    assert hasattr(suite, 'eval_navigation_success')
    assert hasattr(suite, 'eval_transfer_readiness')


def test_locomotion_stability_eval():
    """Test locomotion stability evaluation."""
    # Create small agent (2000 neurons, 30-tick episode)
    lrsg_cfg = LRSGConfig(num_neurons=2000, num_input_neurons=100, num_output_neurons=12)
    brain = LRSGraph(lrsg_cfg)
    agent = Agent(0, (50, 50, 50), brain, AgentConfig(), lrsg_cfg, sanitizer=_sanitizer)
    env = Environment(ArenaConfig(grid_size=(100, 100, 100)))

    suite = EvaluationSuite()
    result = suite.eval_locomotion_stability(agent, env, num_episodes=2)

    # Check required keys
    assert 'avg_tick_time' in result
    assert 'collision_rate' in result
    assert 'smoothness' in result
    assert 'num_ticks' in result

    # Check value ranges
    assert isinstance(result['avg_tick_time'], (int, float))
    assert result['avg_tick_time'] >= 0
    assert 0 <= result['collision_rate'] <= 1
    assert 0 <= result['smoothness'] <= 1
    assert result['num_ticks'] >= 0


def test_navigation_success_eval():
    """Test navigation success evaluation."""
    # Create small agent
    lrsg_cfg = LRSGConfig(num_neurons=2000, num_input_neurons=100, num_output_neurons=12)
    brain = LRSGraph(lrsg_cfg)
    agent = Agent(0, (50, 50, 50), brain, AgentConfig(), lrsg_cfg, sanitizer=_sanitizer)
    env = Environment(ArenaConfig(grid_size=(100, 100, 100)))

    suite = EvaluationSuite()
    result = suite.eval_navigation_success(agent, env, num_episodes=2)

    # Check required keys
    assert 'success_rate' in result
    assert 'avg_steps_to_goal' in result
    assert 'learning_improvement' in result
    assert 'early_success_rate' in result
    assert 'late_success_rate' in result

    # Check value ranges
    assert 0 <= result['success_rate'] <= 1
    assert result['avg_steps_to_goal'] >= 0
    assert isinstance(result['learning_improvement'], (int, float))
    assert 0 <= result['early_success_rate'] <= 1
    assert 0 <= result['late_success_rate'] <= 1


def test_transfer_readiness():
    """Test transfer readiness evaluation."""
    suite = EvaluationSuite()
    result = suite.eval_transfer_readiness()

    # Check required keys
    assert 'determinism_check' in result
    assert 'config_consistency' in result
    assert 'reproducibility_score' in result

    # Check value types
    assert isinstance(result['determinism_check'], bool)
    assert isinstance(result['config_consistency'], bool)
    assert 0 <= result['reproducibility_score'] <= 1
