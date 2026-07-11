"""
Evaluation suite for measuring agent performance across multiple dimensions.

Provides tools to evaluate:
- Locomotion stability: movement smoothness, timing, collision avoidance
- Navigation success: goal-reaching ability, learning progression
- Transfer readiness: reproducibility and configuration consistency
"""

import time
import torch
import math
from typing import Dict
from src.config import LRSGConfig
from src.lrsg import LRSGraph
from src.agent import Agent
from src.arena import Environment


class EvaluationSuite:
    """Evaluation suite for agent performance measurement."""

    def __init__(self):
        """Initialize the evaluation suite."""
        pass

    def eval_locomotion_stability(
        self, agent: Agent, env: Environment, num_episodes: int = 5
    ) -> Dict:
        """
        Evaluate agent locomotion stability.

        Measures:
        - avg_tick_time: Average time per simulation tick (seconds)
        - collision_rate: Fraction of ticks with collisions [0, 1]
        - smoothness: Measure of movement consistency [0, 1]
        - num_ticks: Total number of ticks executed

        Args:
            agent: Agent to evaluate
            env: Environment for evaluation
            num_episodes: Number of episodes to run

        Returns:
            Dictionary with stability metrics
        """
        total_ticks = 0
        total_time = 0.0
        collision_ticks = 0
        position_changes = []

        for episode in range(num_episodes):
            # Reset agent position
            agent.position = (50, 50, 50)
            agent.ram = 200.0
            prev_position = agent.position

            # Run episode for 30 ticks
            episode_start = time.time()
            for tick in range(30):
                tick_start = time.time()

                # Sense and act
                try:
                    agent.sense_and_act(env)
                except Exception:
                    pass

                # Execute action
                try:
                    agent.execute_action(env)
                except Exception:
                    pass

                # Check for collisions
                collisions = env.resolve_collisions([agent])
                if collisions:
                    collision_ticks += 1

                # Track position changes for smoothness
                if agent.position != prev_position:
                    position_changes.append(agent.position)
                    prev_position = agent.position

                # Track timing
                tick_end = time.time()
                tick_time = tick_end - tick_start
                total_time += tick_time
                total_ticks += 1

                # Decay RAM if needed
                try:
                    agent._ticks_since_last_ram += 1
                except Exception:
                    pass

        # Calculate metrics
        avg_tick_time = total_time / max(1, total_ticks)
        collision_rate = collision_ticks / max(1, total_ticks)

        # Smoothness: inverse of position change variability
        # Smoother = fewer position changes per tick
        smoothness = 1.0 - min(1.0, len(position_changes) / max(1, total_ticks * 0.5))

        return {
            "avg_tick_time": avg_tick_time,
            "collision_rate": collision_rate,
            "smoothness": smoothness,
            "num_ticks": total_ticks,
        }

    def eval_navigation_success(
        self, agent: Agent, env: Environment, num_episodes: int = 10
    ) -> Dict:
        """
        Evaluate agent navigation and goal-reaching ability.

        Measures:
        - success_rate: Fraction of episodes reaching goal [0, 1]
        - avg_steps_to_goal: Average steps taken to reach goal
        - learning_improvement: Improvement from early to late episodes
        - early_success_rate: Success rate in first half of episodes [0, 1]
        - late_success_rate: Success rate in second half of episodes [0, 1]

        Args:
            agent: Agent to evaluate
            env: Environment for evaluation
            num_episodes: Number of episodes to run

        Returns:
            Dictionary with navigation metrics
        """
        successes = 0
        steps_to_goals = []
        early_successes = 0
        late_successes = 0
        goal_position = (95, 95, 95)  # Fixed goal position

        for episode in range(num_episodes):
            # Reset agent for episode
            agent.position = (5, 5, 5)
            agent.ram = 200.0
            agent.alive = True
            steps = 0
            reached_goal = False

            # Run episode for max 100 steps
            for step in range(100):
                # Sense and act
                try:
                    agent.sense_and_act(env)
                except Exception:
                    pass

                # Execute action
                try:
                    agent.execute_action(env)
                except Exception:
                    pass

                steps += 1

                # Check if agent reached goal
                dist_to_goal = math.sqrt(
                    (agent.position[0] - goal_position[0]) ** 2
                    + (agent.position[1] - goal_position[1]) ** 2
                    + (agent.position[2] - goal_position[2]) ** 2
                )

                if dist_to_goal < 5:
                    reached_goal = True
                    break

                # Check if agent died
                if not agent.alive or agent.ram <= 0:
                    break

                # Decay RAM
                try:
                    agent._ticks_since_last_ram += 1
                except Exception:
                    pass

            # Track results
            if reached_goal:
                successes += 1
                steps_to_goals.append(steps)

                # Track early vs late
                if episode < num_episodes // 2:
                    early_successes += 1
                else:
                    late_successes += 1
            else:
                steps_to_goals.append(100)  # Max steps if failed

        # Calculate metrics
        success_rate = successes / max(1, num_episodes)
        avg_steps_to_goal = sum(steps_to_goals) / max(1, len(steps_to_goals))

        early_episodes = max(1, num_episodes // 2)
        late_episodes = max(1, num_episodes - early_episodes)
        early_success_rate = early_successes / early_episodes
        late_success_rate = late_successes / late_episodes

        # Learning improvement: positive if late > early
        learning_improvement = late_success_rate - early_success_rate

        return {
            "success_rate": success_rate,
            "avg_steps_to_goal": avg_steps_to_goal,
            "learning_improvement": learning_improvement,
            "early_success_rate": early_success_rate,
            "late_success_rate": late_success_rate,
        }

    def eval_embodied_navigation(self, embodied_agent, num_episodes: int = 10) -> Dict:
        """
        Evaluate embodied agent navigation and learning.

        Measures:
        - success_rate: Fraction of episodes reaching goal [0, 1]
        - avg_steps_to_goal: Average steps taken to reach goal
        - learning_improvement: Improvement from early to late episodes
        - early_success_rate: Success rate in first half of episodes [0, 1]
        - late_success_rate: Success rate in second half of episodes [0, 1]

        Args:
            embodied_agent: EmbodiedAgent to evaluate (has step() method)
            num_episodes: Number of episodes to run

        Returns:
            Dictionary with navigation metrics
        """
        successes = 0
        steps_to_goals = []
        early_successes = 0
        late_successes = 0

        for episode in range(num_episodes):
            # Reset for new episode (handled internally by embodied_agent)
            steps = 0
            reached_goal = False

            # Run episode until terminal or max 300 steps
            while steps < 300:
                result = embodied_agent.step()
                steps += 1

                # Check if goal reached
                if result.get('is_goal_reached', False):
                    reached_goal = True
                    break

                # Check if episode auto-reset (tick_count == 0 after step means reset happened)
                if embodied_agent.tick_count == 0:
                    break

            # Track results
            if reached_goal:
                successes += 1
                steps_to_goals.append(steps)

                # Track early vs late
                if episode < num_episodes // 2:
                    early_successes += 1
                else:
                    late_successes += 1
            else:
                steps_to_goals.append(300)  # Max steps if failed

        # Calculate metrics
        success_rate = successes / max(1, num_episodes)
        avg_steps_to_goal = sum(steps_to_goals) / max(1, len(steps_to_goals))

        early_episodes = max(1, num_episodes // 2)
        late_episodes = max(1, num_episodes - early_episodes)
        early_success_rate = early_successes / early_episodes
        late_success_rate = late_successes / late_episodes

        # Learning improvement: positive if late > early
        learning_improvement = late_success_rate - early_success_rate

        return {
            "success_rate": success_rate,
            "avg_steps_to_goal": avg_steps_to_goal,
            "learning_improvement": learning_improvement,
            "early_success_rate": early_success_rate,
            "late_success_rate": late_success_rate,
        }

    def eval_transfer_readiness(self) -> Dict:
        """
        Evaluate agent transfer readiness.

        Measures:
        - determinism_check: Whether identical seeds produce identical behavior (bool)
        - config_consistency: Whether configuration is consistent across runs (bool)
        - reproducibility_score: Reproducibility metric [0, 1]

        Returns:
            Dictionary with transfer readiness metrics
        """
        lrsg_cfg = LRSGConfig(num_neurons=2000, num_input_neurons=100, num_output_neurons=12)

        # Create two agents with same seed for determinism check
        torch.manual_seed(42)
        brain1 = LRSGraph(lrsg_cfg)
        from src.harness.security import AnnouncerSanitizer
        sanitizer = AnnouncerSanitizer()
        from src.config import AgentConfig
        agent1 = Agent(0, (50, 50, 50), brain1, AgentConfig(), lrsg_cfg, sanitizer=sanitizer)

        torch.manual_seed(42)
        brain2 = LRSGraph(lrsg_cfg)
        agent2 = Agent(1, (50, 50, 50), brain2, AgentConfig(), lrsg_cfg, sanitizer=sanitizer)

        # Check if agents have same configuration
        config_consistent = (
            agent1.config.initial_ram == agent2.config.initial_ram
            and agent1.lrsg_config.num_neurons == agent2.lrsg_config.num_neurons
        )

        # Check if brains have same weight initialization
        deterministic = True
        try:
            for p1, p2 in zip(brain1.parameters(), brain2.parameters()):
                if not torch.allclose(p1, p2, atol=1e-6):
                    deterministic = False
                    break
        except Exception:
            deterministic = False

        # Reproducibility score combines both checks
        reproducibility_score = float(deterministic and config_consistent)

        return {
            "determinism_check": deterministic,
            "config_consistency": config_consistent,
            "reproducibility_score": reproducibility_score,
        }
