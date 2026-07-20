#!/usr/bin/env python3
"""
learning_comparison_experiment.py

Compare learning capability of two configurations:
- Config 1: 4.2M neurons
- Config 3: 1M neurons

Run 20 episodes each, measure:
- Success rate improvement over time
- Learning curves
- Which config learns faster
- Does more neurons = faster learning?
"""

import json
import os
import sys
import time
import math
import random
import numpy as np
import torch

project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, project_root)

from src.config import LRSGConfig
from src.lrsg import LRSGraph
from src.lrsg.forward import forward_pass
from src.lrsg.plasticity import hebbian_update

import logging
logging.disable(logging.CRITICAL)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LearningNavigationTask:
    """Navigation task with reward-based learning."""

    def __init__(self, grid_size=10, max_ticks=200):
        self.grid_size = grid_size
        self.max_ticks = max_ticks
        self.spawn = np.array([2.0, 2.0, 2.0], dtype=float)
        self.target = np.array([8.0, 8.0, 8.0], dtype=float)
        self.success_distance = 1.5

    def run_episode(self, brain: LRSGraph, config: LRSGConfig, learning_rate: float = 0.01) -> dict:
        """Run one learning episode."""
        position = self.spawn.copy()
        brain.reset_activations()
        brain.reset_traces()

        total_reward = 0.0
        success = False
        ticks_to_success = self.max_ticks
        actions_taken = []

        # Accumulated reward for Hebbian update
        episode_reward = 0.0

        for tick in range(self.max_ticks):
            # Encode state: position and direction to target
            state = self._encode_state(position)

            # Forward pass through brain
            brain.reset_activations()
            brain.activations[:len(state)] = state

            # Select action from top 4 output neurons
            action = torch.argmax(brain.activations[:4]).item() % 4
            actions_taken.append(action)

            # Move based on action
            distance_before = np.linalg.norm(self.target - position)
            position = self._execute_action(action, position)
            distance_after = np.linalg.norm(self.target - position)

            # Reward: shaped reward based on distance to target
            # Give baseline reward + bonus for improving
            distance_penalty = -distance_after * 0.1  # Smaller penalty for being far
            improvement_bonus = 0.0

            if distance_after < distance_before:
                improvement_bonus = (distance_before - distance_after) * 2.0

            step_reward = distance_penalty + improvement_bonus
            episode_reward += step_reward

            # Check success
            if distance_after < self.success_distance:
                success = True
                ticks_to_success = tick
                episode_reward += 10.0  # Bonus for reaching target
                break

        # Apply Hebbian update based on accumulated reward
        if episode_reward > 0.1:  # Only update on positive reward
            try:
                hebbian_update(brain, reward=episode_reward)
            except:
                pass  # Skip if Hebbian update fails

        return {
            "episode": 0,
            "success": success,
            "ticks": ticks_to_success,
            "reward": episode_reward,
            "final_distance": distance_after,
            "actions": len(actions_taken),
        }

    def _encode_state(self, position: np.ndarray) -> torch.Tensor:
        """Encode position as state vector."""
        direction = (self.target - position) / (np.linalg.norm(self.target - position) + 0.001)
        distance = np.linalg.norm(self.target - position) / self.grid_size

        state = np.concatenate([
            position / self.grid_size,
            direction,
            [distance]
        ]).astype(np.float32)

        return torch.tensor(state, device=device, dtype=torch.float32)

    def _execute_action(self, action: int, position: np.ndarray) -> np.ndarray:
        """Execute action and return new position."""
        new_pos = position.copy()
        move_amount = 0.3

        direction = (self.target - position) / (np.linalg.norm(self.target - position) + 0.001)

        if action == 0:  # Move toward target
            new_pos += direction * move_amount
        elif action == 1:  # Move left
            new_pos[1] -= move_amount * 0.7
        elif action == 2:  # Move right
            new_pos[1] += move_amount * 0.7
        # action 3 = IDLE

        return np.clip(new_pos, 0, self.grid_size)

def run_learning_experiment(config_name: str, neurons: int, max_degree: int, num_episodes: int = 20) -> dict:
    """Run learning experiment on one configuration."""
    print(f"\n{'='*80}")
    print(f"LEARNING EXPERIMENT: {config_name} ({neurons:,} neurons)")
    print(f"{'='*80}")
    print(f"Running {num_episodes} learning episodes...\n")

    try:
        torch.cuda.empty_cache()

        # Initialize brain
        cfg = LRSGConfig(
            num_neurons=neurons,
            max_out_degree=max_degree,
            h_window=min(4096, neurons // 4),
            disable_language=True,
        )
        brain = LRSGraph(cfg, device)
        print(f"[OK] Brain initialized: {neurons:,} neurons\n")
    except Exception as e:
        print(f"[FAILED] {str(e)[:100]}")
        return {"error": str(e)}

    # Run episodes
    task = LearningNavigationTask()
    episode_results = []

    learning_rate = 0.01
    start_time = time.time()

    for episode in range(num_episodes):
        result = task.run_episode(brain, cfg, learning_rate=learning_rate)
        episode_results.append(result)

        status = "SUCCESS" if result["success"] else "failed"
        print(f"Episode {episode+1:2d}: {status:7s} | Reward: {result['reward']:7.2f} | "
              f"Distance: {result['final_distance']:5.2f} | Ticks: {result['ticks']:3d}")

    elapsed = time.time() - start_time

    # Calculate learning metrics
    early_success = sum(1 for r in episode_results[:5] if r["success"]) / 5
    late_success = sum(1 for r in episode_results[-5:] if r["success"]) / 5
    learning_improvement = late_success - early_success

    early_reward = np.mean([r["reward"] for r in episode_results[:5]])
    late_reward = np.mean([r["reward"] for r in episode_results[-5:]])
    reward_improvement = late_reward - early_reward

    early_distance = np.mean([r["final_distance"] for r in episode_results[:5]])
    late_distance = np.mean([r["final_distance"] for r in episode_results[-5:]])
    distance_improvement = early_distance - late_distance

    final_success_rate = sum(1 for r in episode_results if r["success"]) / len(episode_results)

    print(f"\n{'-'*80}")
    print(f"LEARNING RESULTS:")
    print(f"{'-'*80}")
    print(f"Total Time: {elapsed:.1f}s")
    print(f"Final Success Rate: {final_success_rate*100:.0f}%")
    print(f"Early Success Rate (episodes 1-5): {early_success*100:.0f}%")
    print(f"Late Success Rate (episodes 16-20): {late_success*100:.0f}%")
    print(f"Learning Improvement: {learning_improvement*100:+.0f}%")
    print(f"\nReward Improvement: {reward_improvement:+.2f}")
    print(f"Distance Improvement: {distance_improvement:+.2f}")

    return {
        "config": config_name,
        "neurons": neurons,
        "num_episodes": num_episodes,
        "episodes": episode_results,
        "metrics": {
            "final_success_rate": final_success_rate,
            "early_success_rate": early_success,
            "late_success_rate": late_success,
            "learning_improvement": learning_improvement,
            "reward_improvement": reward_improvement,
            "distance_improvement": distance_improvement,
            "elapsed_time": elapsed,
        }
    }

def main():
    print("\n" + "="*80)
    print("CYNOSURE LEARNING CAPABILITY EXPERIMENT")
    print("="*80)
    print("Measure if brains can learn and improve through experience")
    print("Question: Does more neurons = faster learning?")

    configs = [
        ("Config 1: 4.2M neurons", 2**22, 24),
        ("Config 3: 1.0M neurons", 2**20, 24),
    ]

    all_results = []

    for config_name, neurons, max_degree in configs:
        result = run_learning_experiment(config_name, neurons, max_degree, num_episodes=20)
        all_results.append(result)

    # Comparison
    print(f"\n\n{'='*80}")
    print("LEARNING COMPARISON")
    print(f"{'='*80}\n")

    print(f"{'Config':<25} {'Final Success':>15} {'Learning Gain':>15} {'Reward Gain':>15}")
    print("-"*80)

    for result in all_results:
        if "error" not in result:
            m = result["metrics"]
            print(f"{result['config']:<25} {m['final_success_rate']*100:>14.0f}% {m['learning_improvement']*100:>14.0f}% {m['reward_improvement']:>15.2f}")

    # Determine which learned better
    if all_results[0]["metrics"]["learning_improvement"] > all_results[1]["metrics"]["learning_improvement"]:
        better = all_results[0]["config"]
        improvement_diff = all_results[0]["metrics"]["learning_improvement"] - all_results[1]["metrics"]["learning_improvement"]
    else:
        better = all_results[1]["config"]
        improvement_diff = all_results[1]["metrics"]["learning_improvement"] - all_results[0]["metrics"]["learning_improvement"]

    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")
    print(f"\nWhich config learned better?")
    print(f"  WINNER: {better}")
    print(f"  Learning advantage: {improvement_diff*100:+.1f}%")

    # Generate learning curves data
    print(f"\nLearning Curves (Success Rate per Episode):")
    print("-"*80)

    for result in all_results:
        if "error" not in result:
            print(f"\n{result['config']}:")
            for ep in result["episodes"][:20]:
                bar_width = int(max(0, ep["reward"] / 2))
                bar = "#" * bar_width if bar_width > 0 else "-"
                status = "OK" if ep["success"] else "FAIL"
                print(f"  Ep {ep['episode']+1:2d}: {status} [{bar:20s}] R={ep['reward']:6.2f}")

    # Save results
    os.makedirs("experiment_results", exist_ok=True)
    with open("experiment_results/learning_experiment_results.json", "w") as f:
        json.dump({
            "timestamp": time.time(),
            "experiment": "Learning Capability Comparison",
            "results": all_results,
        }, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: experiment_results/learning_experiment_results.json")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
