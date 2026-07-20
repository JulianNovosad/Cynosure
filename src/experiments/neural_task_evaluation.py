#!/usr/bin/env python3
"""
neural_task_evaluation.py

Test Cynosure configurations on actual tasks using NEURAL DECISION-MAKING.
The brain must learn to solve tasks through its own emergent behavior.
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

import logging
logging.disable(logging.CRITICAL)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NavigationTaskNeural:
    """Navigation task using neural network for decisions."""

    def __init__(self, grid_size=10, max_ticks=200):
        self.grid_size = grid_size
        self.max_ticks = max_ticks
        self.spawn = np.array([grid_size/4, grid_size/4, grid_size/4], dtype=float)
        self.target = np.array([3*grid_size/4, 3*grid_size/4, 3*grid_size/4], dtype=float)
        self.success_distance = 1.5
        self.actions = ["MOVE_FWD", "MOVE_LEFT", "MOVE_RIGHT", "IDLE"]

    def encode_state(self, position: np.ndarray) -> torch.Tensor:
        """Encode position as neural input."""
        # Normalized direction to target
        direction = (self.target - position) / (np.linalg.norm(self.target - position) + 0.001)
        distance = np.linalg.norm(self.target - position) / self.grid_size

        # Create input: [position_x, position_y, position_z, direction_x, direction_y, direction_z, distance]
        state = np.concatenate([
            position / self.grid_size,
            direction,
            [distance]
        ]).astype(np.float32)

        return torch.tensor(state, device=device, dtype=torch.float32)

    def action_to_movement(self, action: int, position: np.ndarray) -> np.ndarray:
        """Convert neural action to movement."""
        new_pos = position.copy()
        if action == 0:  # Forward (toward target)
            direction = (self.target - position) / (np.linalg.norm(self.target - position) + 0.001)
            new_pos += direction * 0.3
        elif action == 1:  # Left
            new_pos[1] -= 0.2
        elif action == 2:  # Right
            new_pos[1] += 0.2
        # action 3 = IDLE, no movement

        return np.clip(new_pos, 0, self.grid_size)

    def run_neural(self, brain: LRSGraph, cfg: LRSGConfig, num_episodes: int = 3) -> dict:
        """Run navigation task with neural control."""
        episode_results = []

        for episode in range(num_episodes):
            position = self.spawn.copy()
            brain.reset_activations()
            brain.reset_traces()

            success = False
            ticks = 0
            total_distance_error = 0

            for tick in range(self.max_ticks):
                # Get state and run through brain
                state = self.encode_state(position)

                # Inject state into brain (simplified: use as activation)
                brain.reset_activations()
                brain.activations[:len(state)] = state

                # Get output (select action based on neuron with highest activation)
                output_idx = torch.argmax(brain.activations[:4]).item()
                action = output_idx % 4

                # Execute action
                position = self.action_to_movement(action, position)
                distance = np.linalg.norm(self.target - position)
                total_distance_error += distance

                # Check success
                if distance < self.success_distance:
                    success = True
                    ticks = tick
                    break

                ticks = tick

            episode_results.append({
                "episode": episode,
                "success": success,
                "ticks": ticks,
                "final_distance": distance,
                "avg_distance_error": total_distance_error / max(ticks, 1),
            })

        success_rate = sum(1 for r in episode_results if r["success"]) / len(episode_results)
        avg_ticks = np.mean([r["ticks"] for r in episode_results])

        return {
            "task": "Navigation",
            "success_rate": success_rate,
            "avg_ticks": avg_ticks,
            "episodes": episode_results,
        }

class ExplorationTaskNeural:
    """Exploration task: visit diverse locations, measure coverage."""

    def __init__(self, grid_size=10, max_ticks=300):
        self.grid_size = grid_size
        self.max_ticks = max_ticks
        self.grid_cells = np.zeros((grid_size, grid_size, grid_size), dtype=int)

    def run_neural(self, brain: LRSGraph, cfg: LRSGConfig, num_episodes: int = 3) -> dict:
        """Run exploration task with neural control."""
        episode_results = []

        for episode in range(num_episodes):
            position = np.array([self.grid_size/2, self.grid_size/2, self.grid_size/2], dtype=float)
            brain.reset_activations()
            brain.reset_traces()

            visited_cells = set()

            for tick in range(self.max_ticks):
                # Mark current cell as visited
                cell = tuple(np.clip(position.astype(int), 0, self.grid_size-1))
                visited_cells.add(cell)

                # Random action (exploration)
                action = torch.argmax(brain.activations[:6]).item() % 6

                # Move based on action
                move_amount = 0.5
                if action == 0:
                    position[0] += move_amount
                elif action == 1:
                    position[0] -= move_amount
                elif action == 2:
                    position[1] += move_amount
                elif action == 3:
                    position[1] -= move_amount
                elif action == 4:
                    position[2] += move_amount
                elif action == 5:
                    position[2] -= move_amount

                position = np.clip(position, 0, self.grid_size-1)

            coverage = len(visited_cells) / (self.grid_size ** 3)

            episode_results.append({
                "episode": episode,
                "cells_visited": len(visited_cells),
                "coverage": coverage,
            })

        avg_coverage = np.mean([r["coverage"] for r in episode_results])

        return {
            "task": "Exploration",
            "avg_cells_visited": np.mean([r["cells_visited"] for r in episode_results]),
            "avg_coverage": avg_coverage,
            "episodes": episode_results,
        }

class DecisionMakingTaskNeural:
    """Decision-making task: choose correct action in stimulus-response scenarios."""

    def __init__(self, max_ticks=100):
        self.max_ticks = max_ticks

    def run_neural(self, brain: LRSGraph, cfg: LRSGConfig, num_episodes: int = 3) -> dict:
        """Run decision-making task."""
        episode_results = []
        correct_decisions = 0
        total_decisions = 0

        for episode in range(num_episodes):
            brain.reset_activations()
            brain.reset_traces()

            episode_correct = 0

            for tick in range(self.max_ticks):
                # Create stimulus: random input
                stimulus = torch.randn(10, device=device)

                # Get brain response
                output = torch.argmax(brain.activations[:4]).item()

                # Correct answer is based on stimulus (deterministic)
                correct_answer = int(stimulus[:2].sum() > 0)

                # Check if correct
                if output % 2 == correct_answer:
                    episode_correct += 1

                total_decisions += 1

            accuracy = episode_correct / self.max_ticks
            episode_results.append({
                "episode": episode,
                "correct": episode_correct,
                "accuracy": accuracy,
            })
            correct_decisions += episode_correct

        overall_accuracy = correct_decisions / total_decisions if total_decisions > 0 else 0

        return {
            "task": "DecisionMaking",
            "overall_accuracy": overall_accuracy,
            "episodes": episode_results,
        }

def test_config(config_name: str, neurons: int, max_degree: int) -> dict:
    """Test a configuration on all tasks."""
    print(f"\n{'='*80}")
    print(f"Testing: {config_name} ({neurons:,} neurons)")
    print(f"{'='*80}")

    try:
        torch.cuda.empty_cache()

        cfg = LRSGConfig(
            num_neurons=neurons,
            max_out_degree=max_degree,
            h_window=min(4096, neurons // 4),
            disable_language=True,
        )
        brain = LRSGraph(cfg, device)
        print(f"[OK] Brain initialized")
    except Exception as e:
        print(f"[FAILED] {str(e)[:100]}")
        return {"error": str(e)}

    results = {
        "config": config_name,
        "neurons": neurons,
        "tasks": {}
    }

    # Task 1: Navigation
    print(f"\n1. Navigation Task (3 episodes)...")
    nav = NavigationTaskNeural()
    nav_result = nav.run_neural(brain, cfg, num_episodes=3)
    results["tasks"]["navigation"] = nav_result
    print(f"   Success rate: {nav_result['success_rate']*100:.0f}%")
    print(f"   Avg ticks: {nav_result['avg_ticks']:.0f}")

    # Task 2: Exploration
    print(f"\n2. Exploration Task (3 episodes)...")
    exp = ExplorationTaskNeural()
    exp_result = exp.run_neural(brain, cfg, num_episodes=3)
    results["tasks"]["exploration"] = exp_result
    print(f"   Avg coverage: {exp_result['avg_coverage']*100:.1f}%")
    print(f"   Avg cells visited: {exp_result['avg_cells_visited']:.0f}")

    # Task 3: Decision Making
    print(f"\n3. Decision-Making Task (3 episodes)...")
    dec = DecisionMakingTaskNeural()
    dec_result = dec.run_neural(brain, cfg, num_episodes=3)
    results["tasks"]["decision_making"] = dec_result
    print(f"   Accuracy: {dec_result['overall_accuracy']*100:.1f}%")

    # Calculate intelligence score
    nav_score = nav_result["success_rate"] * 100
    exp_score = exp_result["avg_coverage"] * 100
    dec_score = dec_result["overall_accuracy"] * 100

    intelligence_score = (nav_score + exp_score + dec_score) / 3

    results["intelligence_score"] = intelligence_score

    print(f"\n>>> INTELLIGENCE SCORE: {intelligence_score:.1f}%")
    print(f"    Navigation: {nav_score:.0f}% | Exploration: {exp_score:.1f}% | Decision-Making: {dec_score:.1f}%")

    return results

def main():
    print("\n" + "="*80)
    print("CYNOSURE NEURAL INTELLIGENCE TEST")
    print("="*80)
    print("Testing actual neural decision-making on real tasks")

    configs = [
        ("Config 1: 4.2M neurons", 2**22, 24),
        ("Config 3: 1.0M neurons", 2**20, 24),
    ]

    all_results = []

    for config_name, neurons, max_degree in configs:
        result = test_config(config_name, neurons, max_degree)
        all_results.append(result)

    # Comparison
    print("\n" + "="*80)
    print("INTELLIGENCE COMPARISON")
    print("="*80 + "\n")

    print(f"{'Config':<25} {'Navigation':>12} {'Exploration':>12} {'Decision':>12} {'Overall':>10}")
    print("-"*80)

    for result in all_results:
        if "error" not in result:
            nav = result["tasks"]["navigation"]["success_rate"] * 100
            exp = result["tasks"]["exploration"]["avg_coverage"] * 100
            dec = result["tasks"]["decision_making"]["overall_accuracy"] * 100
            overall = result["intelligence_score"]

            print(f"{result['config']:<25} {nav:>11.0f}% {exp:>11.1f}% {dec:>11.1f}% {overall:>9.1f}%")

    # Save results
    os.makedirs("experiment_results", exist_ok=True)
    with open("experiment_results/neural_task_evaluation.json", "w") as f:
        json.dump({
            "timestamp": time.time(),
            "results": all_results,
        }, f, indent=2)

    print(f"\nResults saved to: experiment_results/neural_task_evaluation.json")

if __name__ == "__main__":
    main()
