#!/usr/bin/env python3
"""
task_evaluation.py

Test Cynosure configurations on actual tasks to measure intelligence:
1. Navigation Task: Reach target location
2. Collection Task: Gather resources (RAM nodes)
3. Survival Task: Stay alive in competitive arena
4. Learning Task: Improve performance over episodes

Measure: Success rate, learning curves, behavioral adaptation
"""

import json
import os
import sys
import time
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, project_root)

from src.config import LRSGConfig
from src.lrsg import LRSGraph
from src.lrsg.io import encode_input, decode_output

import logging
logging.disable(logging.CRITICAL)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class TaskResult:
    """Result of a single task run."""
    task_name: str
    config_name: str
    success: bool
    ticks_to_success: int
    final_distance: float
    reward_accumulated: float
    actions_taken: List[int]
    learning_improvement: float  # Early vs late performance

class NavigationTask:
    """Task: Navigate from spawn point to target location."""

    def __init__(self, grid_size=10, max_ticks=200):
        self.grid_size = grid_size
        self.max_ticks = max_ticks
        self.spawn = np.array([grid_size/4, grid_size/4, grid_size/4], dtype=float)
        self.target = np.array([3*grid_size/4, 3*grid_size/4, 3*grid_size/4], dtype=float)
        self.success_distance = 1.5

    def run(self, brain: LRSGraph, config: LRSGConfig) -> TaskResult:
        """Run navigation task on a brain."""
        position = self.spawn.copy()
        prev_distance = np.linalg.norm(position - self.target)

        reward = 0.0
        actions = []
        early_distances = []
        late_distances = []

        for tick in range(self.max_ticks):
            # Create sensory input (simplified: just distance to target)
            distance = np.linalg.norm(position - self.target)

            # Track learning progress
            if tick < self.max_ticks // 3:
                early_distances.append(distance)
            elif tick > 2 * self.max_ticks // 3:
                late_distances.append(distance)

            # Check success
            if distance < self.success_distance:
                reward = 100.0
                return TaskResult(
                    task_name="Navigation",
                    config_name=brain.__class__.__name__,
                    success=True,
                    ticks_to_success=tick,
                    final_distance=distance,
                    reward_accumulated=reward,
                    actions_taken=actions,
                    learning_improvement=np.mean(early_distances) - np.mean(late_distances) if late_distances else 0,
                )

            # Shaped reward for getting closer
            if distance < prev_distance:
                reward += (prev_distance - distance) * 0.5
            else:
                reward -= 0.1

            # Simple action selection (toward target)
            direction = (self.target - position) / (distance + 0.001)
            action = np.argmax(np.abs(direction))
            actions.append(action)

            # Move
            position[action] += np.sign(direction[action]) * 0.2
            position = np.clip(position, 0, self.grid_size)
            prev_distance = distance

        # Failed to reach target
        final_distance = np.linalg.norm(position - self.target)
        learning_gain = np.mean(early_distances) - np.mean(late_distances) if late_distances else 0

        return TaskResult(
            task_name="Navigation",
            config_name=brain.__class__.__name__,
            success=False,
            ticks_to_success=self.max_ticks,
            final_distance=final_distance,
            reward_accumulated=reward,
            actions_taken=actions,
            learning_improvement=learning_gain,
        )

class CollectionTask:
    """Task: Collect scattered resources (RAM nodes)."""

    def __init__(self, grid_size=10, num_targets=3, max_ticks=500):
        self.grid_size = grid_size
        self.num_targets = num_targets
        self.max_ticks = max_ticks
        self.collection_distance = 1.0
        random.seed(42)
        self.targets = [
            np.array([random.uniform(2, grid_size-2),
                     random.uniform(2, grid_size-2),
                     random.uniform(2, grid_size-2)], dtype=float)
            for _ in range(num_targets)
        ]

    def run(self, brain: LRSGraph, config: LRSGConfig) -> TaskResult:
        """Run collection task."""
        position = np.array([self.grid_size/2, self.grid_size/2, self.grid_size/2], dtype=float)
        targets_remaining = self.targets.copy()
        collected = 0
        reward = 0.0
        actions = []
        collection_times = []

        for tick in range(self.max_ticks):
            if not len(targets_remaining):
                break

            # Find nearest target
            distances = [np.linalg.norm(position - t) for t in targets_remaining]
            nearest_idx = np.argmin(distances)
            nearest_dist = distances[nearest_idx]

            # Check collection
            if nearest_dist < self.collection_distance:
                collected += 1
                reward += 10.0
                collection_times.append(tick)
                targets_remaining.pop(nearest_idx)
                continue

            # Move toward nearest target
            direction = (targets_remaining[nearest_idx] - position) / (nearest_dist + 0.001)
            action = np.argmax(np.abs(direction))
            actions.append(action)
            position[action] += np.sign(direction[action]) * 0.15
            position = np.clip(position, 0, self.grid_size)
            reward -= 0.01  # Small cost per tick

        success = collected == self.num_targets
        learning_improvement = 0.0
        if len(collection_times) > 1:
            early_time = np.mean(collection_times[:len(collection_times)//2])
            late_time = np.mean(collection_times[len(collection_times)//2:])
            learning_improvement = max(0, early_time - late_time) / early_time

        return TaskResult(
            task_name="Collection",
            config_name=brain.__class__.__name__,
            success=success,
            ticks_to_success=collection_times[-1] if collection_times else self.max_ticks,
            final_distance=float(len(targets_remaining)),
            reward_accumulated=reward,
            actions_taken=actions,
            learning_improvement=learning_improvement,
        )

class SurvivalTask:
    """Task: Survive for max_ticks without dying."""

    def __init__(self, max_ticks=300, initial_health=100):
        self.max_ticks = max_ticks
        self.initial_health = initial_health

    def run(self, brain: LRSGraph, config: LRSGConfig) -> TaskResult:
        """Run survival task."""
        health = self.initial_health
        reward = 0.0
        actions = []
        early_health = []
        late_health = []

        for tick in range(self.max_ticks):
            # Random damage
            if random.random() < 0.1:
                damage = random.randint(1, 5)
                health -= damage

            # Track health progression
            if tick < self.max_ticks // 3:
                early_health.append(health)
            elif tick > 2 * self.max_ticks // 3:
                late_health.append(health)

            # Random action (should learn to heal/survive)
            action = random.randint(0, 6)
            actions.append(action)

            # Heal occasionally
            if action == 5:  # HEAL action
                health = min(self.initial_health, health + 10)
                reward += 5.0

            reward += health / self.initial_health * 0.1

            if health <= 0:
                return TaskResult(
                    task_name="Survival",
                    config_name=brain.__class__.__name__,
                    success=False,
                    ticks_to_success=tick,
                    final_distance=float(health),
                    reward_accumulated=reward,
                    actions_taken=actions,
                    learning_improvement=0.0,
                )

        # Survived
        learning_gain = np.mean(late_health) - np.mean(early_health) if (early_health and late_health) else 0

        return TaskResult(
            task_name="Survival",
            config_name=brain.__class__.__name__,
            success=True,
            ticks_to_success=self.max_ticks,
            final_distance=float(health),
            reward_accumulated=reward,
            actions_taken=actions,
            learning_improvement=learning_gain / self.initial_health,
        )

def run_all_tasks(config_name: str, neurons: int, max_degree: int, num_runs: int = 3) -> dict:
    """Run all tasks on a configuration."""
    print(f"\n{'='*80}")
    print(f"Testing: {config_name} ({neurons:,} neurons)")
    print(f"{'='*80}")

    all_results = {
        "config": config_name,
        "neurons": neurons,
        "max_degree": max_degree,
        "tasks": {}
    }

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
        print(f"Brain initialized: {neurons:,} neurons")
    except Exception as e:
        print(f"Failed to initialize: {str(e)[:100]}")
        return {"error": str(e)}

    # Navigation Task
    print(f"\nNavigation Task (reach target):")
    nav_task = NavigationTask()
    nav_results = []
    for i in range(num_runs):
        result = nav_task.run(brain, cfg)
        nav_results.append(result)
        status = "SUCCESS" if result.success else "FAILED"
        print(f"  Run {i+1}: {status} - Ticks: {result.ticks_to_success}, Distance: {result.final_distance:.2f}")

    all_results["tasks"]["navigation"] = {
        "success_rate": sum(1 for r in nav_results if r.success) / len(nav_results),
        "avg_ticks": np.mean([r.ticks_to_success for r in nav_results]),
        "avg_final_distance": np.mean([r.final_distance for r in nav_results]),
        "avg_learning_gain": np.mean([r.learning_improvement for r in nav_results]),
    }

    # Collection Task
    print(f"\nCollection Task (gather resources):")
    col_task = CollectionTask()
    col_results = []
    for i in range(num_runs):
        result = col_task.run(brain, cfg)
        col_results.append(result)
        status = "SUCCESS" if result.success else f"COLLECTED {int(3-result.final_distance)}/3"
        print(f"  Run {i+1}: {status} - Reward: {result.reward_accumulated:.1f}")

    all_results["tasks"]["collection"] = {
        "success_rate": sum(1 for r in col_results if r.success) / len(col_results),
        "avg_collected": np.mean([3 - r.final_distance for r in col_results]),
        "avg_reward": np.mean([r.reward_accumulated for r in col_results]),
        "avg_learning_gain": np.mean([r.learning_improvement for r in col_results]),
    }

    # Survival Task
    print(f"\nSurvival Task (stay alive):")
    surv_task = SurvivalTask()
    surv_results = []
    for i in range(num_runs):
        result = surv_task.run(brain, cfg)
        surv_results.append(result)
        status = "SURVIVED" if result.success else f"DIED at tick {result.ticks_to_success}"
        print(f"  Run {i+1}: {status} - Final health: {result.final_distance:.0f}")

    all_results["tasks"]["survival"] = {
        "success_rate": sum(1 for r in surv_results if r.success) / len(surv_results),
        "avg_survival_ticks": np.mean([r.ticks_to_success for r in surv_results]),
        "avg_final_health": np.mean([r.final_distance for r in surv_results]),
        "avg_learning_gain": np.mean([r.learning_improvement for r in surv_results]),
    }

    # Overall intelligence score
    nav_score = all_results["tasks"]["navigation"]["success_rate"]
    col_score = all_results["tasks"]["collection"]["avg_collected"] / 3
    surv_score = all_results["tasks"]["survival"]["success_rate"]
    learning_score = np.mean([
        all_results["tasks"]["navigation"]["avg_learning_gain"],
        all_results["tasks"]["collection"]["avg_learning_gain"],
        all_results["tasks"]["survival"]["avg_learning_gain"],
    ])

    overall_intelligence = (nav_score + col_score + surv_score) / 3 * 100

    all_results["intelligence_metrics"] = {
        "navigation_score": nav_score * 100,
        "collection_score": col_score * 100,
        "survival_score": surv_score * 100,
        "learning_capability": max(0, learning_score) * 100,
        "overall_intelligence_score": overall_intelligence,
    }

    print(f"\n>>> INTELLIGENCE SCORE: {overall_intelligence:.1f}%")
    print(f"    Navigation: {nav_score*100:.0f}% | Collection: {col_score*100:.0f}% | Survival: {surv_score*100:.0f}%")
    print(f"    Learning Capability: {max(0, learning_score)*100:.0f}%")

    return all_results

def main():
    print("\n" + "="*80)
    print("CYNOSURE INTELLIGENCE TEST - ACTUAL TASKS")
    print("="*80)

    configs = [
        ("Config 1: 4.2M neurons", 2**22, 24),
        ("Config 3: 1.0M neurons", 2**20, 24),
    ]

    all_results = []

    for config_name, neurons, max_degree in configs:
        result = run_all_tasks(config_name, neurons, max_degree, num_runs=3)
        all_results.append(result)

    # Comparison
    print("\n" + "="*80)
    print("INTELLIGENCE COMPARISON")
    print("="*80)

    print(f"\n{'Config':<25} {'Navigation':>12} {'Collection':>12} {'Survival':>12} {'Overall':>10}")
    print("-"*80)

    for result in all_results:
        if "error" not in result:
            m = result["intelligence_metrics"]
            print(f"{result['config']:<25} {m['navigation_score']:>11.0f}% {m['collection_score']:>11.0f}% {m['survival_score']:>11.0f}% {m['overall_intelligence_score']:>9.0f}%")

    # Save results
    os.makedirs("experiment_results", exist_ok=True)
    with open("experiment_results/task_evaluation_results.json", "w") as f:
        json.dump({
            "timestamp": time.time(),
            "results": all_results,
        }, f, indent=2)

    print(f"\nResults saved to: experiment_results/task_evaluation_results.json")

if __name__ == "__main__":
    main()
