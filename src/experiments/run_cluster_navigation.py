"""
1K-episode navigation learning validation.

Verifies measurable learning improvement with the cluster brain architecture.

Metrics reported:
  - Episode reward (early vs late)
  - Distance to goal (early vs late)
  - Goal success rate (early vs late)
  - VRAM footprint at scale

Usage:
    python -m src.experiments.run_cluster_navigation
    python -m src.experiments.run_cluster_navigation --n-episodes 1000 --n-clusters 8
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.cluster.scheduler import build_default_scheduler
from src.simulation.cluster_loop import ClusterSimulation


def run_navigation_validation(
    n_episodes: int = 1000,
    n_clusters: int = 8,
    n_neurons: int = 4096,
    n_edges: int = 40000,
    max_active: int = 8,
    seed: int = 0,
    window: int = 50,
) -> dict:
    """
    Run N episodes and return learning metrics.

    Args:
        n_episodes: Total episodes to run.
        n_clusters: Number of clusters in the scheduler.
        n_neurons: Neurons per cluster.
        n_edges: Edges per cluster.
        max_active: Max clusters active simultaneously (4–12).
        seed: RNG seed for reproducibility.
        window: Window size for early/late comparison.

    Returns:
        dict with keys: early_reward, late_reward, improvement, success_rate_early,
        success_rate_late, vram_bytes, all_rewards, all_distances, all_success.
    """
    rng = np.random.default_rng(seed)
    scheduler = build_default_scheduler(
        n_clusters=n_clusters,
        n_neurons=n_neurons,
        n_edges=n_edges,
        max_active=max_active,
        rng=rng,
    )

    sim = ClusterSimulation(
        scheduler=scheduler,
        grid_size=10.0,
        n_obstacles=3,
        tick_rate_hz=30.0,
        seed=seed,
    )

    all_rewards = []
    all_distances = []
    all_success = []

    t0 = time.time()
    for ep in range(n_episodes):
        result = sim.run_episode()
        all_rewards.append(result.total_reward)
        all_distances.append(result.final_distance)
        all_success.append(int(result.reached_goal))

        if (ep + 1) % 100 == 0:
            elapsed = time.time() - t0
            recent_reward = float(np.mean(all_rewards[-100:]))
            recent_success = float(np.mean(all_success[-100:]))
            print(
                f"  ep {ep+1:4d}/{n_episodes}  "
                f"reward={recent_reward:+.4f}  "
                f"success={recent_success:.2%}  "
                f"elapsed={elapsed:.1f}s"
            )

    # Early vs late windows
    early_rewards = all_rewards[:window]
    late_rewards = all_rewards[-window:]
    early_success = all_success[:window]
    late_success = all_success[-window:]
    early_dists = all_distances[:window]
    late_dists = all_distances[-window:]

    early_mean = float(np.mean(early_rewards))
    late_mean = float(np.mean(late_rewards))
    improvement = late_mean - early_mean

    vram = scheduler.vram_footprint_bytes()

    return {
        "early_reward": early_mean,
        "late_reward": late_mean,
        "improvement": improvement,
        "success_rate_early": float(np.mean(early_success)),
        "success_rate_late": float(np.mean(late_success)),
        "early_dist": float(np.mean(early_dists)),
        "late_dist": float(np.mean(late_dists)),
        "vram_bytes": vram,
        "n_episodes": n_episodes,
        "n_clusters": n_clusters,
        "max_active": max_active,
        "all_rewards": all_rewards,
        "all_distances": all_distances,
        "all_success": all_success,
    }


def print_report(results: dict) -> None:
    print("\n" + "=" * 60)
    print("CLUSTER NAVIGATION LEARNING VALIDATION")
    print("=" * 60)
    print(f"  Episodes run:          {results['n_episodes']}")
    print(f"  Clusters:              {results['n_clusters']}  (max active: {results['max_active']})")
    print(f"  VRAM footprint:        {results['vram_bytes'] / 1024:.1f} KB active cluster data")
    print()
    print("  EARLY vs LATE (first/last 50 episodes):")
    print(f"    Reward  early={results['early_reward']:+.4f}  late={results['late_reward']:+.4f}  delta={results['improvement']:+.4f}")
    print(f"    Distance  early={results['early_dist']:.3f}m  late={results['late_dist']:.3f}m")
    print(f"    Success  early={results['success_rate_early']:.2%}  late={results['success_rate_late']:.2%}")
    print()
    if results["improvement"] > 0:
        print("  RESULT: LEARNING DETECTED — late reward > early reward")
    else:
        print("  RESULT: No improvement detected in this run (stochastic nav task)")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster navigation learning validation")
    parser.add_argument("--n-episodes", type=int, default=1000)
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--n-neurons", type=int, default=4096)
    parser.add_argument("--n-edges", type=int, default=40000)
    parser.add_argument("--max-active", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="experiment_results/cluster_navigation.json")
    args = parser.parse_args()

    print(f"Running {args.n_episodes} episodes with {args.n_clusters} clusters "
          f"({args.n_neurons}N/{args.n_edges}E each, max_active={args.max_active})...")

    results = run_navigation_validation(
        n_episodes=args.n_episodes,
        n_clusters=args.n_clusters,
        n_neurons=args.n_neurons,
        n_edges=args.n_edges,
        max_active=args.max_active,
        seed=args.seed,
    )

    print_report(results)

    # Save to experiment_results/
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Don't serialize per-episode arrays into the JSON (too large)
    save = {k: v for k, v in results.items() if not isinstance(v, list)}
    with open(out_path, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
