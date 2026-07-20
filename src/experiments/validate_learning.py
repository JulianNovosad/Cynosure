"""
Step 6: Learning validation — agent must improve on a simple navigation task over 1K episodes.

Runs 1000 episodes with ClusterSimulation + Hebbian learning, then compares
first-100 vs last-100 episode averages to verify measurable improvement.

Success criterion (from plans/rewrite.md Step 6):
  Average reward OR goal success rate must be higher in the last 100 episodes
  than in the first 100 episodes.
"""

import sys
import numpy as np

# Allow running from project root
sys.path.insert(0, ".")

from src.cluster.scheduler import build_default_scheduler
from src.simulation.cluster_loop import ClusterSimulation, EpisodeResult


N_EPISODES = 1000
WINDOW = 100         # compare first vs last N episodes
N_CLUSTERS = 6
N_NEURONS = 2000     # lower end of spec range → faster iteration
N_EDGES = 20000      # lower end of spec range
MAX_ACTIVE = 6


def run_learning_validation() -> list[EpisodeResult]:
    rng = np.random.default_rng(0)
    scheduler = build_default_scheduler(
        n_clusters=N_CLUSTERS,
        n_neurons=N_NEURONS,
        n_edges=N_EDGES,
        max_active=MAX_ACTIVE,
        rng=rng,
    )
    sim = ClusterSimulation(scheduler=scheduler, grid_size=10.0, n_obstacles=3, seed=1)

    print(f"Running {N_EPISODES} episodes "
          f"({N_CLUSTERS} clusters × {N_NEURONS} neurons × {N_EDGES} edges) ...")

    results = sim.run(N_EPISODES)

    # Report progress at 100-episode intervals
    for milestone in range(100, N_EPISODES + 1, 100):
        batch = results[milestone - 100 : milestone]
        avg_r = np.mean([r.total_reward for r in batch])
        sr = np.mean([r.reached_goal for r in batch])
        avg_steps = np.mean([r.steps for r in batch])
        print(f"  ep {milestone - 99:4d}–{milestone:4d}: "
              f"avg_reward={avg_r:+.4f}  success={sr:.2f}  avg_steps={avg_steps:.0f}")

    return results


def evaluate(results: list[EpisodeResult]) -> bool:
    first = results[:WINDOW]
    last = results[-WINDOW:]

    first_avg_reward = float(np.mean([r.total_reward for r in first]))
    last_avg_reward = float(np.mean([r.total_reward for r in last]))
    first_success = float(np.mean([r.reached_goal for r in first]))
    last_success = float(np.mean([r.reached_goal for r in last]))

    print()
    print("=" * 55)
    print("LEARNING VALIDATION RESULTS")
    print("=" * 55)
    print(f"  First {WINDOW} episodes — avg reward: {first_avg_reward:+.4f}  "
          f"success rate: {first_success:.2%}")
    print(f"  Last  {WINDOW} episodes — avg reward: {last_avg_reward:+.4f}  "
          f"success rate: {last_success:.2%}")
    print()

    reward_improved = last_avg_reward > first_avg_reward
    success_improved = last_success > first_success
    improved = reward_improved or success_improved

    if improved:
        if reward_improved:
            delta = last_avg_reward - first_avg_reward
            print(f"  [PASS] Avg reward improved by {delta:+.4f}")
        if success_improved:
            delta_sr = last_success - first_success
            print(f"  [PASS] Success rate improved by {delta_sr:+.2%}")
    else:
        print(f"  [FAIL] No measurable improvement detected.")
        print(f"         reward delta = {last_avg_reward - first_avg_reward:+.4f}")
        print(f"         success delta = {last_success - first_success:+.2%}")

    print("=" * 55)
    return improved


def benchmark_vram(n_clusters: int = 12, n_neurons: int = 4096, n_edges: int = 40000) -> None:
    """Step 7: benchmark VRAM footprint for max active cluster count."""
    rng = np.random.default_rng(42)
    scheduler = build_default_scheduler(
        n_clusters=n_clusters,
        n_neurons=n_neurons,
        n_edges=n_edges,
        max_active=n_clusters,
        rng=rng,
    )

    # Force all clusters active by running one tick
    sim = ClusterSimulation(scheduler=scheduler, grid_size=10.0, seed=42)
    sim.run_episode()

    footprint_bytes = scheduler.vram_footprint_bytes()
    footprint_mb = footprint_bytes / (1024 * 1024)
    n_active = scheduler.n_active
    spec_max_mb = 15.0

    print()
    print("=" * 55)
    print("VRAM FOOTPRINT BENCHMARK (Step 7)")
    print("=" * 55)
    print(f"  Config: {n_clusters} clusters × {n_neurons} neurons × {n_edges} edges")
    print(f"  Active clusters: {n_active}")
    print(f"  Cluster data footprint: {footprint_mb:.2f} MB")
    print(f"  Spec target: ≤{spec_max_mb:.0f} MB cluster data")
    if footprint_mb <= spec_max_mb:
        print(f"  [PASS] Within 15 MB spec target")
    else:
        print(f"  [FAIL] Exceeds spec target by {footprint_mb - spec_max_mb:.2f} MB")
    print("=" * 55)


if __name__ == "__main__":
    results = run_learning_validation()
    passed = evaluate(results)

    benchmark_vram(n_clusters=12, n_neurons=4096, n_edges=40000)

    sys.exit(0 if passed else 1)
