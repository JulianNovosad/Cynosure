#!/usr/bin/env python3
"""
Curriculum runner for embodied navigation learning.

Starts training with an obstacle/enemy-free arena (proven to produce real
goal-reaches and Hebbian weight updates) and ramps difficulty up in stages,
keeping the same EmbodiedAgent (and its learned LRSG weights) across stages
by swapping in a harder BattlegroundEnvironment rather than recreating the agent.

Usage:
    python src/experiments/run_embodied_curriculum.py
"""
import sys
from typing import Dict, List

from src.config import LRSGConfig, LocomotionConfig, EmbodiedConfig
from src.simulation.embodied_loop import EmbodiedAgent
from src.simulation.battleground_env import BattlegroundEnvironment

DEFAULT_STAGES: List[Dict[str, int]] = [
    {"num_obstacles": 0, "num_enemies": 0},
    {"num_obstacles": 1, "num_enemies": 0},
    {"num_obstacles": 2, "num_enemies": 0},
    {"num_obstacles": 3, "num_enemies": 0},
    {"num_obstacles": 3, "num_enemies": 1},
    {"num_obstacles": 3, "num_enemies": 3},
]


def _run_stage(agent: EmbodiedAgent, num_obstacles: int, num_enemies: int, num_episodes: int) -> float:
    """Swap in a new environment at the given difficulty, run episodes, return success rate."""
    agent.emb_config.num_obstacles = num_obstacles
    agent.emb_config.num_enemies = num_enemies
    agent.env = BattlegroundEnvironment(
        grid_size=agent.emb_config.grid_size,
        num_obstacles=num_obstacles,
        num_enemies=num_enemies,
        seed=None,
    )
    agent._reset_episode()

    successes = 0
    max_ticks = agent.emb_config.episode_length + agent.emb_config.collision_grace_ticks + 5
    for _ in range(num_episodes):
        result = None
        for _ in range(max_ticks):
            result = agent.step()
            if agent.tick_count == 0:
                break
        if result is not None and result["is_goal_reached"]:
            successes += 1
    return successes / num_episodes


def run_curriculum(
    stages: List[Dict[str, int]] = None,
    episodes_per_stage: int = 200,
    num_neurons: int = 5000,
    goal_reach_threshold: float = 0.35,
    device: str = "cpu",
    advance_threshold: float = 0.05,
) -> List[Dict]:
    """
    Train through a difficulty curriculum, advancing a stage only once the
    current stage's success rate clears advance_threshold.

    Returns list of {stage, success_rate, advanced} dicts, one per stage attempted.
    """
    stages = stages or DEFAULT_STAGES
    lrsg_cfg = LRSGConfig(num_neurons=num_neurons)
    loc_cfg = LocomotionConfig()
    emb_cfg = EmbodiedConfig(
        num_obstacles=stages[0]["num_obstacles"],
        num_enemies=stages[0]["num_enemies"],
        goal_reach_threshold=goal_reach_threshold,
    )
    agent = EmbodiedAgent(lrsg_cfg, loc_cfg, emb_cfg, device=device)

    log = []
    for stage in stages:
        rate = _run_stage(agent, stage["num_obstacles"], stage["num_enemies"], episodes_per_stage)
        advanced = rate >= advance_threshold
        log.append({"stage": stage, "success_rate": rate, "advanced": advanced})
        print(f"stage={stage} success_rate={rate:.3f} advanced={advanced}")
        if not advanced:
            print(f"Stopping curriculum: stage {stage} did not clear advance_threshold={advance_threshold}")
            break
    return log


if __name__ == "__main__":
    run_curriculum()
