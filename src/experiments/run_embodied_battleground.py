#!/usr/bin/env python3
"""
Episode runner for embodied learning battleground.

Orchestrates 100+ episodes with live visualization showing agent learning
from near-zero success to high competence.

Usage:
    python src/experiments/run_embodied_battleground.py --episodes 100 --neurons 4048576
    python src/experiments/run_embodied_battleground.py --episodes 5 --neurons 100000 --device cpu --headless
"""
import argparse
import json
import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict

from src.config import LRSGConfig, LocomotionConfig, EmbodiedConfig
from src.simulation.embodied_loop import EmbodiedAgent
from src.simulation.embodied_visualizer import EmbodiedVisualizer


def run_embodied_learning(
    num_episodes: int = 100,
    num_neurons: int = 4_048_576,
    device: str = None,
    headless: bool = False,
):
    """
    Run embodied learning episodes with live visualization and learning loop.

    Args:
        num_episodes: Number of episodes to run (default 100)
        num_neurons: Number of neurons in LRSG (default 4M)
        device: 'cpu' or 'cuda' (default None = auto-detect)
        headless: If False, start Flask visualization (default False)

    Returns:
        List of episode data dictionaries with telemetry
    """
    # Auto-detect device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create configs
    lrsg_config = LRSGConfig(num_neurons=num_neurons)
    loc_config = LocomotionConfig()
    emb_config = EmbodiedConfig(
        grid_size=10,
        num_obstacles=3,
        episode_length=300,
        physics_freq=240,
        lidar_rays=8,
        goal_reach_threshold=0.3
    )

    # Create agent
    print(f"Initializing EmbodiedAgent with {num_neurons:,} neurons...")
    agent = EmbodiedAgent(
        lrsg_config=lrsg_config,
        locomotion_config=loc_config,
        embodied_config=emb_config,
        device=device
    )

    # Create visualizer if not headless
    visualizer = None
    if not headless:
        print("Initializing visualizer...")
        visualizer = EmbodiedVisualizer(port=8080)
        visualizer.start()
        print("Visualizer running on http://localhost:8080")

    # Episode telemetry collection
    episode_log: List[Dict] = []

    # Running statistics
    episode_rewards = []
    episode_successes = []

    # Main loop: run episodes
    print(f"\nRunning {num_episodes} episodes...\n")
    for episode in range(num_episodes):
        # Reset agent for new episode
        agent._reset_episode()

        # Collect data for this episode
        episode_data = {
            'episode': episode,
            'total_reward': 0.0,
            'goal_reached': False,
            'max_enemies_seen': 0,
            'steps': 0,
            'final_distance_to_goal': 0.0,
        }

        # Run episode until done or max steps
        while agent.tick_count < emb_config.episode_length:
            # Step agent
            step_result = agent.step()

            # Accumulate episode data
            episode_data['total_reward'] += step_result['reward']
            episode_data['max_enemies_seen'] = max(
                episode_data['max_enemies_seen'],
                step_result['num_enemies']
            )
            episode_data['steps'] += 1
            episode_data['final_distance_to_goal'] = step_result['distance_to_goal']

            # Check if goal reached
            if step_result['is_goal_reached']:
                episode_data['goal_reached'] = True
                break

            # Episode can end early (e.g. collision timeout), which resets
            # agent.tick_count to 0 internally — stop the loop here too.
            if step_result['episode_done']:
                break

            # Update visualizer if running
            if visualizer:
                # Prepare enemy data for visualizer
                enemies_data = []
                for enemy in agent.env.get_enemies():
                    enemies_data.append({
                        'position': list(enemy.position),
                        'health': enemy.health,
                        'max_health': enemy.max_health,
                        'strength': getattr(enemy, 'strength', 1.0),
                    })

                # Get semantic state
                semantic_state = agent._build_observation()

                # Update visualizer state
                visualizer.update_state(
                    agent_pos=tuple(agent.agent_pos[:2]),
                    goal_pos=tuple(agent.goal_pos[:2]),
                    enemies=enemies_data,
                    obstacles=[],  # BattlegroundEnvironment doesn't expose obstacles easily
                    episode=episode,
                    tick=agent.tick_count,
                    current_reward=step_result['reward'],
                    accumulated_reward=episode_data['total_reward'],
                    semantic_state=semantic_state,
                )

        # Log episode reward in visualizer
        if visualizer:
            visualizer.log_episode_reward(episode, episode_data['total_reward'])

        # Append to episode log
        episode_log.append(episode_data)
        episode_rewards.append(episode_data['total_reward'])
        episode_successes.append(1 if episode_data['goal_reached'] else 0)

        # Progress reporting every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            success_count = sum(episode_successes[-10:])
            success_rate = success_count / 10.0
            print(
                f"Episode {episode + 1:3d}: "
                f"Avg Reward={avg_reward:8.2f}, "
                f"Success={success_rate * 100:5.1f}%"
            )

    # Ensure experiment_results directory exists
    experiment_results_dir = Path('experiment_results')
    experiment_results_dir.mkdir(parents=True, exist_ok=True)

    # Save telemetry
    telemetry_path = experiment_results_dir / 'embodied_battleground_telemetry.jsonl'
    print(f"\nSaving telemetry to {telemetry_path}...")
    with open(telemetry_path, 'w') as f:
        for record in episode_log:
            f.write(json.dumps(record) + '\n')

    # Print final statistics
    total_reward = sum(episode_rewards)
    success_count = sum(episode_successes)
    success_rate = success_count / num_episodes
    avg_reward = np.mean(episode_rewards)

    # Compute learning improvement (avg last 20 - avg first 20)
    first_20_avg = np.mean(episode_rewards[:20]) if len(episode_rewards) >= 20 else np.mean(episode_rewards[:10])
    last_20_avg = np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards[-10:])
    learning_improvement = last_20_avg - first_20_avg

    # Compute max reward
    max_reward = max(episode_rewards) if episode_rewards else 0.0

    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    print(f"  Total episodes: {num_episodes}")
    print(f"  Total reward: {total_reward:,.2f}")
    print(f"  Success rate: {success_rate * 100:.1f}%")
    print(f"  Avg reward per episode: {avg_reward:.2f}")
    print(f"  Max reward: {max_reward:.2f}")
    print(f"  Learning improvement: {learning_improvement:.2f}")
    print("=" * 60)

    # Return summary statistics as dict
    return {
        'episode_rewards': episode_rewards,
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'max_reward': max_reward,
        'learning_improvement': learning_improvement,
        'episode_log': episode_log,
    }


def main():
    """Parse arguments and run embodied learning."""
    parser = argparse.ArgumentParser(
        description='Episode runner for embodied learning battleground'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help='Number of episodes to run (default 100)'
    )
    parser.add_argument(
        '--neurons',
        type=int,
        default=4_048_576,
        help='Number of neurons in LRSG (default 4048576)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda'],
        help='Device to use (cpu or cuda, default auto-detect)'
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run without visualization (for CI/headless runs)'
    )

    args = parser.parse_args()

    # Run embodied learning
    episode_log = run_embodied_learning(
        num_episodes=args.episodes,
        num_neurons=args.neurons,
        device=args.device,
        headless=args.headless,
    )

    print("\nEpisode runner completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
