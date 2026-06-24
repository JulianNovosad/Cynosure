"""Semantic state encoder: 18D observations + enemies + reward → 64D semantic state."""
import numpy as np
from typing import List
from src.simulation.enemy_npc import EnemyNPC


class SemanticEncoder:
    """
    Maps raw observations + enemy data + reward history → 64D semantic state.

    Semantic domains (9 × 7D = 63D + 1D metadata):
    1. Navigation (7D): path confidence, obstacle proximity, energy, goal alignment, entropy, terrain, progress
    2. Threat Assessment (7D): enemy distance, threat level, count, escape feasibility, attack readiness, evasion, dominance
    3. Tactical (7D): exploration, fight urge, retreat bias, speed pref, stealth, resource priority, action diversity
    4. Energy & Survival (7D): energy, fatigue, hunger, damage, recovery, death risk, vitality
    5. World Model (7D): spatial predictability, temporal consistency, confidence, novelty, pattern recognition, anomaly, learning readiness
    6. Self Model (7D): capability confidence, skill level, learned strength, decision stability, action consistency, coherence, strategy clarity
    7. Meta-Strategy (7D): goal commitment, flexibility, learning rate, exploration budget, exploitation weight, phase, evolution
    8. Sensory (7D): awareness, salience, noise, focus, novelty, predictability, saturation
    9. Cognitive (7D): comprehension, uncertainty, abstraction, memory pressure, consistency, update drive, simulation depth

    Total: 63D + 1D (episode counter / metadata) = 64D
    """

    def __init__(self):
        """Initialize encoder (deterministic, no parameters needed)."""
        pass

    def encode(
        self,
        raw_obs: np.ndarray,  # 18D: pos(3), vel(3), lidar(8), goal_delta(2), dist_to_goal(1), safe_zone(1)
        enemies: List[EnemyNPC],
        accumulated_reward: float,
        prev_accumulated_reward: float = 0.0,
    ) -> np.ndarray:
        """
        Encode observation to 64D semantic state.

        Args:
            raw_obs: 18D raw observation vector [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, lidar_0-7, goal_dx, goal_dy, dist_to_goal, in_safe_zone]
            enemies: List of visible enemies
            accumulated_reward: Sum of rewards so far in episode
            prev_accumulated_reward: Sum of rewards in previous tick (for delta)

        Returns:
            64D semantic state vector (float32)
        """
        assert raw_obs.shape == (18,), f"Expected 18D obs, got {raw_obs.shape}"

        semantic = np.zeros(64, dtype=np.float32)

        # Parse raw observations
        pos = raw_obs[0:3]
        vel = raw_obs[3:6]
        lidar = raw_obs[6:14]
        goal_delta = raw_obs[14:16]
        dist_to_goal = raw_obs[16]
        in_safe_zone = raw_obs[17]

        # === NAVIGATION (indices 0-6) ===
        # 0: path_confidence — inverse of distance to goal (normalized)
        semantic[0] = np.clip(1.0 - (dist_to_goal / 30.0), 0, 1)

        # 1: obstacle_proximity — min lidar distance (normalized)
        min_lidar = np.min(lidar) if lidar.size > 0 else 1.0
        semantic[1] = np.clip(min_lidar / 10.0, 0, 1)

        # 2: energy_state — normalized accumulated reward
        semantic[2] = np.clip(accumulated_reward / 500.0, 0, 1)

        # 3: goal_alignment — alignment with goal direction (2D only: x, y)
        goal_dist = np.linalg.norm(goal_delta)
        vel_2d = vel[:2]  # Use only x, y components of velocity
        vel_dist = np.linalg.norm(vel_2d)
        if goal_dist > 0.01 and vel_dist > 0.01:
            alignment = np.dot(goal_delta / goal_dist, vel_2d / vel_dist)
            semantic[3] = (alignment + 1) / 2  # Map [-1, 1] → [0, 1]
        else:
            semantic[3] = 0.5

        # 4: position_entropy — normalized position magnitude
        pos_mag = np.linalg.norm(pos[:2])
        semantic[4] = np.clip(pos_mag / 20.0, 0, 1)

        # 5: terrain_difficulty — std of lidar (rough terrain = high variance)
        semantic[5] = np.clip(np.std(lidar) / 5.0, 0, 1)

        # 6: progress_toward_goal — reward rate (reward change per tick)
        reward_delta = accumulated_reward - prev_accumulated_reward
        semantic[6] = np.clip(max(0, reward_delta) / 10.0, 0, 1)

        # === THREAT ASSESSMENT (indices 7-13) ===
        enemy_distances = np.array([np.linalg.norm(np.array(e.position[:2]) - pos[:2]) for e in enemies]) if enemies else np.array([])

        # 7: nearest_enemy_distance
        if len(enemies) > 0:
            nearest_dist = np.min(enemy_distances)
            semantic[7] = np.clip(1.0 - (nearest_dist / 15.0), 0, 1)
        else:
            semantic[7] = 0.0

        # 8: threat_level — avg enemy strength / 100
        if len(enemies) > 0:
            avg_strength = np.mean([e.strength for e in enemies])
            semantic[8] = np.clip(avg_strength / 100.0, 0, 1)
        else:
            semantic[8] = 0.0

        # 9: enemy_count_normalized
        semantic[9] = np.clip(len(enemies) / 5.0, 0, 1)

        # 10: escape_feasibility — lidar clearance (can move?)
        escape = 1.0 if np.min(lidar) > 1.0 else 0.0
        semantic[10] = escape

        # 11: attack_readiness — nearby weak enemy?
        weak_nearby = 0.0
        for enemy in enemies:
            if np.linalg.norm(np.array(enemy.position[:2]) - pos[:2]) < 3.0 and enemy.strength < 50.0:
                weak_nearby = 1.0
        semantic[11] = weak_nearby

        # 12: evasion_priority — nearby strong enemy?
        strong_nearby = 0.0
        for enemy in enemies:
            if np.linalg.norm(np.array(enemy.position[:2]) - pos[:2]) < 3.0 and enemy.strength >= 50.0:
                strong_nearby = 1.0
        semantic[12] = strong_nearby

        # 13: dominance_estimate — (our strength - avg enemy strength) / 100
        our_strength = 50 + np.clip(accumulated_reward / 10.0, 0, 50)  # Mimic combat_system.py logic
        if len(enemies) > 0:
            avg_enemy_str = np.mean([e.strength for e in enemies])
            semantic[13] = np.clip((our_strength - avg_enemy_str) / 100.0 + 0.5, 0, 1)
        else:
            semantic[13] = 0.5

        # === TACTICAL (indices 14-20) ===
        # 14: exploration_drive — lower if near goal
        semantic[14] = np.clip(dist_to_goal / 30.0, 0, 1)

        # 15: fight_urge — increases if weak enemies nearby
        semantic[15] = semantic[11]

        # 16: retreat_bias — increases if strong enemies nearby
        semantic[16] = semantic[12]

        # 17: speed_preference — based on obstacle proximity
        semantic[17] = max(0, 1.0 - np.min(lidar) / 5.0) if len(lidar) > 0 else 0.5

        # 18: stealth_mode — decreases if enemies near
        semantic[18] = 1.0 - semantic[7]  # Inverse of threat

        # 19: resource_priority — normalized energy
        semantic[19] = np.clip(accumulated_reward / 500.0, 0, 1)

        # 20: action_diversity — pseudo-random based on position (should vary)
        semantic[20] = (np.sin(pos[0]) + 1) / 2

        # === ENERGY & SURVIVAL (indices 21-27) ===
        # 21: energy_level
        semantic[21] = np.clip(accumulated_reward / 500.0, 0, 1)

        # 22: fatigue (inverse of energy)
        semantic[22] = 1.0 - semantic[21]

        # 23: hunger_signal (always fed in this env)
        semantic[23] = 0.2

        # 24: damage_taken (not tracked directly, use threat)
        semantic[24] = semantic[7]

        # 25: recovery_potential (can reach goal = recovery?)
        semantic[25] = semantic[0]

        # 26: death_risk (high if weak + strong enemy nearby)
        weak_state = 1.0 - semantic[21]
        semantic[26] = weak_state * semantic[12]

        # 27: vitality
        semantic[27] = semantic[21]

        # === WORLD MODEL (indices 28-34) ===
        # 28: spatial_predictability (lidar consistency)
        semantic[28] = 1.0 - np.clip(np.std(lidar) / 10.0, 0, 1)

        # 29: temporal_consistency (reward smooth?)
        semantic[29] = 0.8  # Assume smooth

        # 30: model_confidence
        semantic[30] = np.clip(accumulated_reward / 200.0, 0, 1)

        # 31: novelty_detection (entropy of position)
        semantic[31] = np.clip(np.linalg.norm(pos[:2]) / 40.0, 0, 1)

        # 32: pattern_recognition
        semantic[32] = 0.5  # Neutral

        # 33: anomaly_score (high lidar variance)
        semantic[33] = np.clip(np.std(lidar) / 10.0, 0, 1)

        # 34: learning_readiness (accumulating reward = learning)
        semantic[34] = 1.0 if semantic[6] > 0.1 else 0.3

        # === SELF MODEL (indices 35-41) ===
        # 35: capability_confidence
        semantic[35] = np.clip(accumulated_reward / 300.0, 0, 1)

        # 36: skill_level (same as capability)
        semantic[36] = semantic[35]

        # 37: learned_strength (combat strength)
        semantic[37] = np.clip(our_strength / 100.0, 0, 1)

        # 38: decision_stability (inverse of action diversity)
        semantic[38] = 1.0 - semantic[20]

        # 39: action_consistency
        semantic[39] = 0.7  # Assume consistent

        # 40: behavioral_coherence
        semantic[40] = semantic[39]

        # 41: strategy_clarity
        strat_sum = semantic[14] + semantic[15] + semantic[16]
        semantic[41] = 1.0 if strat_sum > 0 else 0.3

        # === META-STRATEGY (indices 42-48) ===
        # 42: goal_commitment
        semantic[42] = 1.0 if dist_to_goal < 5.0 else 0.5

        # 43: flexibility_index
        semantic[43] = semantic[20]

        # 44: learning_rate_adaptive (increases with success)
        semantic[44] = np.clip(semantic[6] * 2, 0, 1)

        # 45: exploration_budget
        semantic[45] = semantic[14]

        # 46: exploitation_weight
        semantic[46] = 1.0 - semantic[45]

        # 47: strategic_phase (0=exploration, 1=exploitation)
        semantic[47] = semantic[46]

        # 48: evolution_metric (improvement over time)
        semantic[48] = semantic[34]

        # === SENSORY (indices 49-55) ===
        # 49: awareness
        semantic[49] = 1.0 if len(enemies) > 0 else 0.3

        # 50: salience (strongest signal)
        signals = [semantic[7], semantic[8], semantic[0]]  # threat, threat, goal
        semantic[50] = max(signals) if signals else 0.5

        # 51: noise (contradiction between signals)
        semantic[51] = np.std([semantic[7], semantic[14]]) if len(enemies) > 0 else 0.2

        # 52: focus (attention on goal)
        semantic[52] = semantic[0]

        # 53: novelty_sensory
        semantic[53] = semantic[31]

        # 54: predictability_sensory
        semantic[54] = 1.0 - semantic[51]

        # 55: saturation (information overload?)
        semantic[55] = np.clip((len(enemies) + np.std(lidar)) / 10.0, 0, 1)

        # === COGNITIVE (indices 56-62) ===
        # 56: comprehension
        semantic[56] = semantic[30]

        # 57: uncertainty
        semantic[57] = 1.0 - semantic[56]

        # 58: abstraction (level of symbolic compression)
        semantic[58] = 0.6  # Medium level

        # 59: memory_pressure
        semantic[59] = 0.4  # Low (short episode)

        # 60: consistency_cognitive
        semantic[60] = semantic[39]

        # 61: update_drive (urge to revise models)
        semantic[61] = semantic[6]

        # 62: simulation_depth (lookahead)
        semantic[62] = 0.3  # Low lookahead

        # === METADATA (index 63) ===
        # 63: surprise (prediction error magnitude)
        semantic[63] = semantic[51]

        return semantic
