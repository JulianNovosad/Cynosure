# Embodied Learning Battleground: Design Specification
**Date:** 2026-06-24  
**Goal:** Build a complete intelligent embodied learning agent system where a 4m-neuron LRSG agent learns to navigate, fight, and survive in a procedural battleground over 100+ episodes, with real-time visualization.

---

## 1. System Overview

**Core Components:**
- **EmbodiedAgent:** LRSG brain (4M neurons) + differential-drive locomotion + sensor fusion
- **BattlegroundEnvironment:** Procedural 2D grid with randomized obstacles, enemies, goal position
- **EnemyNPCs:** 3-5 simple enemies per episode (random walk + chase/evade behavior, no LRSG)
- **ObservationEncoder:** 18D raw observations → 64D semantic state vector
- **CombatSystem:** Strength comparison → attack success probability
- **RewardShaper:** Distance-to-goal + combat victories + survival bonus
- **VisualizationDashboard:** Real-time web canvas (position, enemies, goal, reward/episode curve)

---

## 2. Observation & Semantic State (64D)

**Raw Observations (18D):**
- Position (3): x, y, z
- Velocity (3): vx, vy, vz
- LiDAR (8): 8 ray distances
- Goal delta (2): Δx, Δy to goal
- Distance to goal (1): scalar
- Safe zone flag (1): always 0 for grid env

**Semantic State Mapping (64D):**

| Domain | States (7D per domain) | Details |
|--------|------------------------|---------|
| **Navigation** (7D) | Path confidence, obstacle proximity, energy state, goal alignment, position entropy, terrain difficulty, progress toward goal | Derived from position, LiDAR, distance-to-goal |
| **Threat Assessment** (7D) | Nearest enemy distance, threat level, enemy count, escape feasibility, attack readiness, evasion priority, dominance estimate | Derived from enemy positions/health |
| **Tactical** (7D) | Exploration drive, fight urge, retreat bias, speed preference, stealth mode, resource priority, action diversity | Derived from reward history, combat outcomes |
| **Energy & Survival** (7D) | Energy level, fatigue, hunger signal, damage taken, recovery potential, death risk, vitality | Derived from episode reward accumulation |
| **World Model** (7D) | Spatial predictability, temporal consistency, model confidence, novelty detection, pattern recognition, anomaly score, learning readiness | Derived from observation variance |
| **Self Model** (7D) | Capability confidence, skill level, learned strength, decision stability, action consistency, behavioral coherence, strategy clarity | Derived from success rates, action variance |
| **Meta-Strategy** (7D) | Long-term goal commitment, flexibility index, learning rate (adaptive), exploration budget, exploitation weight, strategic phase, evolution metric | Derived from episode-to-episode improvement |

**Encoder Method:** Hand-designed heuristic functions (not learned neural nets)—each semantic dimension computed directly from raw observations and reward history, ensuring interpretability.

---

## 3. BattlegroundEnvironment

**Procedural Generation (per episode):**
- **Grid:** 20×20 tiles, walls/obstacles spawned randomly
- **Start position:** Random open tile
- **Goal position:** Random open tile (>10 units from start)
- **Enemies:** 3-5 NPCs, spawn at random locations (>5 units from start and goal)
- **Reset:** Every 300 ticks or when goal reached

**Enemy Behavior:**
- **Strength:** Random HP (20-100), speed (0.5-1.5x agent speed)
- **AI:** Simple state machine:
  - Random walk (50% of time)
  - Chase agent if within 5 units (30%)
  - Flee if agent is stronger (20%)
- **Combat:** When agent attacks and within range (1 unit):
  - Win probability = 50% if agent_strength > enemy_strength
  - On win: enemy dies, +50 reward, restore energy
  - On loss: agent takes 20 damage, −10 reward

---

## 4. Combat & Reward System

**Agent Strength Metric:**
- Base: 50
- Scales with accumulated episode reward (capped at 100)
- Higher strength = higher win probability in combat

**Reward Shaping (per tick):**
- **Navigation:** `+0.1 * (distance_reduction)` — reward for moving closer to goal
- **Goal reached:** `+100` — major victory bonus
- **Combat victory:** `+50`
- **Collision:** `−5`
- **Enemy proximity (too close & weaker):** `−10` (encourages evasion)
- **Idle action:** `−1` (encourages action)

**Learning Trigger:**
- Hebbian plasticity applied after each episode
- Accumulated episode reward (sum of all ticks) used as learning signal
- Learning rate: 0.01 (configured)
- Eligibility trace reset between episodes

---

## 5. Real-Time Visualization

**Web Dashboard (Port 8080):**
- **Canvas:** 20×20 grid, agent at center (fixed), world scrolls around it
- **Rendering:**
  - Agent: blue circle
  - Goal: gold star
  - Enemies: red circles (size ∝ HP)
  - Obstacles: gray squares
  - Trail: light blue path history (last 20 ticks)
- **Metrics Panel (right sidebar):**
  - Episode #, tick count, current reward
  - Agent strength, energy state
  - Enemies visible: count + list (distance, HP, threat level)
  - Semantic state (7D heatmap showing most active dimensions)
- **Learning Curve:** Accumulated reward per episode (updated every episode)

---

## 6. Integration Points

**Modified Files:**
- `src/simulation/embodied_loop.py` → Add enemy system, expand observation to 64D
- `src/simulation/embodied_reward.py` → Enhanced reward shaping (navigation + combat + survival)
- `src/simulation/navigation.py` → Procedural generation + enemy spawning
- `src/simulation/embodied_visualizer.py` (NEW) → Web dashboard

**New Files:**
- `src/simulation/enemy_npc.py` → Simple enemy behavior
- `src/simulation/semantic_encoder.py` → 64D semantic state mapping
- `src/simulation/combat_system.py` → Attack/defense logic
- `src/simulation/battleground_env.py` (extends NavigationEnvironment) → Procedural generation + enemy management

---

## 7. Success Criteria

- **Learning Observable:** Agent improves from ~5% goal success in early episodes to >70% by episode 50
- **Combat Learning:** Agent learns to attack weak enemies (~40% of encounters) and evade strong ones (~70% success)
- **Visualization:** Real-time dashboard updates smoothly (30+ FPS), shows agent position, enemies, reward curve
- **Semantic State:** At least 3 dimensions correlate visibly with learned behavior changes
- **Run Duration:** 100 episodes × 300 ticks = 30K ticks, completes in <10 minutes on RTX 2070

---

## 8. Implementation Phases

**Phase 1 (Day 1):** Enemy NPC system + procedural battleground
**Phase 2 (Day 1-2):** Combat system + reward shaping
**Phase 3 (Day 2):** Semantic state encoder (64D)
**Phase 4 (Day 2-3):** Visualization dashboard
**Phase 5 (Day 3):** Integration + learning loop validation + episode runner

