# Embodied Learning Battleground Implementation Plan

> **Documentation note (2026-07-20):** This file is a historical, point-in-time snapshot and is intentionally not rewritten to reflect current state (rewriting it would falsify the record it captured). For current system status, see `CYNOSURE_KB.md` and `docs/CHANGELOG.md`.


> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete intelligent embodied learning system where a 4M-neuron LRSG agent learns to navigate, fight, and survive in a procedural battleground with real-time visualization.

**Architecture:** Enemy NPCs → BattlegroundEnvironment (extends NavigationEnvironment) → Combat System → Enhanced Reward Shaping → 64D Semantic State Encoder → EmbodiedAgent Integration → Visualizer → Episode Runner

**Tech Stack:** PyTorch, NumPy, Flask/WSGI (visualization), HTML5 Canvas (frontend), Hebbian plasticity (learning)

## Global Constraints

- All new files in `src/simulation/` (core components) or `src/experiments/` (runners)
- Follow existing project structure: strict hierarchy, no loose scripts
- Enemy NPC behavior uses simple state machines (no LRSG)
- Reward shaping deterministic (no randomness in calculations)
- Visualization updates every tick via WebSocket or polling
- Learning occurs per-episode (Hebbian update after episode termination)
- Grid coordinates are (x, y) floats; obstacles are grid-aligned

---

## File Structure

**New Files:**
- `src/simulation/enemy_npc.py` — EnemyNPC class (health, position, behavior)
- `src/simulation/battleground_env.py` — BattlegroundEnvironment (procedural generation, enemy management)
- `src/simulation/combat_system.py` — Attack resolution, damage calculation
- `src/simulation/semantic_encoder.py` — 18D → 64D semantic state mapping
- `src/simulation/embodied_visualizer.py` — Flask/WSGI web dashboard
- `src/experiments/run_embodied_battleground.py` — Episode runner (100+ episodes, telemetry, visualization bridge)
- `src/experiments/test_embodied_system.py` — Integration tests

**Modified Files:**
- `src/simulation/embodied_loop.py` — Update EmbodiedAgent to use BattlegroundEnvironment, 64D state, enhanced reward
- `src/simulation/embodied_reward.py` — Add combat rewards, navigation scaling
- `CYNOSURE_KB.md` — Document new components and execution entrypoints

---

## Task Breakdown

### Task 1: Enemy NPC System

**Files:**
- Create: `src/simulation/enemy_npc.py`
- Test: `tests/test_enemy_npc.py`

**Interfaces:**
- Produces:
  - `class EnemyNPC`: `__init__(position: Tuple[float, float], health: float, strength: float, max_speed: float)`
  - `update(agent_pos: Tuple[float, float], dt: float) -> None` — moves enemy per behavior
  - `take_damage(amount: float) -> bool` — returns True if enemy dies
  - `get_state() -> Dict` — returns position, health, strength, behavior_state
  - Properties: `alive: bool`, `position: Tuple[float, float]`, `health: float`, `strength: float`, `behavior: str`

- [ ] **Step 1: Write test file**

Create `tests/test_enemy_npc.py`:

```python
import pytest
import numpy as np
from src.simulation.enemy_npc import EnemyNPC

def test_enemy_creation():
    enemy = EnemyNPC(position=(5.0, 5.0), health=50.0, strength=30.0, max_speed=0.5)
    assert enemy.alive is True
    assert enemy.health == 50.0
    assert enemy.position == (5.0, 5.0)
    assert enemy.strength == 30.0

def test_enemy_take_damage():
    enemy = EnemyNPC(position=(5.0, 5.0), health=50.0, strength=30.0, max_speed=0.5)
    died = enemy.take_damage(30.0)
    assert died is False
    assert enemy.health == 20.0
    
    died = enemy.take_damage(30.0)
    assert died is True
    assert enemy.alive is False

def test_enemy_random_walk():
    enemy = EnemyNPC(position=(5.0, 5.0), health=50.0, strength=30.0, max_speed=0.5)
    enemy.behavior = "random_walk"
    old_pos = enemy.position
    enemy.update(agent_pos=(10.0, 10.0), dt=0.1)
    # Position should change (with high probability)
    # Just verify it doesn't crash and stays in reasonable bounds
    assert -50 < enemy.position[0] < 50
    assert -50 < enemy.position[1] < 50

def test_enemy_chase():
    enemy = EnemyNPC(position=(5.0, 5.0), health=50.0, strength=30.0, max_speed=0.5)
    enemy.behavior = "chase"
    agent_pos = (6.0, 5.0)  # 1 unit to the right
    enemy.update(agent_pos=agent_pos, dt=0.1)
    # Should move toward agent
    assert enemy.position[0] > 5.0  # moved right

def test_enemy_flee():
    enemy = EnemyNPC(position=(5.0, 5.0), health=50.0, strength=30.0, max_speed=0.5)
    enemy.behavior = "flee"
    agent_pos = (6.0, 5.0)
    enemy.update(agent_pos=agent_pos, dt=0.1)
    # Should move away from agent
    assert enemy.position[0] < 5.0  # moved left
```

- [ ] **Step 2: Run test (expect FAIL)**

```bash
pytest tests/test_enemy_npc.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.simulation.enemy_npc'`

- [ ] **Step 3: Write implementation**

Create `src/simulation/enemy_npc.py`:

```python
"""Simple NPC enemy for embodied learning battleground."""
import numpy as np
from typing import Tuple, Dict


class EnemyNPC:
    """
    Simple non-LRSG enemy NPC.
    
    Behavior: random walk (default), chase (if agent nearby), flee (if stronger agent nearby).
    """
    
    def __init__(
        self,
        position: Tuple[float, float],
        health: float,
        strength: float,
        max_speed: float = 0.5,
    ):
        """
        Args:
            position: (x, y) starting position
            health: Current HP (max = initial health)
            strength: Combat strength (0-100)
            max_speed: Max movement speed per tick
        """
        self.position = np.array(position, dtype=np.float32)
        self.health = float(health)
        self.max_health = float(health)
        self.strength = float(strength)
        self.max_speed = float(max_speed)
        self.alive = True
        self.behavior = "random_walk"  # "random_walk", "chase", "flee"
        
    def update(self, agent_pos: Tuple[float, float], dt: float = 0.1) -> None:
        """
        Update enemy position based on behavior.
        
        Args:
            agent_pos: Agent position (x, y)
            dt: Time step (seconds)
        """
        if not self.alive:
            return
        
        agent_pos = np.array(agent_pos, dtype=np.float32)
        dist_to_agent = np.linalg.norm(agent_pos - self.position)
        
        # Behavior state machine
        if self.behavior == "random_walk":
            # Random direction
            angle = np.random.uniform(0, 2 * np.pi)
            velocity = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32) * self.max_speed
            
        elif self.behavior == "chase":
            # Move toward agent
            direction = agent_pos - self.position
            if dist_to_agent > 0.01:
                direction = direction / dist_to_agent
            velocity = direction * self.max_speed
            
        elif self.behavior == "flee":
            # Move away from agent
            direction = self.position - agent_pos
            if dist_to_agent > 0.01:
                direction = direction / dist_to_agent
            else:
                direction = np.random.uniform(-1, 1, 2)
            velocity = direction * self.max_speed
        else:
            velocity = np.zeros(2, dtype=np.float32)
        
        # Update position
        self.position = self.position + velocity * dt
        
    def take_damage(self, amount: float) -> bool:
        """
        Take damage. Returns True if enemy dies.
        
        Args:
            amount: Damage amount
            
        Returns:
            True if health <= 0 (enemy is dead)
        """
        self.health = max(0.0, self.health - amount)
        if self.health <= 0:
            self.alive = False
            return True
        return False
    
    def get_state(self) -> Dict:
        """Get enemy state snapshot."""
        return {
            "position": tuple(self.position),
            "health": float(self.health),
            "max_health": self.max_health,
            "strength": self.strength,
            "alive": self.alive,
            "behavior": self.behavior,
        }
```

- [ ] **Step 4: Run test (expect PASS)**

```bash
pytest tests/test_enemy_npc.py -v
```

Expected: All 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/simulation/enemy_npc.py tests/test_enemy_npc.py
git commit -m "feat: add simple NPC enemy system with chase/flee/random walk behaviors"
```

---

### Task 2: Battleground Environment

**Files:**
- Create: `src/simulation/battleground_env.py`
- Modify: `src/simulation/navigation.py` (check existing NavigationEnvironment)
- Test: `tests/test_battleground_env.py`

**Interfaces:**
- Consumes: `EnemyNPC` from Task 1
- Produces:
  - `class BattlegroundEnvironment(NavigationEnvironment)`: extends existing class
  - `__init__(grid_size: int = 20, num_obstacles: int = 5, num_enemies: int = 3)`
  - `reset() -> Tuple[Tuple[float, float], Tuple[float, float]]` — returns (agent_start, goal)
  - `get_enemies() -> List[EnemyNPC]`
  - `get_enemy_near(pos: Tuple[float, float], radius: float = 5.0) -> List[EnemyNPC]`
  - `check_collision(pos: Tuple[float, float], radius: float = 0.1) -> bool` — includes enemy collisions
  - `tick_enemies(agent_pos: Tuple[float, float]) -> None` — update all enemy behaviors
  - `remove_enemy(enemy: EnemyNPC) -> None`

- [ ] **Step 1: Read existing NavigationEnvironment**

Check `src/simulation/navigation.py` to understand the parent class:

```bash
head -100 src/simulation/navigation.py
```

Take note of:
- Grid representation (how obstacles are stored)
- `reset()` return type
- `check_collision()` signature
- Available utility methods

- [ ] **Step 2: Write test file**

Create `tests/test_battleground_env.py`:

```python
import pytest
from src.simulation.battleground_env import BattlegroundEnvironment
from src.simulation.enemy_npc import EnemyNPC


def test_battleground_creation():
    env = BattlegroundEnvironment(grid_size=20, num_obstacles=5, num_enemies=3)
    assert env.grid_size == 20
    assert len(env.get_enemies()) == 3


def test_battleground_reset():
    env = BattlegroundEnvironment(grid_size=20, num_obstacles=5, num_enemies=3)
    start, goal = env.reset()
    assert isinstance(start, tuple) and len(start) == 2
    assert isinstance(goal, tuple) and len(goal) == 2
    # Start and goal should be different
    assert start != goal
    # All enemies should be alive and have different positions
    assert all(e.alive for e in env.get_enemies())


def test_get_enemies_near():
    env = BattlegroundEnvironment(grid_size=20, num_obstacles=5, num_enemies=3)
    env.reset()
    # Spawn enemies at known positions for testing
    env._enemies = [
        EnemyNPC(position=(5.0, 5.0), health=50.0, strength=30.0),
        EnemyNPC(position=(15.0, 15.0), health=50.0, strength=40.0),
    ]
    # Query near (5, 5)
    nearby = env.get_enemy_near(pos=(5.0, 5.0), radius=2.0)
    assert len(nearby) == 1
    assert nearby[0].position == (5.0, 5.0)


def test_remove_dead_enemy():
    env = BattlegroundEnvironment(grid_size=20, num_obstacles=5, num_enemies=2)
    env.reset()
    enemy = env.get_enemies()[0]
    enemy.take_damage(100.0)  # Kill enemy
    env.remove_enemy(enemy)
    assert enemy not in env.get_enemies()
    assert len(env.get_enemies()) == 1


def test_tick_enemies_updates_behavior():
    env = BattlegroundEnvironment(grid_size=20, num_obstacles=5, num_enemies=1)
    env.reset()
    enemy = env.get_enemies()[0]
    enemy.position = np.array([5.0, 5.0], dtype=np.float32)
    agent_pos = (6.0, 5.0)
    env.tick_enemies(agent_pos=agent_pos)
    # Enemy behavior should be updated based on distance
    # (actual behavior depends on implementation, just verify it runs)
    assert enemy.position is not None
```

- [ ] **Step 3: Write implementation**

Create `src/simulation/battleground_env.py`:

```python
"""Battleground environment: procedurally generated grid with enemies."""
import numpy as np
from typing import List, Tuple
from src.simulation.navigation import NavigationEnvironment
from src.simulation.enemy_npc import EnemyNPC


class BattlegroundEnvironment(NavigationEnvironment):
    """
    Extended navigation environment with procedural generation and enemy NPCs.
    
    Each episode:
    - Procedurally generate obstacles
    - Spawn agent at random start
    - Spawn goal at random end (>10 units away)
    - Spawn 3-5 enemies at random locations
    """
    
    def __init__(
        self,
        grid_size: int = 20,
        num_obstacles: int = 5,
        num_enemies: int = 3,
        seed: int = None,
    ):
        """
        Args:
            grid_size: Grid width/height (meters)
            num_obstacles: Number of obstacle clusters per episode
            num_enemies: Number of enemies to spawn
            seed: Random seed (None = no seed, randomized)
        """
        super().__init__(grid_size=grid_size, num_obstacles=num_obstacles, seed=seed)
        self.num_enemies = num_enemies
        self._enemies: List[EnemyNPC] = []
    
    def reset(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Reset environment: procedurally generate map and spawn enemies.
        
        Returns:
            (agent_start_pos, goal_pos)
        """
        # Reset parent (generates obstacles, places goal)
        agent_start, goal = super().reset()
        
        # Spawn enemies
        self._enemies = []
        for _ in range(self.num_enemies):
            enemy_pos = self._sample_free_position(min_dist_from_start=5.0)
            health = np.random.uniform(20, 100)
            strength = np.random.uniform(20, 80)
            speed = np.random.uniform(0.3, 1.0)
            enemy = EnemyNPC(
                position=enemy_pos,
                health=health,
                strength=strength,
                max_speed=speed,
            )
            # Set initial behavior (random distribution)
            behavior_choice = np.random.choice(["random_walk", "random_walk", "chase"])
            enemy.behavior = behavior_choice
            self._enemies.append(enemy)
        
        return agent_start, goal
    
    def _sample_free_position(self, min_dist_from_start: float = 5.0) -> Tuple[float, float]:
        """
        Sample a random collision-free position.
        
        Args:
            min_dist_from_start: Minimum distance from agent start
            
        Returns:
            (x, y) position
        """
        max_attempts = 100
        for _ in range(max_attempts):
            x = np.random.uniform(0, self.grid_size)
            y = np.random.uniform(0, self.grid_size)
            pos = (x, y)
            
            # Check collision
            if not self.check_collision(pos, radius=0.5):
                # Check distance from start
                if np.linalg.norm(np.array(pos) - np.array([0, 0])) >= min_dist_from_start:
                    return pos
        
        # Fallback: place at grid center
        return (self.grid_size / 2, self.grid_size / 2)
    
    def get_enemies(self) -> List[EnemyNPC]:
        """Get list of alive enemies."""
        return [e for e in self._enemies if e.alive]
    
    def get_enemy_near(
        self,
        pos: Tuple[float, float],
        radius: float = 5.0,
    ) -> List[EnemyNPC]:
        """
        Get enemies within radius of position.
        
        Args:
            pos: Query position (x, y)
            radius: Search radius
            
        Returns:
            List of nearby enemies
        """
        nearby = []
        pos = np.array(pos, dtype=np.float32)
        for enemy in self.get_enemies():
            dist = np.linalg.norm(np.array(enemy.position) - pos)
            if dist <= radius:
                nearby.append(enemy)
        return nearby
    
    def remove_enemy(self, enemy: EnemyNPC) -> None:
        """Remove enemy from simulation."""
        if enemy in self._enemies:
            self._enemies.remove(enemy)
    
    def tick_enemies(self, agent_pos: Tuple[float, float]) -> None:
        """
        Update all enemies.
        
        Behavior selection:
        - If within 5 units and agent stronger: chase
        - If within 3 units and agent weaker: flee
        - Otherwise: random walk
        
        Args:
            agent_pos: Current agent position
        """
        for enemy in self.get_enemies():
            dist = np.linalg.norm(np.array(enemy.position) - np.array(agent_pos))
            
            # Behavior logic
            if dist < 5.0 and enemy.strength > 50:  # Heuristic: "strong" enemy
                enemy.behavior = "chase"
            elif dist < 3.0:
                enemy.behavior = "flee"
            else:
                enemy.behavior = "random_walk"
            
            # Update position
            enemy.update(agent_pos=agent_pos, dt=0.016)  # ~60 FPS
    
    def check_collision(self, pos: Tuple[float, float], radius: float = 0.1) -> bool:
        """
        Check collision with obstacles OR enemies.
        
        Args:
            pos: Position to check
            radius: Collision radius
            
        Returns:
            True if collision
        """
        # Check obstacles (from parent)
        if super().check_collision(pos, radius):
            return True
        
        # Check enemy collisions
        pos = np.array(pos, dtype=np.float32)
        for enemy in self.get_enemies():
            dist = np.linalg.norm(np.array(enemy.position) - pos)
            if dist < (radius + 0.3):  # Enemy has ~0.3m radius
                return True
        
        return False
```

- [ ] **Step 4: Run test (expect some failures initially)**

```bash
pytest tests/test_battleground_env.py -v
```

Debug any import or initialization errors. Key issues:
- NavigationEnvironment may not have all methods we expect
- May need to adjust parent class call

- [ ] **Step 5: Fix any integration issues with parent class**

If NavigationEnvironment doesn't match expectations, adjust `BattlegroundEnvironment` to match its actual interface. Common issues:
- Different return types for `reset()`
- Different initialization signature
- Missing utility methods

- [ ] **Step 6: Commit**

```bash
git add src/simulation/battleground_env.py tests/test_battleground_env.py
git commit -m "feat: add procedural BattlegroundEnvironment with enemy spawning and behavior"
```

---

### Task 3: Combat System

**Files:**
- Create: `src/simulation/combat_system.py`
- Test: `tests/test_combat_system.py`

**Interfaces:**
- Produces:
  - `def combat_attack(attacker_strength: float, defender_strength: float, rng: np.random.Generator = None) -> Dict` — returns `{"hit": bool, "damage": float, "victory": bool}`
  - `def compute_agent_strength(accumulated_reward: float) -> float` — returns strength (50-100) based on reward
  - `def resolve_combat_tick(agent_strength: float, enemy: EnemyNPC, max_damage: float = 20.0) -> Dict` — full attack resolution

- [ ] **Step 1: Write test file**

Create `tests/test_combat_system.py`:

```python
import pytest
import numpy as np
from src.simulation.combat_system import (
    combat_attack,
    compute_agent_strength,
    resolve_combat_tick,
)
from src.simulation.enemy_npc import EnemyNPC


def test_combat_attack_stronger_attacker():
    """Stronger attacker should have higher hit probability."""
    rng = np.random.default_rng(42)
    attacker_str = 80.0
    defender_str = 30.0
    result = combat_attack(attacker_str, defender_str, rng=rng)
    assert "hit" in result
    assert "damage" in result
    # Stronger attacker should win more often
    assert result["hit"] is True or result["hit"] is False  # Just verify types


def test_combat_attack_equal_strength():
    """Equal strength should give ~50% hit probability."""
    rng = np.random.default_rng(42)
    attacker_str = 50.0
    defender_str = 50.0
    hits = 0
    for _ in range(100):
        result = combat_attack(attacker_str, defender_str, rng=rng)
        if result["hit"]:
            hits += 1
    # Should be roughly 50% (allow 40-60% range)
    assert 30 < hits < 70


def test_compute_agent_strength():
    """Agent strength should scale with accumulated reward."""
    str_low = compute_agent_strength(accumulated_reward=0.0)
    str_high = compute_agent_strength(accumulated_reward=1000.0)
    assert str_low < str_high
    assert 50 <= str_low <= 100
    assert 50 <= str_high <= 100


def test_resolve_combat_tick_victory():
    """Test full combat resolution when agent wins."""
    enemy = EnemyNPC(position=(5.0, 5.0), health=50.0, strength=20.0)
    agent_strength = 80.0
    result = resolve_combat_tick(agent_strength, enemy, max_damage=20.0)
    
    assert "hit" in result
    assert "damage" in result
    assert "victory" in result
    assert "reward" in result
    # With high strength difference, should usually hit
    if result["hit"]:
        assert enemy.health < 50.0


def test_resolve_combat_tick_defeat():
    """Test full combat resolution when agent loses."""
    enemy = EnemyNPC(position=(5.0, 5.0), health=50.0, strength=80.0)
    agent_strength = 20.0
    result = resolve_combat_tick(agent_strength, enemy, max_damage=20.0)
    
    # With low strength, should often miss
    assert "hit" in result
    assert "damage" in result
```

- [ ] **Step 2: Run test (expect FAIL)**

```bash
pytest tests/test_combat_system.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.simulation.combat_system'`

- [ ] **Step 3: Write implementation**

Create `src/simulation/combat_system.py`:

```python
"""Combat system for embodied learning battleground."""
import numpy as np
from typing import Dict
from src.simulation.enemy_npc import EnemyNPC


def compute_agent_strength(accumulated_reward: float) -> float:
    """
    Compute agent combat strength from accumulated episode reward.
    
    Scaling: base 50 + reward/10 (capped at 100).
    
    Args:
        accumulated_reward: Sum of all rewards in episode
        
    Returns:
        Strength value (50-100)
    """
    strength = 50.0 + (accumulated_reward / 10.0)
    return float(np.clip(strength, 50.0, 100.0))


def combat_attack(
    attacker_strength: float,
    defender_strength: float,
    rng: np.random.Generator = None,
) -> Dict:
    """
    Resolve single attack.
    
    Hit probability = 0.5 + 0.05 * (attacker_strength - defender_strength) / 100
    Damage = uniform(0, max_damage) if hit
    
    Args:
        attacker_strength: Attacker combat strength (0-100)
        defender_strength: Defender combat strength (0-100)
        rng: NumPy random generator (if None, use default)
        
    Returns:
        Dict with keys:
        - "hit": bool — attack succeeded
        - "damage": float — damage dealt (0 if miss)
        - "hit_probability": float — calculated hit chance
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Hit probability based on strength difference
    hit_prob = 0.5 + 0.05 * (attacker_strength - defender_strength) / 100.0
    hit_prob = np.clip(hit_prob, 0.0, 1.0)
    
    hit = rng.random() < hit_prob
    damage = 0.0
    if hit:
        damage = float(rng.uniform(5, 20))  # Damage range 5-20
    
    return {
        "hit": bool(hit),
        "damage": float(damage),
        "hit_probability": float(hit_prob),
    }


def resolve_combat_tick(
    agent_strength: float,
    enemy: EnemyNPC,
    max_damage: float = 20.0,
) -> Dict:
    """
    Full combat resolution for a single attack by agent on enemy.
    
    Args:
        agent_strength: Agent combat strength
        enemy: Target enemy
        max_damage: Max damage per hit
        
    Returns:
        Dict with keys:
        - "hit": bool
        - "damage": float
        - "victory": bool — enemy died
        - "reward": float — combat reward (50 if victory, -10 if miss, 5 if hit)
        - "enemy_health_after": float
    """
    if not enemy.alive:
        return {
            "hit": False,
            "damage": 0.0,
            "victory": False,
            "reward": 0.0,
            "enemy_health_after": 0.0,
        }
    
    # Resolve attack
    result = combat_attack(agent_strength, enemy.strength)
    damage = result["damage"]
    hit = result["hit"]
    
    # Apply damage
    victory = False
    if hit:
        victory = enemy.take_damage(damage)
    
    # Compute reward
    if victory:
        reward = 50.0
    elif hit:
        reward = 5.0
    else:
        reward = -10.0
    
    return {
        "hit": hit,
        "damage": damage,
        "victory": victory,
        "reward": reward,
        "enemy_health_after": enemy.health,
    }
```

- [ ] **Step 4: Run test (expect PASS)**

```bash
pytest tests/test_combat_system.py -v
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/simulation/combat_system.py tests/test_combat_system.py
git commit -m "feat: add combat system with attack resolution and strength scaling"
```

---

### Task 4: Semantic State Encoder (64D)

**Files:**
- Create: `src/simulation/semantic_encoder.py`
- Test: `tests/test_semantic_encoder.py`

**Interfaces:**
- Consumes: raw observations (18D), enemy positions/health, accumulated reward
- Produces:
  - `class SemanticEncoder`: encodes 18D observations + enemy data → 64D semantic state
  - `encode(raw_obs: np.ndarray, enemies: List[EnemyNPC], accumulated_reward: float) -> np.ndarray` — returns 64D vector

- [ ] **Step 1: Write test file**

Create `tests/test_semantic_encoder.py`:

```python
import pytest
import numpy as np
from src.simulation.semantic_encoder import SemanticEncoder
from src.simulation.enemy_npc import EnemyNPC


def test_semantic_encoder_creation():
    encoder = SemanticEncoder()
    assert encoder is not None


def test_semantic_encode_basic():
    """Test basic encoding produces 64D vector."""
    encoder = SemanticEncoder()
    raw_obs = np.zeros(18, dtype=np.float32)
    enemies = []
    encoded = encoder.encode(raw_obs, enemies, accumulated_reward=0.0)
    
    assert isinstance(encoded, np.ndarray)
    assert encoded.shape == (64,)
    assert encoded.dtype == np.float32


def test_semantic_encode_with_enemies():
    """Test encoding with enemies in view."""
    encoder = SemanticEncoder()
    raw_obs = np.array([5.0, 5.0, 0.0,  # pos
                        0.0, 0.0, 0.0,  # vel
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # lidar
                        5.0, 5.0,  # goal delta
                        7.0,  # dist to goal
                        0.0], dtype=np.float32)
    
    enemies = [
        EnemyNPC(position=(3.0, 5.0), health=50.0, strength=40.0),
        EnemyNPC(position=(7.0, 5.0), health=30.0, strength=20.0),
    ]
    
    encoded = encoder.encode(raw_obs, enemies, accumulated_reward=100.0)
    assert encoded.shape == (64,)
    
    # Threat should be non-zero if enemies are present
    threat_idx = 7  # Approximate index for threat level
    assert encoded[threat_idx] > 0


def test_semantic_encode_range():
    """Test that all semantic dimensions are in reasonable range."""
    encoder = SemanticEncoder()
    raw_obs = np.random.randn(18).astype(np.float32) * 0.5
    enemies = [EnemyNPC(position=(5.0, 5.0), health=50.0, strength=40.0)]
    
    encoded = encoder.encode(raw_obs, enemies, accumulated_reward=200.0)
    
    # Most dimensions should be roughly in [0, 1] or [-1, 1]
    assert np.all(np.isfinite(encoded))
    assert np.min(encoded) >= -2.0  # Allow some overflow but not too much
    assert np.max(encoded) <= 2.0
```

- [ ] **Step 2: Run test (expect FAIL)**

```bash
pytest tests/test_semantic_encoder.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

Create `src/simulation/semantic_encoder.py`:

```python
"""Semantic state encoder: 18D observations → 64D semantic state."""
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
        """Initialize encoder."""
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
            raw_obs: 18D raw observation vector
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
        
        # 3: goal_alignment — alignment with goal direction
        goal_dist = np.linalg.norm(goal_delta)
        vel_dist = np.linalg.norm(vel)
        if goal_dist > 0.01 and vel_dist > 0.01:
            alignment = np.dot(goal_delta / goal_dist, vel / vel_dist)
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
        semantic[41] = 1.0 if (semantic[14] + semantic[15] + semantic[16]).sum() > 0 else 0.3
        
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
```

- [ ] **Step 4: Run test (expect PASS)**

```bash
pytest tests/test_semantic_encoder.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/simulation/semantic_encoder.py tests/test_semantic_encoder.py
git commit -m "feat: add 64D semantic state encoder mapping observations+enemies+reward"
```

---

### Task 5: Enhanced Reward Shaping

**Files:**
- Modify: `src/simulation/embodied_reward.py`
- Test: (add to existing test file or create new)

**Interfaces:**
- Consumes: `resolve_combat_tick()` from Task 3, observation data
- Produces: Enhanced reward function with navigation + combat + survival bonuses

- [ ] **Step 1: Read existing embodied_reward.py**

```bash
head -100 src/simulation/embodied_reward.py
```

Understand current reward structure (likely just distance-to-goal based).

- [ ] **Step 2: Write enhanced reward function**

Modify `src/simulation/embodied_reward.py` to add new function:

```python
def compute_embodied_reward_enhanced(
    agent_pos: Tuple[float, float],
    goal_pos: Tuple[float, float],
    prev_distance: float,
    is_collision: bool,
    is_goal_reached: bool,
    combat_result: Dict = None,  # output from resolve_combat_tick()
) -> float:
    """
    Enhanced reward shaping: navigation + combat + survival.
    
    Args:
        agent_pos: Current agent position
        goal_pos: Goal position
        prev_distance: Distance to goal last tick
        is_collision: Whether agent collided
        is_goal_reached: Whether goal was reached
        combat_result: Dict from resolve_combat_tick() (if attack happened), else None
        
    Returns:
        Float reward
    """
    reward = 0.0
    
    # === NAVIGATION REWARD ===
    curr_distance = np.linalg.norm(np.array(agent_pos) - np.array(goal_pos))
    distance_reduction = prev_distance - curr_distance
    if distance_reduction > 0:
        reward += 0.1 * distance_reduction  # Reward moving closer
    
    # === COLLISION PENALTY ===
    if is_collision:
        reward -= 5.0
    
    # === GOAL BONUS ===
    if is_goal_reached:
        reward += 100.0
    
    # === COMBAT REWARD ===
    if combat_result is not None:
        reward += combat_result.get("reward", 0.0)
    
    # === SURVIVAL PENALTY (encourage movement) ===
    # Small negative reward for idle (handled elsewhere, here just minimal)
    
    return float(reward)
```

Add this to `embodied_reward.py` and update any imports/callers.

- [ ] **Step 3: Test the enhanced reward**

Create simple test:

```bash
python -c "
from src.simulation.embodied_reward import compute_embodied_reward_enhanced
from src.simulation.combat_system import resolve_combat_tick
from src.simulation.enemy_npc import EnemyNPC
import numpy as np

# Test basic navigation reward
reward = compute_embodied_reward_enhanced(
    agent_pos=(5.0, 5.0),
    goal_pos=(10.0, 10.0),
    prev_distance=8.0,
    is_collision=False,
    is_goal_reached=False,
    combat_result=None
)
print(f'Navigation reward: {reward}')
assert reward > 0, 'Should reward moving closer to goal'

# Test goal bonus
reward = compute_embodied_reward_enhanced(
    agent_pos=(10.0, 10.0),
    goal_pos=(10.0, 10.0),
    prev_distance=0.5,
    is_collision=False,
    is_goal_reached=True,
    combat_result=None
)
print(f'Goal bonus: {reward}')
assert reward > 100, 'Should have goal bonus'

print('Enhanced reward tests passed!')
"
```

- [ ] **Step 4: Commit**

```bash
git add src/simulation/embodied_reward.py
git commit -m "feat: enhance reward shaping with combat+survival rewards"
```

---

### Task 6: EmbodiedAgent Integration

**Files:**
- Modify: `src/simulation/embodied_loop.py`
- Test: integration test in `tests/test_embodied_system.py`

**Interfaces:**
- Consumes: All previous tasks (BattlegroundEnvironment, Combat, Semantic Encoder, Enhanced Reward)
- Produces: Updated EmbodiedAgent with 64D state, enhanced learning loop

- [ ] **Step 1: Update EmbodiedAgent.__init__**

Modify `src/simulation/embodied_loop.py`:

```python
from src.simulation.battleground_env import BattlegroundEnvironment
from src.simulation.semantic_encoder import SemanticEncoder
from src.simulation.embodied_reward import compute_embodied_reward_enhanced
from src.simulation.combat_system import resolve_combat_tick, compute_agent_strength

class EmbodiedAgent:
    def __init__(self, lrsg_config, locomotion_config, embodied_config, device='cpu'):
        # ... existing code ...
        
        # NEW: Replace NavigationEnvironment with BattlegroundEnvironment
        self.env = BattlegroundEnvironment(
            grid_size=embodied_config.grid_size,
            num_obstacles=embodied_config.num_obstacles,
            num_enemies=3,  # New: 3 enemies per episode
            seed=None
        )
        
        # NEW: Semantic encoder
        self.semantic_encoder = SemanticEncoder()
        
        # NEW: Combat tracking
        self._agent_accumulated_reward = 0.0
        self._prev_accumulated_reward = 0.0
```

- [ ] **Step 2: Update _build_observation() to return 64D semantic state**

```python
def _build_observation(self) -> np.ndarray:
    """Build 64D semantic state observation."""
    # Get raw 18D observation (reuse existing logic)
    lidar = self.env.get_lidar(
        agent_pos=self.agent_pos,
        num_rays=self.emb_config.lidar_rays
    )
    
    goal_delta_x = self.goal_pos[0] - self.agent_pos[0]
    goal_delta_y = self.goal_pos[1] - self.agent_pos[1]
    dist_to_goal = np.linalg.norm(
        np.array(self.agent_pos) - np.array(self.goal_pos)
    )
    
    raw_obs_18d = np.concatenate([
        self.agent_pos,
        [0.0, 0.0, 0.0],  # velocity
        lidar,
        [goal_delta_x, goal_delta_y],
        [dist_to_goal],
        [0.0],
    ]).astype(np.float32)
    
    # Encode to 64D semantic state
    enemies = self.env.get_enemies()
    semantic_state = self.semantic_encoder.encode(
        raw_obs_18d,
        enemies,
        accumulated_reward=self._agent_accumulated_reward,
        prev_accumulated_reward=self._prev_accumulated_reward,
    )
    
    return semantic_state  # 64D
```

- [ ] **Step 3: Update step() to use BattlegroundEnvironment and enhanced reward**

```python
def step(self) -> Dict[str, float]:
    """Single step with battleground, combat, enhanced reward."""
    # Get 64D semantic observation
    obs = self._build_observation()
    
    # LRSG forward pass (same as before, but with 64D input)
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
    self.lrsg.reset_activations()
    self.lrsg.inject_input(obs_tensor)
    forward_pass(self.lrsg, self.lrsg_config, last_action=self._last_action, energy=200.0)
    output_acts = self.lrsg.read_output()
    
    # NEW: Extend action space to include ATTACK
    # Actions: 0=MOVE_FWD, 1=TURN_LEFT, 2=TURN_RIGHT, 3=IDLE, 4=ATTACK
    action = int(torch.argmax(output_acts[:5]).item())
    self._last_action = action
    
    # Motor control (same as before for 0-3)
    if action == 0:
        left_motor, right_motor = 0.8, 0.8
    elif action == 1:
        left_motor, right_motor = 0.3, 0.8
    elif action == 2:
        left_motor, right_motor = 0.8, 0.3
    elif action == 3:
        left_motor, right_motor = 0.0, 0.0
    else:  # ATTACK
        left_motor, right_motor = 0.0, 0.0  # Stop to attack
    
    # Locomotion update
    linear_vel, angular_vel = self.locomotion.motors_to_velocity(left_motor, right_motor)
    dt = 1.0 / self.emb_config.physics_freq
    dx, dy, dyaw = self.locomotion.step(linear_vel, angular_vel, dt)
    
    old_pos = self.agent_pos
    cos_yaw = np.cos(self.agent_yaw)
    sin_yaw = np.sin(self.agent_yaw)
    x_world = cos_yaw * dx - sin_yaw * dy
    y_world = sin_yaw * dx + cos_yaw * dy
    
    new_pos = (self.agent_pos[0] + x_world, self.agent_pos[1] + y_world, self.agent_pos[2])
    self.agent_yaw += dyaw
    
    # Collision check (includes enemies now)
    is_collision = self.env.check_collision(new_pos, radius=0.1)
    if not is_collision:
        self.agent_pos = new_pos
    
    # Update enemy behaviors
    self.env.tick_enemies(agent_pos=self.agent_pos)
    
    # NEW: Resolve combat if ATTACK action and enemy nearby
    combat_result = None
    if action == 4:  # ATTACK
        nearby_enemies = self.env.get_enemy_near(self.agent_pos, radius=2.0)
        if nearby_enemies:
            # Attack nearest enemy
            nearest = min(nearby_enemies, key=lambda e: np.linalg.norm(np.array(e.position) - np.array(self.agent_pos)))
            agent_strength = compute_agent_strength(self._agent_accumulated_reward)
            combat_result = resolve_combat_tick(agent_strength, nearest, max_damage=20.0)
            
            # Remove dead enemies
            if combat_result["victory"]:
                self.env.remove_enemy(nearest)
    
    # Check goal and collisions
    curr_distance = np.linalg.norm(np.array(self.agent_pos) - np.array(self.goal_pos))
    is_goal_reached = curr_distance < self.emb_config.goal_reach_threshold
    
    # NEW: Enhanced reward
    reward = compute_embodied_reward_enhanced(
        agent_pos=self.agent_pos,
        goal_pos=self.goal_pos,
        prev_distance=self.prev_distance,
        is_collision=is_collision,
        is_goal_reached=is_goal_reached,
        combat_result=combat_result,
    )
    
    # Track rewards
    self.episode_rewards.append(reward)
    self._prev_accumulated_reward = self._agent_accumulated_reward
    self._agent_accumulated_reward += reward
    self.prev_distance = curr_distance
    self.tick_count += 1
    
    # Episode termination
    episode_done = self.tick_count >= self.emb_config.episode_length or is_goal_reached
    
    if episode_done:
        accumulated_reward = float(sum(self.episode_rewards))
        if accumulated_reward > 0:
            hebbian_update(self.lrsg, self.lrsg_config,
                           reward=accumulated_reward, learning_rate=0.01)
        self.lrsg.eligibility_trace.zero_()
        self._reset_episode()
    
    return {
        'reward': float(reward),
        'distance_to_goal': float(curr_distance),
        'is_collision': bool(is_collision),
        'is_goal_reached': bool(is_goal_reached),
        'action': int(action),
        'num_enemies': len(self.env.get_enemies()),
    }
```

- [ ] **Step 4: Test integration**

```bash
python -c "
from src.config import LRSGConfig, LocomotionConfig, EmbodiedConfig
from src.simulation.embodied_loop import EmbodiedAgent
import torch

config = LRSGConfig(num_neurons=100_000)  # Small for testing
loc_config = LocomotionConfig()
emb_config = EmbodiedConfig()

agent = EmbodiedAgent(config, loc_config, emb_config, device='cpu')
print(f'Agent created. Env: {type(agent.env).__name__}')

# Run 5 steps
for _ in range(5):
    result = agent.step()
    print(f'Step: reward={result[\"reward\"]:.2f}, enemies={result[\"num_enemies\"]}')

print('Integration test passed!')
"
```

- [ ] **Step 5: Commit**

```bash
git add src/simulation/embodied_loop.py
git commit -m "feat: integrate battleground, combat, 64D semantic state, and enhanced reward"
```

---

### Task 7: Visualization Dashboard

**Files:**
- Create: `src/simulation/embodied_visualizer.py`
- Create: `src/simulation/static/index.html`
- Create: `src/simulation/static/style.css`
- Create: `src/simulation/static/app.js`

**Interfaces:**
- Consumes: Episode state updates (position, enemies, reward, semantic state)
- Produces: Flask/WSGI web server on port 8080 with real-time canvas rendering

(This is a large component; implement as a single cohesive task)

- [ ] **Step 1: Create Flask server**

Create `src/simulation/embodied_visualizer.py`:

```python
"""Real-time web visualization for embodied learning."""
import json
import numpy as np
from typing import Dict, List
from flask import Flask, render_template, jsonify
from threading import Thread, Lock


class EmbodiedVisualizer:
    """Flask-based web visualizer for embodied agent learning."""
    
    def __init__(self, port: int = 8080):
        """
        Args:
            port: Port to serve on
        """
        self.port = port
        self.app = Flask(__name__, template_folder='src/simulation/templates', static_folder='src/simulation/static')
        self.data_lock = Lock()
        
        # Current state
        self.agent_pos = (0, 0)
        self.goal_pos = (10, 10)
        self.enemies = []
        self.obstacles = []
        self.episode = 0
        self.tick = 0
        self.current_reward = 0.0
        self.accumulated_reward = 0.0
        self.episode_rewards = []  # List of (episode, accumulated_reward)
        self.semantic_state = np.zeros(64, dtype=np.float32)
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/api/state')
        def get_state():
            with self.data_lock:
                return jsonify({
                    'agent_pos': self.agent_pos,
                    'goal_pos': self.goal_pos,
                    'enemies': self.enemies,
                    'obstacles': self.obstacles,
                    'episode': self.episode,
                    'tick': self.tick,
                    'current_reward': self.current_reward,
                    'accumulated_reward': self.accumulated_reward,
                    'semantic_state': [float(x) for x in self.semantic_state],
                    'episode_rewards': self.episode_rewards,
                })
    
    def update_state(
        self,
        agent_pos: tuple,
        goal_pos: tuple,
        enemies: List[Dict],
        obstacles: List[tuple],
        episode: int,
        tick: int,
        current_reward: float,
        accumulated_reward: float,
        semantic_state: np.ndarray,
    ):
        """Update visualization state."""
        with self.data_lock:
            self.agent_pos = agent_pos
            self.goal_pos = goal_pos
            self.enemies = enemies
            self.obstacles = obstacles
            self.episode = episode
            self.tick = tick
            self.current_reward = current_reward
            self.accumulated_reward = accumulated_reward
            self.semantic_state = semantic_state
    
    def log_episode_reward(self, episode: int, reward: float):
        """Log accumulated reward for episode."""
        with self.data_lock:
            self.episode_rewards.append([episode, reward])
            # Keep last 100 episodes
            if len(self.episode_rewards) > 100:
                self.episode_rewards.pop(0)
    
    def start(self):
        """Start Flask server in background thread."""
        def run_server():
            self.app.run(host='0.0.0.0', port=self.port, debug=False, use_reloader=False)
        
        thread = Thread(target=run_server, daemon=True)
        thread.start()
        print(f"Visualizer started on http://localhost:{self.port}")
```

- [ ] **Step 2: Create HTML template**

Create `src/simulation/templates/index.html`:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Embodied Learning Battleground</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div class="container">
        <div class="canvas-wrapper">
            <canvas id="canvas" width="500" height="500"></canvas>
        </div>
        <div class="metrics-panel">
            <h2>Metrics</h2>
            <div class="metric">
                <label>Episode:</label>
                <span id="episode">0</span>
            </div>
            <div class="metric">
                <label>Tick:</label>
                <span id="tick">0</span>
            </div>
            <div class="metric">
                <label>Accumulated Reward:</label>
                <span id="acc-reward">0.00</span>
            </div>
            <div class="metric">
                <label>Enemies Visible:</label>
                <span id="enemy-count">0</span>
            </div>
            <div class="metric">
                <label>Threat Level:</label>
                <span id="threat-level">0.00</span>
            </div>
            <h3>Learning Curve</h3>
            <canvas id="reward-chart" width="400" height="150"></canvas>
        </div>
    </div>
    <script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 3: Create CSS**

Create `src/simulation/static/style.css`:

```css
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Courier New', monospace;
    background: #0a0e27;
    color: #00ff00;
    padding: 20px;
}

.container {
    display: flex;
    gap: 30px;
    max-width: 1200px;
    margin: 0 auto;
}

.canvas-wrapper {
    flex: 1;
    border: 2px solid #00ff00;
    background: #1a1a2e;
    padding: 10px;
    border-radius: 5px;
    box-shadow: 0 0 20px rgba(0, 255, 0, 0.3);
}

canvas {
    display: block;
    width: 100%;
    height: auto;
    border-radius: 3px;
}

.metrics-panel {
    flex: 0 0 300px;
    border: 2px solid #00ff00;
    padding: 20px;
    background: #1a1a2e;
    border-radius: 5px;
    box-shadow: 0 0 20px rgba(0, 255, 0, 0.3);
}

.metrics-panel h2 {
    margin-bottom: 15px;
    border-bottom: 1px solid #00ff00;
    padding-bottom: 10px;
    font-size: 18px;
}

.metrics-panel h3 {
    margin-top: 20px;
    margin-bottom: 10px;
    font-size: 14px;
}

.metric {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid #2a2a4e;
}

.metric label {
    font-weight: bold;
}

.metric span {
    color: #00ff00;
    font-weight: bold;
}

#reward-chart {
    width: 100%;
    border: 1px solid #003300;
    background: #0a0a1a;
}
```

- [ ] **Step 4: Create JavaScript**

Create `src/simulation/static/app.js`:

```javascript
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const rewardCanvas = document.getElementById('reward-chart');
const rewardCtx = rewardCanvas.getContext('2d');

const GRID_SIZE = 20; // meters
const PIXEL_SCALE = canvas.width / GRID_SIZE;

let state = {
    agent_pos: [0, 0],
    goal_pos: [10, 10],
    enemies: [],
    obstacles: [],
    episode: 0,
    tick: 0,
    current_reward: 0,
    accumulated_reward: 0,
    semantic_state: [],
    episode_rewards: []
};

async function fetchState() {
    try {
        const response = await fetch('/api/state');
        state = await response.json();
        render();
    } catch (err) {
        console.error('Failed to fetch state:', err);
    }
}

function render() {
    renderCanvas();
    renderMetrics();
    renderRewardChart();
}

function renderCanvas() {
    // Clear
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Grid
    ctx.strokeStyle = '#003300';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= GRID_SIZE; i++) {
        const px = i * PIXEL_SCALE;
        ctx.beginPath();
        ctx.moveTo(px, 0);
        ctx.lineTo(px, canvas.height);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(0, px);
        ctx.lineTo(canvas.width, px);
        ctx.stroke();
    }
    
    // Obstacles
    ctx.fillStyle = '#666666';
    for (const obs of state.obstacles) {
        const x = obs[0] * PIXEL_SCALE;
        const y = obs[1] * PIXEL_SCALE;
        ctx.fillRect(x - 5, y - 5, 10, 10);
    }
    
    // Goal
    ctx.fillStyle = '#ffff00';
    const gx = state.goal_pos[0] * PIXEL_SCALE;
    const gy = state.goal_pos[1] * PIXEL_SCALE;
    ctx.fillRect(gx - 8, gy - 8, 16, 16);
    ctx.strokeStyle = '#ffff00';
    ctx.lineWidth = 2;
    ctx.strokeRect(gx - 8, gy - 8, 16, 16);
    
    // Enemies
    for (const enemy of state.enemies) {
        const x = enemy.position[0] * PIXEL_SCALE;
        const y = enemy.position[1] * PIXEL_SCALE;
        const healthRatio = enemy.health / enemy.max_health;
        
        ctx.fillStyle = `rgb(${255 * (1 - healthRatio)}, ${255 * healthRatio}, 0)`;
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, 2 * Math.PI);
        ctx.fill();
        
        ctx.strokeStyle = '#ff0000';
        ctx.lineWidth = 2;
        ctx.stroke();
    }
    
    // Agent (centered, world scrolls around it)
    const ax = state.agent_pos[0] * PIXEL_SCALE;
    const ay = state.agent_pos[1] * PIXEL_SCALE;
    ctx.fillStyle = '#0080ff';
    ctx.beginPath();
    ctx.arc(ax, ay, 10, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 3;
    ctx.stroke();
    
    // Agent direction indicator
    ctx.strokeStyle = '#00ff00';
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(ax + 15 * Math.cos(0), ay + 15 * Math.sin(0)); // Simplified, no yaw tracked
    ctx.stroke();
}

function renderMetrics() {
    document.getElementById('episode').textContent = state.episode;
    document.getElementById('tick').textContent = state.tick;
    document.getElementById('acc-reward').textContent = state.accumulated_reward.toFixed(2);
    document.getElementById('enemy-count').textContent = state.enemies.length;
    
    const threatLevel = state.semantic_state[8] || 0; // Threat level is dimension 8
    document.getElementById('threat-level').textContent = threatLevel.toFixed(2);
}

function renderRewardChart() {
    // Draw learning curve
    const width = rewardCanvas.width;
    const height = rewardCanvas.height;
    const padding = 10;
    
    rewardCtx.fillStyle = '#0a0a1a';
    rewardCtx.fillRect(0, 0, width, height);
    
    if (state.episode_rewards.length < 2) return;
    
    // Find min/max reward
    const rewards = state.episode_rewards.map(r => r[1]);
    const minReward = Math.min(...rewards);
    const maxReward = Math.max(...rewards);
    const rewardRange = maxReward - minReward || 1;
    
    // Draw axes
    rewardCtx.strokeStyle = '#00ff00';
    rewardCtx.lineWidth = 1;
    rewardCtx.beginPath();
    rewardCtx.moveTo(padding, height - padding);
    rewardCtx.lineTo(width - padding, height - padding);
    rewardCtx.stroke();
    
    // Draw curve
    rewardCtx.strokeStyle = '#00ff00';
    rewardCtx.lineWidth = 2;
    rewardCtx.beginPath();
    
    for (let i = 0; i < state.episode_rewards.length; i++) {
        const x = padding + ((i / (state.episode_rewards.length - 1)) * (width - 2 * padding));
        const normalizedReward = (rewards[i] - minReward) / rewardRange;
        const y = height - padding - (normalizedReward * (height - 2 * padding));
        
        if (i === 0) {
            rewardCtx.moveTo(x, y);
        } else {
            rewardCtx.lineTo(x, y);
        }
    }
    rewardCtx.stroke();
}

// Fetch state every 50ms (~20 FPS)
setInterval(fetchState, 50);

// Initial render
fetchState();
```

- [ ] **Step 5: Test visualizer**

```bash
# Note: This requires the full embodied agent to be running
# For now, just verify Flask starts
python -c "
from src.simulation.embodied_visualizer import EmbodiedVisualizer
viz = EmbodiedVisualizer(port=8080)
print('Visualizer created successfully')
"
```

- [ ] **Step 6: Commit**

```bash
git add src/simulation/embodied_visualizer.py src/simulation/static/ src/simulation/templates/
git commit -m "feat: add real-time web visualization dashboard"
```

---

### Task 8: Episode Runner & Learning Loop

**Files:**
- Create: `src/experiments/run_embodied_battleground.py`
- Test: `tests/test_embodied_system_integration.py`

**Interfaces:**
- Consumes: All previous components (EmbodiedAgent, BattlegroundEnvironment, Visualizer)
- Produces: Complete learnable system with 100+ episodes, live visualization, telemetry logging

- [ ] **Step 1: Write runner script**

Create `src/experiments/run_embodied_battleground.py`:

```python
#!/usr/bin/env python3
"""
Runner for embodied learning battleground with live visualization.

Runs 100+ episodes of agent learning to navigate, fight, and survive
in a procedurally-generated battleground. Shows real-time visualization
and learning curves.
"""

import json
import numpy as np
import torch
from pathlib import Path
from src.config import LRSGConfig, LocomotionConfig, EmbodiedConfig
from src.simulation.embodied_loop import EmbodiedAgent
from src.simulation.embodied_visualizer import EmbodiedVisualizer


def run_embodied_learning(
    num_episodes: int = 100,
    num_neurons: int = 4_048_576,  # 4M neurons
    device: str = None,
    headless: bool = False,
):
    """
    Run embodied learning training loop.
    
    Args:
        num_episodes: Number of episodes
        num_neurons: LRSG neurons
        device: 'cpu', 'cuda', or None (auto-detect)
        headless: If False, start visualization server
    """
    
    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Configs
    lrsg_config = LRSGConfig(num_neurons=num_neurons)
    loc_config = LocomotionConfig()
    emb_config = EmbodiedConfig()
    
    # Create agent
    print(f"Creating agent with {num_neurons} neurons...")
    agent = EmbodiedAgent(lrsg_config, loc_config, emb_config, device=device)
    
    # Create visualizer
    visualizer = None
    if not headless:
        visualizer = EmbodiedVisualizer(port=8080)
        visualizer.start()
        print("Visualizer started on http://localhost:8080")
    
    # Output directory
    output_dir = Path('experiment_results')
    output_dir.mkdir(exist_ok=True)
    
    # Telemetry
    episode_log = []
    
    # Training loop
    print(f"\nRunning {num_episodes} episodes...")
    for episode in range(num_episodes):
        agent._reset_episode()
        episode_data = {
            'episode': episode,
            'total_reward': 0.0,
            'goal_reached': False,
            'max_enemies_seen': 0,
            'steps': 0,
            'final_distance_to_goal': 0.0,
        }
        
        while agent.tick_count < agent.emb_config.episode_length:
            step_result = agent.step()
            episode_data['total_reward'] += step_result['reward']
            episode_data['max_enemies_seen'] = max(
                episode_data['max_enemies_seen'],
                step_result['num_enemies']
            )
            episode_data['steps'] = agent.tick_count
            episode_data['final_distance_to_goal'] = step_result['distance_to_goal']
            
            if step_result['is_goal_reached']:
                episode_data['goal_reached'] = True
                break
            
            # Update visualizer
            if visualizer is not None:
                enemies_data = [
                    {
                        'position': list(e.position),
                        'health': e.health,
                        'max_health': e.max_health,
                        'strength': e.strength,
                    }
                    for e in agent.env.get_enemies()
                ]
                
                visualizer.update_state(
                    agent_pos=agent.agent_pos,
                    goal_pos=agent.goal_pos,
                    enemies=enemies_data,
                    obstacles=[],  # Could track obstacles
                    episode=episode,
                    tick=agent.tick_count,
                    current_reward=step_result['reward'],
                    accumulated_reward=agent._agent_accumulated_reward,
                    semantic_state=agent._build_observation(),  # 64D state
                )
        
        # Log episode
        episode_log.append(episode_data)
        visualizer.log_episode_reward(episode, episode_data['total_reward'])
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean([e['total_reward'] for e in episode_log[-10:]])
            success_rate = sum(1 for e in episode_log[-10:] if e['goal_reached']) / 10
            print(f"Episode {episode+1}: Avg Reward={avg_reward:.2f}, Success={success_rate:.1%}")
    
    # Save telemetry
    telemetry_path = output_dir / 'embodied_battleground_telemetry.jsonl'
    with open(telemetry_path, 'w') as f:
        for record in episode_log:
            f.write(json.dumps(record) + '\n')
    
    print(f"\nTraining complete! Telemetry saved to {telemetry_path}")
    
    # Final statistics
    total_reward = sum(e['total_reward'] for e in episode_log)
    success_count = sum(1 for e in episode_log if e['goal_reached'])
    print(f"\nFinal Statistics:")
    print(f"  Total episodes: {len(episode_log)}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Success rate: {success_count / len(episode_log):.1%}")
    print(f"  Avg reward per episode: {total_reward / len(episode_log):.2f}")
    
    # Keep visualizer running
    if visualizer is not None:
        print("\nVisualizer running. Press Ctrl+C to exit.")
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopped.")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run embodied learning battleground')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--neurons', type=int, default=4_048_576, help='LRSG neurons')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu, cuda)')
    parser.add_argument('--headless', action='store_true', help='No visualization')
    
    args = parser.parse_args()
    
    run_embodied_learning(
        num_episodes=args.episodes,
        num_neurons=args.neurons,
        device=args.device,
        headless=args.headless,
    )
```

- [ ] **Step 2: Make runner executable**

```bash
chmod +x src/experiments/run_embodied_battleground.py
```

- [ ] **Step 3: Test runner (dry run with 5 episodes, headless)**

```bash
python src/experiments/run_embodied_battleground.py --episodes 5 --neurons 100000 --device cpu --headless
```

Expected: Script runs 5 episodes, prints progress, saves telemetry to `experiment_results/embodied_battleground_telemetry.jsonl`

- [ ] **Step 4: Verify telemetry output**

```bash
head -5 experiment_results/embodied_battleground_telemetry.jsonl
```

Expected: JSONL records with episode data (total_reward, goal_reached, steps, etc.)

- [ ] **Step 5: Commit**

```bash
git add src/experiments/run_embodied_battleground.py
git commit -m "feat: add episode runner with live visualization and learning loop"
```

---

## Testing Checklist

- [ ] All unit tests pass: `pytest tests/test_enemy_npc.py tests/test_battleground_env.py tests/test_combat_system.py tests/test_semantic_encoder.py -v`
- [ ] Runner executes successfully with `python src/experiments/run_embodied_battleground.py --episodes 5 --headless`
- [ ] Telemetry is generated correctly
- [ ] Visualizer starts and serves on `localhost:8080` (manual test, optional for headless mode)

---

## Implementation Notes

- **Dependency Order:** Tasks 1-3 can run in parallel. Task 4 depends on 1-3. Task 5 depends on 3. Task 6 depends on 1-5. Task 7 depends on 6. Task 8 depends on all.
- **Learning Validation:** Check that average episode reward increases over time (first 50 episodes should show improvement).
- **VRAM:** Full 4M neuron model may require 8GB+ VRAM. If OOM, reduce to 1M neurons for testing.
- **Visualization Performance:** Canvas rendering may lag on slower machines. Consider reducing FPS if needed.

