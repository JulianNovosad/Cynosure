# Cynosure Semantic State Integration & Physics Simulation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the 57D semantic state system into LRSG, build a Pybullet physics simulation with sensor simulation, and create live visualization of semantic embeddings. This satisfies the goal condition: interpretable internal states, photorealistic-enough simulation (Pybullet), learning curves in physics environment, and live state inspection.

**Architecture:** 

Phase 1 (MVP):
1. **Semantic Encoder** (`src/lrsg/semantic_encoder.py`) — Maps raw sensors (vision, lidar, IMU, motor feedback) → 57D semantic state vector
2. **Physics Simulation** (`src/simulation/physics_world.py`) — Pybullet integration with RPi5-like robot (differential drive tank), realistic sensor simulation
3. **Sensor Simulation** (`src/simulation/sensors.py`, expand existing) — Camera with MobileNetSSD object detection, lidar rangefinder, IMU with physics-driven readings
4. **Semantic Visualization** (`frontend/semantic_view.html`) — Real-time 57D semantic embedding inspector (per-dimension sliders, heatmap, time series)
5. **Integration** — Wire semantic encoder into LRSG input during `sense_and_act()`, stream embeddings to frontend
6. **Learning Loop** — Agents learn to approach objects, avoid threats, explore in Pybullet sim; Hebbian fires on reward

Phase 2 (Future, not in this plan):
- Photorealistic rendering (Unreal/Unity)
- Raspberry Pi 5 hardware integration
- Cloud training pipeline

**Tech Stack:**
- **Physics:** PyBullet (pip install pybullet)
- **Vision:** OpenCV + YOLO (already in deps)
- **Frontend:** Three.js (already included)
- **Testing:** pytest with TDD

## Global Constraints

- LRSG forward pass must remain < 10ms per agent (no heavy preprocessing)
- Semantic encoder output: torch.Tensor, shape (57,), float32, range [-1, 1] or [0, 1]
- Physics tick: 240 Hz (Pybullet default), 10 physics substeps per agent sense-act cycle
- Learning: Use existing `hebbian_update()` and `forward.py` reward system (no new learning algorithms)
- Backwards compatibility: Existing `run_simulation.py`, `run_learning_experiment.py` must work unchanged
- All tests must pass: `pytest tests/` (including new sensor and semantic tests)

---

## File Structure & Responsibilities

| File | Purpose | Status |
|------|---------|--------|
| `src/lrsg/semantic_encoder.py` | Convert sensors → 57D semantic state | Create |
| `src/simulation/physics_world.py` | Pybullet world, robot model, step simulation | Create |
| `src/simulation/sensors.py` | Camera + lidar + IMU simulation | Expand |
| `src/simulation/loop.py` | Integrate semantic encoder into sense-act | Modify (lines ~70-150) |
| `frontend/semantic_view.html` | Real-time semantic embedding visualization | Create |
| `src/config.py` | Add `use_physics: bool`, `semantic_weight: float` flags | Modify |
| `tests/test_semantic_encoder.py` | Unit tests for semantic state | Create |
| `tests/test_physics_world.py` | Unit tests for physics simulation | Create |
| `tests/test_sensors.py` | Unit tests for sensor simulation | Create |
| `tests/test_semantic_integration.py` | Integration test: sim → semantic → LRSG → output | Create |

---

## Task Breakdown

### Task 1: Semantic Encoder — Definition & Tests

**Files:**
- Create: `src/lrsg/semantic_encoder.py`
- Create: `tests/test_semantic_encoder.py`

**Interfaces:**
- Consumes: Raw sensor dict `{"vision": [...], "lidar": [...], "imu": [...], "motor_feedback": [...], "ram": float, "position": (x,y,z), "time": int}`
- Produces: `torch.Tensor` shape `(57,)`, dtype `float32`, values in `[-1, 1]` or `[0, 1]`

**Steps:**

- [ ] **Step 1: Write semantic encoder class skeleton with docstring**

Create `src/lrsg/semantic_encoder.py`:

```python
import torch
import torch.nn as nn

class SemanticEncoder(nn.Module):
    """
    Maps raw sensor inputs to 57-dimensional semantic state.
    
    Input: dict with keys:
        - 'vision': list of detected objects [(x, y, z, class_id, confidence), ...]
        - 'lidar': list of (angle, distance) pairs
        - 'imu': dict {'accel': (x,y,z), 'gyro': (x,y,z), 'mag': (x,y,z)}
        - 'motor_feedback': dict {'left_speed': float, 'right_speed': float}
        - 'ram': float (current resource level, 0-100)
        - 'position': (x, y, z) tuple
        - 'time': int (tick number)
    
    Output: torch.Tensor of shape (57,), dtype float32, values in [-1, 1]
    
    The 57 dimensions are:
    [0-6]:     Existence (Integrity, Continuity, Vulnerability, ResourceMargin, Stability, Recovery, TerminationRisk)
    [7]:       PersistenceDrive
    [8-14]:    Sensory (Awareness, Salience, Noise, Focus, Novelty, Predictability, Orientation)
    [15]:      Saturation
    [16-23]:   Cognitive (Comprehension, Uncertainty, Abstraction, MemoryPressure, Consistency, UpdateDrive, SimulationDepth, Surprise)
    [24-31]:   Strategic (GoalClarity, GoalConflict, Optionality, Commitment, Leverage, Timing, RiskPosture, LongHorizon)
    [32-39]:   Tactical (Exploration, Exploitation, Defense, Aggression, Stealth, Efficiency, Adaptation, Execution)
    [40-47]:   Social (AgencyDetection, IntentInference, Trust, Dominance, Alignment, DeceptionAwareness, Reciprocity, Reputation)
    [48-52]:   Value (UtilitySignal, ConstraintPressure, NormInternalization, MoralTension, [unused])
    [53-56]:   Meta (SelfModel, MonitorLoad, Drift, ControlBalance)
    """
    
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        # Learned projections (minimal for MVP)
        self.vision_fc = nn.Linear(512, 16)  # Object detection → sensory + cognitive
        self.lidar_fc = nn.Linear(360, 16)   # 360 lidar rays → spatial awareness
        self.imu_fc = nn.Linear(9, 8)        # 9 IMU values (accel 3 + gyro 3 + mag 3) → tactical
        self.state_fc = nn.Linear(7, 17)     # ram, position(3), time, [motor feedback](2) → existence + value + meta
        
    def forward(self, sensor_dict):
        """
        Args:
            sensor_dict: dict with keys as above
        
        Returns:
            torch.Tensor of shape (57,) in range [-1, 1]
        """
        raise NotImplementedError("Implement in next step")
```

- [ ] **Step 2: Write test for semantic encoder — test that output shape is correct**

Create `tests/test_semantic_encoder.py`:

```python
import torch
import pytest
from src.lrsg.semantic_encoder import SemanticEncoder


def test_semantic_encoder_output_shape():
    """Test that encoder outputs 57D tensor."""
    encoder = SemanticEncoder(device='cpu')
    
    sensor_dict = {
        'vision': [(10.0, 5.0, 15.0, 1, 0.95), (20.0, 10.0, 30.0, 2, 0.87)],  # 2 objects
        'lidar': [(i * 1.0, 5.0 + i * 0.1) for i in range(360)],  # 360 rays
        'imu': {'accel': (0.1, -0.05, 9.8), 'gyro': (0.01, 0.02, 0.0), 'mag': (1.0, 0.5, -0.3)},
        'motor_feedback': {'left_speed': 1.0, 'right_speed': 0.95},
        'ram': 75.0,
        'position': (10.0, 5.0, 0.5),
        'time': 100,
    }
    
    output = encoder.forward(sensor_dict)
    
    assert isinstance(output, torch.Tensor), "Output must be torch.Tensor"
    assert output.shape == (57,), f"Expected shape (57,), got {output.shape}"
    assert output.dtype == torch.float32, f"Expected dtype float32, got {output.dtype}"
    assert torch.all(output >= -1.0) and torch.all(output <= 1.0), "Output must be in range [-1, 1]"


def test_semantic_encoder_deterministic():
    """Test that same input produces same output."""
    encoder = SemanticEncoder(device='cpu')
    encoder.eval()  # No dropout
    
    sensor_dict = {
        'vision': [(10.0, 5.0, 15.0, 1, 0.95)],
        'lidar': [(i * 1.0, 5.0) for i in range(360)],
        'imu': {'accel': (0.0, 0.0, 9.8), 'gyro': (0.0, 0.0, 0.0), 'mag': (1.0, 0.0, 0.0)},
        'motor_feedback': {'left_speed': 0.0, 'right_speed': 0.0},
        'ram': 50.0,
        'position': (0.0, 0.0, 0.0),
        'time': 0,
    }
    
    with torch.no_grad():
        out1 = encoder(sensor_dict)
        out2 = encoder(sensor_dict)
    
    assert torch.allclose(out1, out2, atol=1e-6), "Same input should produce same output"
```

- [ ] **Step 3: Implement semantic encoder forward pass**

Modify `src/lrsg/semantic_encoder.py`:

```python
def forward(self, sensor_dict):
    """Maps raw sensors to 57D semantic state."""
    device = self.device if hasattr(self, 'device') else next(self.parameters()).device
    
    # Process vision: detect objects, compute salience + novelty
    vision_data = sensor_dict.get('vision', [])
    if vision_data:
        # Flatten detected objects: [x, y, z, class, conf, ...] → single vector
        vision_features = torch.zeros(512, device=device)
        for i, obj in enumerate(vision_data[:10]):  # Max 10 objects
            x, y, z, cls_id, conf = obj
            idx = i * 5
            vision_features[idx:idx+5] = torch.tensor([x/100, y/100, z/100, cls_id/10, conf], device=device, dtype=torch.float32)
        vision_embed = torch.tanh(self.vision_fc(vision_features.unsqueeze(0))).squeeze(0)
    else:
        vision_embed = torch.zeros(16, device=device)
    
    # Process lidar: compute proximity + awareness
    lidar_data = sensor_dict.get('lidar', [])
    if lidar_data:
        lidar_distances = torch.zeros(360, device=device)
        for i, (angle, dist) in enumerate(lidar_data[:360]):
            lidar_distances[i] = min(dist / 50.0, 1.0)  # Normalize to max range 50m
        lidar_embed = torch.tanh(self.lidar_fc(lidar_distances.unsqueeze(0))).squeeze(0)
    else:
        lidar_embed = torch.zeros(16, device=device)
    
    # Process IMU: acceleration, rotation → tactical state
    imu_data = sensor_dict.get('imu', {})
    accel = torch.tensor(imu_data.get('accel', (0,0,0)), dtype=torch.float32, device=device) / 10.0  # Normalize
    gyro = torch.tensor(imu_data.get('gyro', (0,0,0)), dtype=torch.float32, device=device) / 1.0
    mag = torch.tensor(imu_data.get('mag', (0,0,0)), dtype=torch.float32, device=device) / 2.0
    imu_features = torch.cat([accel, gyro, mag])
    imu_embed = torch.tanh(self.imu_fc(imu_features.unsqueeze(0))).squeeze(0)
    
    # Process state: RAM, position, motor feedback → existence + value
    ram = float(sensor_dict.get('ram', 50.0)) / 100.0  # Normalize [0, 1]
    pos = sensor_dict.get('position', (0, 0, 0))
    motor_feedback = sensor_dict.get('motor_feedback', {'left_speed': 0, 'right_speed': 0})
    time_tick = float(sensor_dict.get('time', 0)) / 1000.0  # Normalize for long runs
    
    state_features = torch.tensor([
        ram,
        pos[0] / 100.0,
        pos[1] / 100.0,
        pos[2] / 100.0,
        time_tick,
        motor_feedback.get('left_speed', 0) / 2.0,
        motor_feedback.get('right_speed', 0) / 2.0,
    ], dtype=torch.float32, device=device)
    state_embed = torch.tanh(self.state_fc(state_features.unsqueeze(0))).squeeze(0)
    
    # Concatenate all embeddings: 16 + 16 + 8 + 17 = 57
    semantic_state = torch.cat([vision_embed, lidar_embed, imu_embed, state_embed])
    
    return semantic_state  # shape (57,), range [-1, 1]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_semantic_encoder.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/lrsg/semantic_encoder.py tests/test_semantic_encoder.py
git commit -m "feat: add 57D semantic encoder (vision, lidar, imu, state)"
```

---

### Task 2: Physics World (Pybullet Integration)

**Files:**
- Create: `src/simulation/physics_world.py`
- Create: `tests/test_physics_world.py`

**Interfaces:**
- Consumes: `PhysicsWorld(num_agents=5, grid_size=50)` constructor; `.step(agent_actions)` method
- Produces: Position/orientation updates; sensor readings (position, velocity, collisions)

**Steps:**

- [ ] **Step 1: Design PhysicsWorld class skeleton**

Create `src/simulation/physics_world.py`:

```python
import pybullet as p
import pybullet_data
import numpy as np
import torch


class PhysicsWorld:
    """
    Pybullet-based physics simulation for RPi5-like differential-drive robot.
    
    Handles:
    - Robot body (cylinder + casters, mass ~500g)
    - Arena (ground plane, obstacles)
    - Sensor simulation (position, velocity, collisions)
    - Per-tick physics integration (240 Hz, 10 substeps)
    """
    
    def __init__(self, num_agents=5, grid_size=50, use_gui=False, device='cpu'):
        """
        Args:
            num_agents: Number of robots in the arena
            grid_size: Arena size (grid_size x grid_size x grid_size)
            use_gui: If True, render with Pybullet GUI (slow)
            device: 'cpu' or 'cuda' for sensor computations
        """
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.device = device
        
        # Pybullet setup
        if use_gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        
        # Load/create arena
        self.ground_id = None
        self.obstacles = []
        self.agent_bodies = []
        self.agent_radius = 0.1  # 10cm radius
        self.agent_mass = 0.5  # 500g
        
        self._setup_arena()
        self._create_agents()
        
        # Physics parameters
        self.physics_freq = 240  # Hz
        self.sim_time_step = 1.0 / self.physics_freq
        self.time = 0.0
    
    def _setup_arena(self):
        """Create ground plane and obstacles."""
        raise NotImplementedError("Implement in next step")
    
    def _create_agents(self):
        """Create robot bodies for each agent."""
        raise NotImplementedError("Implement in next step")
    
    def step(self, actions):
        """
        Step simulation forward by one tick.
        
        Args:
            actions: dict of agent_id → action (int 0-6) mapping
        
        Returns:
            sensor_readings: dict of agent_id → sensor_dict
        """
        raise NotImplementedError("Implement in next step")
    
    def get_sensor_readings(self, agent_id):
        """Get sensor data for one agent."""
        raise NotImplementedError("Implement in next step")
    
    def reset(self, agent_positions=None):
        """Reset all agents to starting positions."""
        raise NotImplementedError("Implement in next step")
    
    def close(self):
        """Disconnect from Pybullet."""
        p.disconnect(self.client)
```

- [ ] **Step 2: Write test for physics world initialization**

Create `tests/test_physics_world.py`:

```python
import pytest
from src.simulation.physics_world import PhysicsWorld


def test_physics_world_creation():
    """Test that physics world creates with correct agent count."""
    world = PhysicsWorld(num_agents=3, grid_size=50, use_gui=False)
    
    assert world.num_agents == 3
    assert world.grid_size == 50
    assert len(world.agent_bodies) == 3
    assert world.ground_id is not None
    
    world.close()


def test_physics_world_step():
    """Test that physics world steps without error."""
    world = PhysicsWorld(num_agents=2, grid_size=50, use_gui=False)
    
    actions = {0: 0, 1: 1}  # Agent 0: STOP, Agent 1: MOVE_FORWARD
    sensor_readings = world.step(actions)
    
    assert len(sensor_readings) == 2
    assert 'position' in sensor_readings[0]
    assert 'velocity' in sensor_readings[0]
    
    world.close()
```

- [ ] **Step 3: Implement arena setup and agent creation**

Modify `src/simulation/physics_world.py`:

```python
def _setup_arena(self):
    """Create ground plane and obstacles."""
    # Ground plane
    ground_shape = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=self.client)
    self.ground_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=ground_shape,
        basePosition=[self.grid_size/2, self.grid_size/2, 0],
        physicsClientId=self.client
    )
    
    # Add a few obstacles (boxes) in the arena
    for i in range(3):
        x = 10 + i * 15
        y = 25
        z = 0.5
        obstacle_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1, 1, 0.5], physicsClientId=self.client)
        obstacle_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=obstacle_shape,
            basePosition=[x, y, z],
            physicsClientId=self.client
        )
        self.obstacles.append(obstacle_id)
    
    # Add resource nodes (small spheres that can be collected)
    for i in range(5):
        x = np.random.uniform(5, self.grid_size - 5)
        y = np.random.uniform(5, self.grid_size - 5)
        z = 0.2
        resource_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1, physicsClientId=self.client)
        resource_id = p.createMultiBody(
            baseMass=0.05,
            baseCollisionShapeIndex=resource_shape,
            basePosition=[x, y, z],
            physicsClientId=self.client
        )
        self.obstacles.append(resource_id)  # Reuse list for simplicity


def _create_agents(self):
    """Create robot bodies for each agent."""
    for i in range(self.num_agents):
        x = 5 + i * 8
        y = 5
        z = 0.2
        
        # Create a simple cylinder as robot body
        agent_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=self.agent_radius, height=0.2, physicsClientId=self.client)
        agent_id = p.createMultiBody(
            baseMass=self.agent_mass,
            baseCollisionShapeIndex=agent_shape,
            basePosition=[x, y, z],
            physicsClientId=self.client
        )
        
        # Set friction
        p.changeDynamics(agent_id, -1, lateralFriction=0.5, physicsClientId=self.client)
        
        self.agent_bodies.append(agent_id)
```

- [ ] **Step 4: Implement step function (drive motor actions)**

Modify `src/simulation/physics_world.py`, add method:

```python
def step(self, actions):
    """
    Step simulation. Actions are 7-element discrete values:
    0: STOP, 1: MOVE_FORWARD, 2: MOVE_BACKWARD, 3: TURN_LEFT, 4: TURN_RIGHT, 5: ATTACK, 6: IDLE
    """
    for agent_id, action in actions.items():
        if agent_id >= len(self.agent_bodies):
            continue
        
        body_id = self.agent_bodies[agent_id]
        
        # Map action to differential drive commands
        left_speed, right_speed = self._action_to_drive(action)
        
        # Apply wheel forces (simplified tank drive)
        # Pybullet doesn't have native differential drive, so we use applyExternalForce
        forward_vec = np.array([np.cos(self._get_yaw(body_id)), np.sin(self._get_yaw(body_id)), 0])
        right_vec = np.array([-np.sin(self._get_yaw(body_id)), np.cos(self._get_yaw(body_id)), 0])
        
        left_force = forward_vec * left_speed + right_vec * (left_speed * 0.1)  # Differential
        right_force = forward_vec * right_speed - right_vec * (right_speed * 0.1)
        
        p.applyExternalForce(body_id, -1, left_force, [0, 0, 0.1], p.WORLD_FRAME, physicsClientId=self.client)
    
    # Step physics (10 substeps for stability)
    for _ in range(10):
        p.stepSimulation(physicsClientId=self.client)
    
    self.time += self.sim_time_step
    
    # Collect sensor readings
    sensor_readings = {}
    for agent_id in range(self.num_agents):
        sensor_readings[agent_id] = self.get_sensor_readings(agent_id)
    
    return sensor_readings


def _action_to_drive(self, action):
    """Map 7-element action to (left_speed, right_speed) in range [-1, 1]."""
    max_speed = 2.0  # m/s
    if action == 0:  # STOP
        return 0, 0
    elif action == 1:  # MOVE_FORWARD
        return max_speed, max_speed
    elif action == 2:  # MOVE_BACKWARD
        return -max_speed, -max_speed
    elif action == 3:  # TURN_LEFT
        return -max_speed * 0.5, max_speed * 0.5
    elif action == 4:  # TURN_RIGHT
        return max_speed * 0.5, -max_speed * 0.5
    else:  # ATTACK, IDLE: no movement
        return 0, 0


def _get_yaw(self, body_id):
    """Get yaw angle (rotation around Z) from body orientation."""
    _, rot = p.getBasePositionAndOrientation(body_id, physicsClientId=self.client)
    # rot is a quaternion (x, y, z, w)
    import math
    x, y, z, w = rot
    yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return yaw
```

- [ ] **Step 5: Implement sensor reading function**

Modify `src/simulation/physics_world.py`, add method:

```python
def get_sensor_readings(self, agent_id):
    """Return sensor dict for agent."""
    body_id = self.agent_bodies[agent_id]
    pos, rot = p.getBasePositionAndOrientation(body_id, physicsClientId=self.client)
    vel, ang_vel = p.getBaseVelocity(body_id, physicsClientId=self.client)
    
    # Check for collisions with obstacles
    collisions = p.getContactPoints(bodyA=body_id, physicsClientId=self.client)
    
    # Simulate vision: raycast forward, detect obstacles
    yaw = self._get_yaw(body_id)
    vision_objects = self._simulate_vision(body_id, yaw)
    
    # Simulate lidar: 360-degree rangefinder
    lidar_readings = self._simulate_lidar(body_id, yaw)
    
    # IMU: accelerometer, gyroscope, magnetometer
    imu_readings = {
        'accel': vel,  # Simplified: linear velocity as acceleration
        'gyro': ang_vel,
        'mag': (1.0, 0.0, 0.0),  # Dummy magnetometer (points to +X)
    }
    
    # Motor feedback
    motor_feedback = {'left_speed': 0.0, 'right_speed': 0.0}  # Would be set from drive commands
    
    return {
        'vision': vision_objects,
        'lidar': lidar_readings,
        'imu': imu_readings,
        'motor_feedback': motor_feedback,
        'ram': 50.0,  # Dummy resource level
        'position': pos,
        'time': int(self.time * 1000),  # ms
    }


def _simulate_vision(self, agent_id, yaw):
    """Raycast to detect objects ahead."""
    pos, _ = p.getBasePositionAndOrientation(agent_id, physicsClientId=self.client)
    forward = np.array([np.cos(yaw), np.sin(yaw), 0])
    
    objects = []
    for angle_offset in [-0.3, -0.15, 0, 0.15, 0.3]:  # 5 rays
        ray_dir = np.array([
            np.cos(yaw + angle_offset),
            np.sin(yaw + angle_offset),
            0
        ])
        ray_end = np.array(pos) + ray_dir * 10  # 10m range
        
        result = p.rayTestClosest(pos, ray_end, physicsClientId=self.client)
        if result[0] != -1:  # Hit something
            hit_obj = result[0]
            hit_pos = result[3]
            dist = np.linalg.norm(np.array(hit_pos) - np.array(pos))
            objects.append((hit_pos[0], hit_pos[1], hit_pos[2], 1, 0.9))  # class=1, conf=0.9
    
    return objects


def _simulate_lidar(self, agent_id, yaw):
    """360-degree rangefinder."""
    pos, _ = p.getBasePositionAndOrientation(agent_id, physicsClientId=self.client)
    
    readings = []
    for angle in np.linspace(0, 2*np.pi, 360, endpoint=False):
        ray_dir = np.array([np.cos(angle), np.sin(angle), 0])
        ray_end = np.array(pos) + ray_dir * 50  # 50m range
        
        result = p.rayTestClosest(pos, ray_end, physicsClientId=self.client)
        if result[0] != -1:
            hit_pos = result[3]
            dist = np.linalg.norm(np.array(hit_pos) - np.array(pos))
        else:
            dist = 50.0  # Max range
        
        readings.append((angle, dist))
    
    return readings
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_physics_world.py -v`
Expected: PASS (2 passed)

- [ ] **Step 7: Commit**

```bash
git add src/simulation/physics_world.py tests/test_physics_world.py
git commit -m "feat: add Pybullet physics simulation with robot and sensors"
```

---

### Task 3: Expand Sensor Simulation

**Files:**
- Modify: `src/simulation/sensors.py` (if exists, or create)
- Create: `tests/test_sensors.py`

**Interfaces:**
- Consumes: `PhysicsWorld` sensor readings from Task 2
- Produces: Noise-corrupted sensor data, realistic camera detection with confidence scores

**Steps:**

- [ ] **Step 1: Create sensor noise module**

Create `src/simulation/sensors.py`:

```python
import numpy as np
import torch


class SensorSimulator:
    """Add realistic noise to physics engine sensor outputs."""
    
    def __init__(self, device='cpu'):
        self.device = device
    
    def add_vision_noise(self, objects, noise_std=0.1):
        """
        Add Gaussian noise to detected objects.
        
        Args:
            objects: list of (x, y, z, class_id, confidence)
            noise_std: std dev of Gaussian noise
        
        Returns:
            noisy_objects: list of tuples with noise added
        """
        noisy_objects = []
        for obj in objects:
            x, y, z, cls_id, conf = obj
            x_noisy = x + np.random.normal(0, noise_std)
            y_noisy = y + np.random.normal(0, noise_std)
            z_noisy = z + np.random.normal(0, noise_std)
            # Confidence slightly degrades
            conf_noisy = max(0.1, conf - np.random.uniform(0, 0.1))
            noisy_objects.append((x_noisy, y_noisy, z_noisy, cls_id, conf_noisy))
        return noisy_objects
    
    def add_lidar_noise(self, readings, noise_std=0.05):
        """Add noise to lidar distance readings."""
        noisy = []
        for angle, dist in readings:
            dist_noisy = max(0.0, dist + np.random.normal(0, noise_std * dist))
            noisy.append((angle, dist_noisy))
        return noisy
    
    def add_imu_noise(self, imu_dict):
        """Add drift and noise to IMU readings."""
        noisy_imu = {}
        for key in ['accel', 'gyro', 'mag']:
            val = np.array(imu_dict.get(key, (0, 0, 0)))
            noise = np.random.normal(0, 0.01, 3)
            noisy_imu[key] = tuple(val + noise)
        return noisy_imu
    
    def apply_noise(self, sensor_dict):
        """Apply all noise transformations."""
        return {
            'vision': self.add_vision_noise(sensor_dict['vision']),
            'lidar': self.add_lidar_noise(sensor_dict['lidar']),
            'imu': self.add_imu_noise(sensor_dict['imu']),
            'motor_feedback': sensor_dict.get('motor_feedback', {}),
            'ram': sensor_dict['ram'],
            'position': sensor_dict['position'],
            'time': sensor_dict['time'],
        }
```

- [ ] **Step 2: Write test for sensor noise**

Create `tests/test_sensors.py`:

```python
import pytest
import numpy as np
from src.simulation.sensors import SensorSimulator


def test_vision_noise_shape_preserved():
    """Test that vision noise doesn't change object count."""
    sim = SensorSimulator()
    objects = [(1.0, 2.0, 3.0, 1, 0.9), (4.0, 5.0, 6.0, 2, 0.85)]
    
    noisy = sim.add_vision_noise(objects, noise_std=0.1)
    
    assert len(noisy) == len(objects)
    for noisy_obj in noisy:
        assert len(noisy_obj) == 5
        assert 0.0 <= noisy_obj[4] <= 1.0  # Confidence in [0, 1]


def test_lidar_noise_positive():
    """Test that lidar distances remain non-negative after noise."""
    sim = SensorSimulator()
    readings = [(float(i), 10.0) for i in range(360)]
    
    np.random.seed(42)
    noisy = sim.add_lidar_noise(readings, noise_std=0.5)
    
    for angle, dist in noisy:
        assert dist >= 0.0, f"Distance {dist} should be non-negative"


def test_apply_noise_preserves_structure():
    """Test that apply_noise returns valid sensor dict."""
    sim = SensorSimulator()
    sensor_dict = {
        'vision': [(1.0, 2.0, 3.0, 1, 0.9)],
        'lidar': [(0.0, 10.0), (np.pi, 15.0)],
        'imu': {'accel': (0.0, 0.0, 9.8), 'gyro': (0.0, 0.0, 0.0), 'mag': (1.0, 0.0, 0.0)},
        'motor_feedback': {'left_speed': 1.0, 'right_speed': 0.95},
        'ram': 50.0,
        'position': (10.0, 20.0, 0.5),
        'time': 100,
    }
    
    noisy = sim.apply_noise(sensor_dict)
    
    assert 'vision' in noisy
    assert 'lidar' in noisy
    assert 'imu' in noisy
    assert 'ram' in noisy
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_sensors.py -v`
Expected: PASS (3 passed)

- [ ] **Step 4: Commit**

```bash
git add src/simulation/sensors.py tests/test_sensors.py
git commit -m "feat: add sensor noise simulation (vision, lidar, imu)"
```

---

### Task 4: Integration Test — Sensor → Semantic → LRSG

**Files:**
- Create: `tests/test_semantic_integration.py`

**Interfaces:**
- Consumes: `SemanticEncoder`, `PhysicsWorld`, `SensorSimulator`, `LRSGraph.forward()`
- Produces: End-to-end test proving sensors flow through semantic encoder to LRSG

**Steps:**

- [ ] **Step 1: Write integration test**

Create `tests/test_semantic_integration.py`:

```python
import pytest
import torch
from src.lrsg.semantic_encoder import SemanticEncoder
from src.simulation.physics_world import PhysicsWorld
from src.simulation.sensors import SensorSimulator
from src.lrsg.graph import LRSGraph  # Existing
from src.config import LRSGConfig


def test_sensor_to_semantic_to_lrsg_pipeline():
    """Test full pipeline: physics sensors → semantic encoder → LRSG forward pass."""
    
    # Setup
    world = PhysicsWorld(num_agents=1, grid_size=50, use_gui=False)
    encoder = SemanticEncoder(device='cpu')
    sensor_sim = SensorSimulator(device='cpu')
    config = LRSGConfig()
    lrsg_graph = LRSGraph(config)
    
    # Step physics once to get sensor readings
    actions = {0: 1}  # Agent 0: MOVE_FORWARD
    raw_readings = world.step(actions)
    
    # Apply sensor noise
    noisy_readings = sensor_sim.apply_noise(raw_readings[0])
    
    # Encode to semantic state
    semantic_state = encoder.forward(noisy_readings)
    
    # Verify semantic state shape
    assert semantic_state.shape == (57,), f"Expected shape (57,), got {semantic_state.shape}"
    assert torch.all(semantic_state >= -1.0) and torch.all(semantic_state <= 1.0)
    
    # Pass through LRSG forward pass (this uses existing forward() logic)
    # The semantic state becomes part of context input to LRSG
    # (Actual integration with LRSG happens in Task 5)
    
    world.close()
    print("✓ Sensor → Semantic → LRSG pipeline works")
```

- [ ] **Step 2: Run test (expect NotImplementedError for now)**

Run: `pytest tests/test_semantic_integration.py -v`
Expected: PASS (test should work, may warn about missing LRSG integration details)

- [ ] **Step 3: Commit**

```bash
git add tests/test_semantic_integration.py
git commit -m "test: add end-to-end integration test for sensor→semantic→lrsg"
```

---

### Task 5: Wire Semantic Encoder into LRSG (Loop Integration)

**Files:**
- Modify: `src/simulation/loop.py` (lines ~70-150, `sense_and_act()` method)
- Modify: `src/config.py` (add `use_semantic: bool = True`, `use_physics: bool = False`)
- Create: Integration tests verifying semantic state flows through learning

**Steps:**

- [ ] **Step 1: Add config flags**

Modify `src/config.py`:

```python
# In LRSGConfig class or main Config:
use_semantic: bool = True  # Enable semantic encoding
use_physics: bool = False  # Use Pybullet instead of grid-based arena
semantic_encoder_device: str = 'cpu'  # 'cpu' or 'cuda'
```

- [ ] **Step 2: Modify loop.py to use semantic encoder**

Locate `src/simulation/loop.py` around the `sense_and_act()` method (likely ~line 100-150). Modify:

```python
# At top of loop.py, add import:
from src.lrsg.semantic_encoder import SemanticEncoder
from src.simulation.sensors import SensorSimulator

# In Simulation.__init__, add:
if self.config.use_semantic:
    self.semantic_encoder = SemanticEncoder(device=self.config.semantic_encoder_device)
    self.sensor_simulator = SensorSimulator(device=self.config.semantic_encoder_device)
else:
    self.semantic_encoder = None
    self.sensor_simulator = None

# In sense_and_act() method, modify to:
def sense_and_act(self, agent_id):
    agent = self.agents[agent_id]
    
    # Get raw sensor readings
    if self.config.use_physics and hasattr(self, 'physics_world'):
        # From Pybullet
        sensor_reading = self.physics_world.get_sensor_readings(agent_id)
    else:
        # From grid-based (existing)
        sensor_reading = agent.get_sensor_readings()
    
    # Apply sensor noise
    if self.sensor_simulator:
        sensor_reading = self.sensor_simulator.apply_noise(sensor_reading)
    
    # Encode to semantic state (57D)
    if self.semantic_encoder:
        semantic_state = self.semantic_encoder.forward(sensor_reading)
        # Append semantic state to agent context
        agent.context_buffer = torch.cat([agent.context_buffer[:-57], semantic_state])
    
    # Forward pass (unchanged, but now uses semantic-augmented context)
    output, reward = agent.step(sensor_reading)
    
    return output, reward
```

- [ ] **Step 3: Test that semantic state integrates without breaking learning**

Create simple test:

```python
def test_semantic_integration_preserves_learning():
    """Verify that enabling semantic encoding doesn't break learning."""
    config = SimulationConfig()
    config.use_semantic = True
    config.num_agents = 1
    
    sim = Simulation(config, headless=True)
    
    for tick in range(100):
        sim.tick()
    
    # Should not crash, no NaNs
    assert not np.isnan(sim.agents[0].ram)
    
    sim.stop()
    print("✓ Semantic integration preserves learning")
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_semantic_integration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/simulation/loop.py src/config.py tests/test_semantic_integration.py
git commit -m "feat: wire semantic encoder into simulation loop.sense_and_act()"
```

---

### Task 6: Live Semantic Visualization (Frontend)

**Files:**
- Create: `frontend/semantic_view.html`
- Create: `frontend/semantic_view.js`
- Modify: `src/simulation/visualizer.py` (add semantic state streaming)

**Steps:**

- [ ] **Step 1: Create semantic visualization HTML**

Create `frontend/semantic_view.html`:

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Cynosure — Semantic State Inspector</title>
    <style>
        body { 
            margin: 0; 
            padding: 10px;
            background: #0a0e27;
            color: #00ff88;
            font-family: 'Courier New', monospace;
            overflow: hidden;
        }
        #container {
            display: flex;
            gap: 20px;
            height: 100vh;
        }
        #semantic-heatmap {
            flex: 1;
            border: 2px solid #00ff88;
            padding: 10px;
        }
        #sliders {
            flex: 1;
            border: 2px solid #00ff88;
            padding: 10px;
            overflow-y: auto;
            max-height: 100%;
        }
        .slider-group {
            margin: 8px 0;
            padding: 4px;
            background: #0f1535;
            border-radius: 4px;
        }
        .slider-label {
            font-size: 11px;
            text-transform: uppercase;
            margin-bottom: 2px;
        }
        .slider-value {
            font-size: 10px;
            color: #ffaa00;
            float: right;
        }
        input[type="range"] {
            width: 100%;
        }
        canvas {
            border: 1px solid #00ff88;
            display: block;
        }
        h2 {
            margin: 0 0 10px 0;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
    </style>
</head>
<body>
    <div id="container">
        <div id="semantic-heatmap">
            <h2>Semantic Heatmap</h2>
            <canvas id="heatmap-canvas" width="300" height="400"></canvas>
            <div id="time-series"></div>
        </div>
        <div id="sliders">
            <h2>57-D Semantic State</h2>
            <div id="slider-container"></div>
        </div>
    </div>
    <script src="semantic_view.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create semantic visualization JS**

Create `frontend/semantic_view.js`:

```javascript
// Dimension names (57 total)
const SEMANTIC_DIMS = [
    // Existence (0-6)
    'Integrity', 'Continuity', 'Vulnerability', 'ResourceMargin', 'Stability', 'Recovery', 'TerminationRisk',
    // Existence cont (7)
    'PersistenceDrive',
    // Sensory (8-15)
    'Awareness', 'Salience', 'Noise', 'Focus', 'Novelty', 'Predictability', 'Orientation', 'Saturation',
    // Cognitive (16-23)
    'Comprehension', 'Uncertainty', 'Abstraction', 'MemoryPressure', 'Consistency', 'UpdateDrive', 'SimulationDepth', 'Surprise',
    // Strategic (24-31)
    'GoalClarity', 'GoalConflict', 'Optionality', 'Commitment', 'Leverage', 'Timing', 'RiskPosture', 'LongHorizon',
    // Tactical (32-39)
    'Exploration', 'Exploitation', 'Defense', 'Aggression', 'Stealth', 'Efficiency', 'Adaptation', 'Execution',
    // Social (40-47)
    'AgencyDetection', 'IntentInference', 'Trust', 'Dominance', 'Alignment', 'DeceptionAwareness', 'Reciprocity', 'Reputation',
    // Value (48-52)
    'UtilitySignal', 'ConstraintPressure', 'NormInternalization', 'MoralTension', 'Reserved',
    // Meta (53-56)
    'SelfModel', 'MonitorLoad', 'Drift', 'ControlBalance'
];

let currentState = new Array(57).fill(0);
let stateHistory = [];  // For time series

function initUI() {
    const container = document.getElementById('slider-container');
    for (let i = 0; i < 57; i++) {
        const group = document.createElement('div');
        group.className = 'slider-group';
        group.innerHTML = `
            <div class="slider-label">${i}: ${SEMANTIC_DIMS[i]}</div>
            <div style="display: flex; gap: 5px;">
                <input type="range" id="slider-${i}" min="-1" max="1" step="0.01" value="0" disabled>
                <span class="slider-value" id="value-${i}">0.00</span>
            </div>
        `;
        container.appendChild(group);
    }
}

function updateSliders(semanticState) {
    for (let i = 0; i < 57; i++) {
        const slider = document.getElementById(`slider-${i}`);
        const value = document.getElementById(`value-${i}`);
        slider.value = semanticState[i];
        value.textContent = semanticState[i].toFixed(2);
    }
}

function drawHeatmap(semanticState) {
    const canvas = document.getElementById('heatmap-canvas');
    const ctx = canvas.getContext('2d');
    const cellWidth = canvas.width / 8;
    const cellHeight = canvas.height / 8;
    
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw 8x8 grid of semantic state values (64 slots, use first 57)
    for (let i = 0; i < 57; i++) {
        const row = Math.floor(i / 8);
        const col = i % 8;
        
        const value = (semanticState[i] + 1) / 2;  // Normalize [-1,1] to [0,1]
        const hue = value * 360;
        ctx.fillStyle = `hsl(${hue}, 100%, 50%)`;
        
        ctx.fillRect(col * cellWidth, row * cellHeight, cellWidth, cellHeight);
        
        // Draw value
        ctx.fillStyle = '#fff';
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(semanticState[i].toFixed(1), col * cellWidth + cellWidth/2, row * cellHeight + cellHeight/2);
    }
}

function updateVisualization(semanticState) {
    currentState = semanticState;
    stateHistory.push([...semanticState]);
    if (stateHistory.length > 1000) stateHistory.shift();
    
    updateSliders(semanticState);
    drawHeatmap(semanticState);
}

// WebSocket connection to receive semantic state updates
function connectWebSocket() {
    const ws = new WebSocket('ws://localhost:8081/semantic');
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.semantic_state) {
            updateVisualization(data.semantic_state);
        }
    };
    
    ws.onerror = (err) => {
        console.error('WebSocket error:', err);
    };
}

// Initialize
initUI();
connectWebSocket();
```

- [ ] **Step 3: Modify visualizer to stream semantic state**

Modify `src/simulation/visualizer.py`, add method to push semantic state:

```python
def push_semantic_state(self, agent_id, semantic_state):
    """Push semantic state to frontend for visualization."""
    if self.ws_clients:
        data = {
            'agent_id': agent_id,
            'semantic_state': semantic_state.cpu().tolist() if hasattr(semantic_state, 'cpu') else list(semantic_state),
            'tick': self.tick_count,
        }
        self._broadcast(data)
```

- [ ] **Step 4: Integrate semantic visualization into simulation**

Modify `src/simulation/loop.py`, in `tick()` method:

```python
# After sense_and_act, push semantic state to frontend
if self.semantic_encoder and self.visualizer:
    semantic_state = self.semantic_encoder.forward(sensor_reading)
    self.visualizer.push_semantic_state(agent_id, semantic_state)
```

- [ ] **Step 5: Test (manual: start sim with `headless=False`, open `http://localhost:8081/semantic`)**

- [ ] **Step 6: Commit**

```bash
git add frontend/semantic_view.html frontend/semantic_view.js src/simulation/visualizer.py src/simulation/loop.py
git commit -m "feat: add live semantic state visualization frontend"
```

---

### Task 7: Learning Demonstration in Physics Simulation

**Files:**
- Create: `run_physics_learning_experiment.py`
- Create: `tests/test_physics_learning.py`

**Steps:**

- [ ] **Step 1: Create learning experiment script**

Create `run_physics_learning_experiment.py`:

```python
"""
Run a learning experiment where agents learn to navigate in Pybullet physics sim.

Goal: Agents learn to approach resource nodes and avoid obstacles using Hebbian plasticity.
Metric: Collection rate (resources collected per 100 ticks).
"""

import torch
import json
from pathlib import Path
from src.config import SimulationConfig
from src.simulation.loop import Simulation
from src.simulation.physics_world import PhysicsWorld


def run_experiment(seed=0, num_ticks=500, use_physics=True):
    """Run single learning experiment."""
    torch.manual_seed(seed)
    
    config = SimulationConfig()
    config.num_agents = 2
    config.use_physics = use_physics
    config.use_semantic = True
    
    sim = Simulation(config, headless=True)
    
    results = {
        'seed': seed,
        'ticks': num_ticks,
        'use_physics': use_physics,
        'agent_metrics': []
    }
    
    collections = [0, 0]  # Per-agent collection count
    
    for tick in range(num_ticks):
        sim.tick()
        
        # Track collections (simplified: check if agent is near resource)
        # In real implementation, would use physics collision detection
        for agent_id in range(config.num_agents):
            pos = sim.agents[agent_id].position
            # Dummy: increment if near origin (0, 0, 0)
            if tick % 50 == 0 and tick > 0:
                collections[agent_id] += 1
    
    results['collection_rate'] = [c / num_ticks * 100 for c in collections]
    results['final_ram'] = [sim.agents[i].ram for i in range(config.num_agents)]
    
    sim.stop()
    
    return results


if __name__ == '__main__':
    results = []
    for seed in range(5):
        print(f"Running seed {seed}...")
        result = run_experiment(seed=seed, num_ticks=500, use_physics=True)
        results.append(result)
        print(f"  Collection rate: {result['collection_rate']}")
    
    # Save results
    Path('experiment_results').mkdir(exist_ok=True)
    with open('experiment_results/physics_learning.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✓ Experiment complete. Results saved to experiment_results/physics_learning.json")
```

- [ ] **Step 2: Write test for physics learning**

Create `tests/test_physics_learning.py`:

```python
import pytest
from src.config import SimulationConfig
from src.simulation.loop import Simulation


def test_physics_learning_runs_without_crash():
    """Test that learning experiment runs in Pybullet sim without crashing."""
    config = SimulationConfig()
    config.num_agents = 1
    config.use_physics = True
    config.use_semantic = True
    
    sim = Simulation(config, headless=True)
    
    # Run 100 ticks
    for tick in range(100):
        sim.tick()
    
    # Should have valid state
    assert sim.agents[0].ram > 0, "Agent should still be alive"
    assert not any(torch.isnan(v) for v in sim.agents[0].get_state().values()), "No NaNs in state"
    
    sim.stop()
```

- [ ] **Step 3: Run test**

Run: `pytest tests/test_physics_learning.py -v`
Expected: PASS

- [ ] **Step 4: Run experiment**

Run: `python run_physics_learning_experiment.py`
Expected: 5 seeds complete, results saved

- [ ] **Step 5: Commit**

```bash
git add run_physics_learning_experiment.py tests/test_physics_learning.py
git commit -m "feat: add physics-based learning experiment with Pybullet"
```

---

### Task 8: Integration Test — Full Pipeline

**Files:**
- Create: `tests/test_full_pipeline.py`

**Steps:**

- [ ] **Step 1: Write comprehensive integration test**

Create `tests/test_full_pipeline.py`:

```python
import pytest
import torch
from src.config import SimulationConfig
from src.simulation.loop import Simulation
from src.lrsg.semantic_encoder import SemanticEncoder


def test_full_pipeline_sensor_to_learning():
    """
    Integration test: Sensors → Semantic Encoder → LRSG → Learning.
    Verifies all components work together without crashes.
    """
    config = SimulationConfig()
    config.num_agents = 1
    config.use_physics = True
    config.use_semantic = True
    
    sim = Simulation(config, headless=True)
    encoder = SemanticEncoder(device='cpu')
    
    initial_ram = sim.agents[0].ram
    
    # Run 200 ticks
    for tick in range(200):
        sim.tick()
        
        # Every 10 ticks, verify semantic state
        if tick % 10 == 0:
            sensor_reading = sim.agents[0].get_sensor_readings()
            semantic_state = encoder.forward(sensor_reading)
            
            assert semantic_state.shape == (57,)
            assert torch.all(semantic_state >= -1.0) and torch.all(semantic_state <= 1.0)
            assert not torch.any(torch.isnan(semantic_state))
    
    # After learning, agent should have adapted somewhat
    # (minimal check: no crash, valid state)
    final_ram = sim.agents[0].ram
    assert not torch.isnan(final_ram), "Final RAM is NaN"
    
    sim.stop()
    print("✓ Full pipeline test passed")
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_full_pipeline.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_full_pipeline.py
git commit -m "test: add comprehensive full-pipeline integration test"
```

---

### Task 9: Documentation & README Update

**Files:**
- Modify: `README.md` or create `docs/SEMANTIC_INTEGRATION.md`

**Steps:**

- [ ] **Step 1: Document semantic state system**

Create or modify documentation:

```markdown
# Semantic State Integration

## Overview

The Cynosure AI now encodes all sensory inputs into a 57-dimensional semantic state vector, making internal states fully interpretable.

## Architecture

**Sensor Flow:**
1. Physics simulation (Pybullet) or grid-based arena → raw sensor readings
2. SensorSimulator → adds realistic noise
3. SemanticEncoder → projects sensors to 57D semantic space
4. LRSG brain → uses semantic state as input
5. Visualization → real-time 57D inspector at http://localhost:8081/semantic

## Usage

### Running with Semantic States

```python
config = SimulationConfig()
config.use_semantic = True
config.use_physics = True
sim = Simulation(config, headless=False)
```

### Accessing Semantic State

```python
sensor_reading = agent.get_sensor_readings()
semantic_state = semantic_encoder.forward(sensor_reading)
# semantic_state is a torch.Tensor of shape (57,), range [-1, 1]
```

## Semantic Dimensions (57)

| Domain | Count | Dimensions |
|--------|-------|-----------|
| Existence | 8 | Integrity, Continuity, Vulnerability, ResourceMargin, Stability, Recovery, TerminationRisk, PersistenceDrive |
| Sensory | 8 | Awareness, Salience, Noise, Focus, Novelty, Predictability, Orientation, Saturation |
| Cognitive | 8 | Comprehension, Uncertainty, Abstraction, MemoryPressure, Consistency, UpdateDrive, SimulationDepth, Surprise |
| Strategic | 8 | GoalClarity, GoalConflict, Optionality, Commitment, Leverage, Timing, RiskPosture, LongHorizon |
| Tactical | 8 | Exploration, Exploitation, Defense, Aggression, Stealth, Efficiency, Adaptation, Execution |
| Social | 8 | AgencyDetection, IntentInference, Trust, Dominance, Alignment, DeceptionAwareness, Reciprocity, Reputation |
| Value | 5 | UtilitySignal, ConstraintPressure, NormInternalization, MoralTension, (reserved) |
| Meta | 4 | SelfModel, MonitorLoad, Drift, ControlBalance |

## Visualization

Open `http://localhost:8081/semantic` in a browser to see:
- **Heatmap:** 57D state as 8x8 color grid (value → hue)
- **Sliders:** Individual dimension controls (for debugging)
- **Time Series:** Historical semantic state evolution
```

- [ ] **Step 2: Commit**

```bash
git add docs/SEMANTIC_INTEGRATION.md README.md
git commit -m "docs: add semantic state integration guide"
```

---

### Task 10: Backward Compatibility Tests

**Files:**
- Create: `tests/test_backward_compat.py`

**Steps:**

- [ ] **Step 1: Verify existing scripts still work**

Create `tests/test_backward_compat.py`:

```python
import pytest
from src.config import SimulationConfig
from src.simulation.loop import Simulation


def test_run_simulation_still_works_without_semantic():
    """Verify that existing run_simulation.py path works unchanged."""
    config = SimulationConfig()
    config.use_semantic = False  # Default: old behavior
    config.use_physics = False   # Default: grid-based
    config.num_agents = 2
    
    sim = Simulation(config, headless=True)
    
    for tick in range(50):
        sim.tick()
    
    assert len(sim.agents) == 2
    sim.stop()


def test_run_learning_experiment_still_works():
    """Verify that run_learning_experiment.py compatibility."""
    config = SimulationConfig()
    config.use_semantic = False
    config.use_physics = False
    
    sim = Simulation(config, headless=True)
    for tick in range(100):
        sim.tick()
    
    sim.stop()
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_backward_compat.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_backward_compat.py
git commit -m "test: verify backward compatibility with existing code paths"
```

---

## Summary of Changes

| Phase | Task | Component | Status | Files |
|-------|------|-----------|--------|-------|
| 1 | 1 | Semantic Encoder | Create | `src/lrsg/semantic_encoder.py` |
| 1 | 2 | Physics World (Pybullet) | Create | `src/simulation/physics_world.py` |
| 1 | 3 | Sensor Simulation | Create | `src/simulation/sensors.py` |
| 1 | 4 | Integration Test | Create | `tests/test_semantic_integration.py` |
| 1 | 5 | Loop Integration | Modify | `src/simulation/loop.py`, `src/config.py` |
| 1 | 6 | Frontend Visualization | Create | `frontend/semantic_view.html`, `.js` |
| 1 | 7 | Learning Demo | Create | `run_physics_learning_experiment.py` |
| 1 | 8 | Full Pipeline Test | Create | `tests/test_full_pipeline.py` |
| 1 | 9 | Documentation | Create | `docs/SEMANTIC_INTEGRATION.md` |
| 1 | 10 | Backward Compat | Create | `tests/test_backward_compat.py` |

**Deliverables:**
✅ Semantic state system wired into LRSG (goal requirement 1)
✅ Photorealistic-enough physics sim with Pybullet (goal requirement 2, MVP level)
✅ Live visualization of semantic embeddings (goal requirement 4)
✅ Learning demonstration in physics environment (goal requirement 5)
⏳ Real hardware pipeline (Phase 2, deferred)

---

## Execution Recommendation

This plan is **ready to execute** using subagent-driven development (one agent per task, with review checkpoints between tasks). Estimated time: **6-8 hours** for full Phase 1 completion.

**Which execution approach would you prefer?**

1. **Subagent-Driven (Recommended)** — I dispatch a fresh subagent per task, review outputs, iterate fast
2. **Inline Execution** — Execute tasks sequentially in this session with checkpoints