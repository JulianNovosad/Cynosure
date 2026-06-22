# Semantic State Integration Guide

## 1. Overview

Cynosure's semantic integration system maps raw sensor inputs (vision, lidar, IMU) into a continuous 57-dimensional semantic embedding that represents the AI's internal state. This embedding captures existential, perceptual, cognitive, strategic, tactical, social, and value-driven aspects of the agent's "experience."

The semantic state flows through the entire system:
- **Input:** Raw sensors (vision, lidar, IMU, motor feedback) + RAM level + position
- **Encoding:** `SemanticEncoder` neural network projects sensors to 57D semantic space
- **Integration:** Semantic state is concatenated with LRSG context and fed into the brain's forward pass
- **Output:** 57D embedding influences action selection bias and motor output modulation
- **Visualization:** Live web dashboard at `http://localhost:8080/semantic` displays semantic state in real-time

This enables interpretability: at any simulation tick, you can read the semantic embedding and understand what the AI is "thinking/feeling."

---

## 2. Architecture: Sensor → Semantic → Output Flow

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ SENSORY INPUT LAYER                                          │
├─────────────────────────────────────────────────────────────┤
│ Vision (detected objects)  │ Lidar (360 rays)               │
│ IMU (accel, gyro, mag)     │ Motor feedback (speeds)        │
│ RAM level (0-100)          │ Position (x, y, z)            │
│ Time (tick number)         │                                │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ SEMANTIC ENCODER (SemanticEncoder PyTorch module)           │
├─────────────────────────────────────────────────────────────┤
│ vision_fc:      Objects → 16D (sensory + cognitive)        │
│ lidar_fc:       360 rays → 16D (spatial awareness)         │
│ imu_fc:         9 IMU values → 8D (tactical state)         │
│ state_fc:       RAM + pos + time → 17D (existence + value) │
│ Concat all:     16 + 16 + 8 + 17 = 57D semantic state      │
│ Activation:     tanh applied to sub-embeddings             │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ LRSG BRAIN                                                   │
├─────────────────────────────────────────────────────────────┤
│ Input: 57D semantic state + previous context                │
│ Forward pass: 5-step propagation through LRSG graph        │
│ Output: Brain activations, predicted next state             │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ SEMANTIC-MOTOR INTERFACE (SemanticMotorInterface)          │
├─────────────────────────────────────────────────────────────┤
│ semantic_to_action_bias(): 57D → 7D action biases          │
│ semantic_to_motor_output(): 57D + brain_output → motor     │
│ modulate_reward_by_semantic_progress(): track learning     │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ MOTOR OUTPUT                                                 │
├─────────────────────────────────────────────────────────────┤
│ 7 Discrete Actions: ATTACK, HUNT, COLLECT, MOVE, WHISPER,  │
│                     PROBE, MANIPULATE                        │
│ Modulated by: direction, speed, force (all semantic-driven) │
└─────────────────────────────────────────────────────────────┘
```

### Processing Chain Details

1. **Sensor Collection** (`simulation/loop.py` → `agent.py`)
   - Vision: detected objects `[(x, y, z, class_id, confidence), ...]`
   - Lidar: 360 distance readings (normalized to 0-1 range)
   - IMU: acceleration (3D), rotation rate (3D), magnetic field (3D)
   - State: RAM level, 3D position, elapsed ticks, motor feedback speeds

2. **Semantic Encoding** (`src/lrsg/semantic_encoder.py`)
   - Vision objects flattened to 512D feature vector → projected to 16D via `vision_fc`
   - Lidar distances → 16D spatial awareness via `lidar_fc`
   - IMU values → 8D tactical state via `imu_fc`
   - RAM/position/time/motor → 17D existence via `state_fc`
   - All sub-embeddings tanh-activated and concatenated to form 57D semantic state

3. **LRSG Integration** (`src/lrsg/forward.py`)
   - Semantic state concatenated with temporal context
   - Fed as input to the forward pass (part of sensory window)
   - LRSG output fed to action selection and semantic-motor interface

4. **Action Selection & Motor Output** (`src/lrsg/semantic_motor.py`)
   - `semantic_to_action_bias()` maps 57D state → 7D action biases
   - Each action weighted by relevant semantic dimensions:
     - **ATTACK**: aggression + agency detection + (1 - vulnerability)
     - **HUNT**: aggression + exploration + agency detection
     - **COLLECT**: salience + (1 - resource margin) + awareness
     - **MOVE**: awareness + exploration (exploration weighted 0.3, awareness 0.7) + goal clarity
     - **WHISPER**: uncertainty + agency detection
     - **PROBE**: exploration + uncertainty
     - **MANIPULATE**: leverage + update drive
   - Motor output (direction, speed, force) modulated by execution, awareness, goal clarity

---

## 3. Usage: Running Simulations with Semantic States

### Basic Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Import the semantic modules:**
   ```python
   from src.lrsg.semantic_encoder import SemanticEncoder
   from src.lrsg.semantic_motor import SemanticMotorInterface
   from src.simulation.loop import Simulation
   ```

3. **Initialize the simulation:**
   ```python
   from src.simulation.loop import Simulation
   from src.config import BASE_CONFIG
   
   config = BASE_CONFIG
   sim = Simulation(config=config, headless=False)
   ```

### Example 1: Run Standard Simulation with Semantic States

```python
"""
Run a 2-agent, 200-tick simulation with semantic states enabled.
Outputs telemetry to telemetry.jsonl and displays semantic visualization.
"""

from src.simulation.loop import Simulation
from src.config import BASE_CONFIG

config = BASE_CONFIG
config.num_agents = 2
config.num_ticks = 200

sim = Simulation(config=config, headless=False)
results = sim.run()

# Telemetry written to ./telemetry.jsonl
# Visualization available at http://localhost:8080
# Semantic view at http://localhost:8080/semantic
```

### Example 2: Log Semantic States During Training

```python
"""
Track semantic state evolution across an agent's lifecycle.
Useful for understanding learning dynamics.
"""

import json
from src.simulation.loop import Simulation
from src.config import BASE_CONFIG

semantic_log = []

config = BASE_CONFIG
config.num_agents = 1
config.num_ticks = 500

sim = Simulation(config=config, headless=True)

for tick in range(500):
    tick_results = sim.physics_tick(tick)
    
    # Extract semantic state for each agent
    for agent_id, agent in enumerate(sim.agents):
        semantic_state = agent.semantic_state  # 57D tensor
        semantic_log.append({
            'tick': tick,
            'agent_id': agent_id,
            'semantic_state': semantic_state.cpu().tolist(),
            'ram': float(agent.ram),
            'action': agent.last_action_name,
            'reward': float(agent.reward)
        })

# Save for post-hoc analysis
with open('semantic_evolution.jsonl', 'w') as f:
    for entry in semantic_log:
        f.write(json.dumps(entry) + '\n')
```

### Example 3: Query Semantic State at Specific Tick

```python
"""
Access the 57D semantic state of an agent at a specific tick.
Useful for debugging or understanding agent "state of mind."
"""

from src.simulation.loop import Simulation
from src.config import BASE_CONFIG

config = BASE_CONFIG
config.num_agents = 1
config.num_ticks = 100

sim = Simulation(config=config, headless=True)

# Run simulation
for tick in range(100):
    sim.physics_tick(tick)

# Query agent 0's semantic state at tick 50
agent = sim.agents[0]
semantic_state = agent.semantic_state  # shape: (57,), dtype: float32

# Interpret semantic state
existence = semantic_state[0:8]      # Existential domain
sensory = semantic_state[8:16]       # Sensory domain
cognitive = semantic_state[16:24]    # Cognitive domain
strategic = semantic_state[24:32]    # Strategic domain
tactical = semantic_state[32:40]     # Tactical domain
social = semantic_state[40:48]       # Social domain
value = semantic_state[48:53]        # Value domain
meta = semantic_state[53:57]         # Meta domain

print(f"Vulnerability: {existence[2]:.3f}")
print(f"Resource margin: {existence[3]:.3f}")
print(f"Awareness: {sensory[0]:.3f}")
print(f"Salience: {sensory[1]:.3f}")
```

### Example 4: Custom Semantic State Reward

```python
"""
Modulate reward based on semantic state progression.
Agents are rewarded for meaningful state changes.
"""

from src.lrsg.semantic_motor import SemanticMotorInterface
import torch

interface = SemanticMotorInterface(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

prev_semantic = torch.randn(57)  # Previous state
curr_semantic = torch.randn(57)  # Current state
base_reward = 1.0

# Reward progression in semantic space (novelty bonus)
modulated_reward = interface.modulate_reward_by_semantic_progress(
    base_reward,
    prev_semantic,
    curr_semantic
)

print(f"Base reward: {base_reward:.3f}")
print(f"Modulated reward: {modulated_reward:.3f}")
```

---

## 4. 57-Dimensional Semantic State Reference

### Dimension Table

| Index | Domain       | Name                  | Meaning                                          | Range |
|-------|--------------|----------------------|--------------------------------------------------|-------|
| 0     | Existence    | Integrity             | Degree of internal coherence and error          | [0,1] |
| 1     | Existence    | Continuity            | Expectation of continued operation               | [0,1] |
| 2     | Existence    | Vulnerability         | Exposure to irreversible failure                 | [0,1] |
| 3     | Existence    | ResourceMargin        | Distance from minimum viable resources           | [0,1] |
| 4     | Existence    | Stability             | Variance in internal processes                   | [0,1] |
| 5     | Existence    | Recovery              | Capacity to return from damage                   | [0,1] |
| 6     | Existence    | TerminationRisk       | Estimated probability of shutdown                | [0,1] |
| 7     | Existence    | PersistenceDrive      | Pressure to avoid non-recoverable states         | [0,1] |
| 8     | Sensory      | Awareness             | Breadth of active perception                     | [0,1] |
| 9     | Sensory      | Salience              | Strength of dominant signals                     | [0,1] |
| 10    | Sensory      | Noise                 | Degree of sensory contradiction                  | [0,1] |
| 11    | Sensory      | Focus                 | Allocation tightness of attention                | [0,1] |
| 12    | Sensory      | Novelty               | Detection of unfamiliar patterns                 | [0,1] |
| 13    | Sensory      | Predictability        | Match between expectation and input              | [0,1] |
| 14    | Sensory      | Orientation           | Spatial and temporal grounding                   | [0,1] |
| 15    | Sensory      | Saturation            | Perceptual overload level                        | [0,1] |
| 16    | Cognitive    | Comprehension         | Model fit to observed reality                    | [0,1] |
| 17    | Cognitive    | Uncertainty           | Confidence spread across hypotheses              | [0,1] |
| 18    | Cognitive    | Abstraction           | Level of symbolic compression                    | [0,1] |
| 19    | Cognitive    | MemoryPressure        | Load on working and long-term memory             | [0,1] |
| 20    | Cognitive    | Consistency           | Internal belief alignment                        | [0,1] |
| 21    | Cognitive    | UpdateDrive           | Urge to revise models                            | [0,1] |
| 22    | Cognitive    | SimulationDepth       | Lookahead horizon                                | [0,1] |
| 23    | Cognitive    | Surprise              | Prediction error magnitude                       | [0,1] |
| 24    | Strategic    | GoalClarity           | Sharpness of current objectives                  | [0,1] |
| 25    | Strategic    | GoalConflict          | Competition between objectives                   | [0,1] |
| 26    | Strategic    | Optionality           | Count of viable future paths                     | [0,1] |
| 27    | Strategic    | Commitment            | Resistance to goal switching                     | [0,1] |
| 28    | Strategic    | Leverage              | Ability to affect future states                  | [0,1] |
| 29    | Strategic    | Timing                | Sensitivity to delay or haste                    | [0,1] |
| 30    | Strategic    | RiskPosture           | Preference across outcome variance               | [0,1] |
| 31    | Strategic    | LongHorizon           | Weight given to distant outcomes                 | [0,1] |
| 32    | Tactical     | Exploration           | Drive to sample unknown states                   | [0,1] |
| 33    | Tactical     | Exploitation          | Drive to use known advantages                    | [0,1] |
| 34    | Tactical     | Defense               | Bias toward damage avoidance                     | [0,1] |
| 35    | Tactical     | Aggression            | Bias toward forceful intervention                | [0,1] |
| 36    | Tactical     | Stealth               | Preference for low observability                 | [0,1] |
| 37    | Tactical     | Efficiency            | Resource cost minimization                       | [0,1] |
| 38    | Tactical     | Adaptation            | Speed of behavioral change                       | [0,1] |
| 39    | Tactical     | Execution             | Action throughput and precision                  | [0,1] |
| 40    | Social       | AgencyDetection       | Recognition of other actors                      | [0,1] |
| 41    | Social       | IntentInference       | Estimation of others' goals                      | [0,1] |
| 42    | Social       | Trust                 | Expected reliability of others                   | [0,1] |
| 43    | Social       | Dominance             | Relative power assessment                        | [0,1] |
| 44    | Social       | Alignment             | Goal overlap with others                         | [0,1] |
| 45    | Social       | DeceptionAwareness    | Detection of adversarial signaling               | [0,1] |
| 46    | Social       | Reciprocity           | Memory of exchanges                              | [0,1] |
| 47    | Social       | Reputation            | Modeled external perception                      | [0,1] |
| 48    | Value        | UtilitySignal         | Scalar measure of desirability                   | [0,1] |
| 49    | Value        | ConstraintPressure    | Force from hard rules                            | [0,1] |
| 50    | Value        | NormInternalization   | Weight of learned norms                          | [0,1] |
| 51    | Value        | MoralTension          | Conflict between outcomes and rules              | [0,1] |
| 52    | Value        | Unused                | Reserved for future expansion                    | [0,1] |
| 53    | Meta         | SelfModel             | Accuracy of internal self-representation         | [0,1] |
| 54    | Meta         | MonitorLoad           | Cost of self-observation                         | [0,1] |
| 55    | Meta         | Drift                 | Distance from initial parameters                 | [0,1] |
| 56    | Meta         | ControlBalance        | Ratio of reflex to deliberation                  | [0,1] |

### Domain Groupings

**Existence (8D: indices 0-7)**
- Captures the agent's survival state and continuity expectations
- Critical for understanding self-preservation behavior
- RAM level maps directly to ResourceMargin

**Sensory (8D: indices 8-15)**
- Represents perceptual breadth and quality
- Vision data drives Awareness + Salience
- Lidar data drives Orientation + Noise (conflicting distance readings)

**Cognitive (8D: indices 16-23)**
- Models internal world knowledge
- Uncertainty rises when predictions fail (low Comprehension)
- Surprise magnitude drives UpdateDrive

**Strategic (8D: indices 24-31)**
- Long-term planning and commitment
- GoalClarity modulates action confidence
- Leverage correlates with threat/opportunity detection

**Tactical (8D: indices 32-39)**
- Immediate behavior modulation
- Exploration vs. Exploitation trade-off
- Execution quality affects motor output precision

**Social (8D: indices 40-47)**
- Models perception of other agents
- AgencyDetection drives action selection toward other agents
- Dominance assessment affects ATTACK vs. HUNT bias

**Value (5D: indices 48-52)**
- Scalar reward signal and norm alignment
- UtilitySignal integrates multiple reward sources
- MoralTension tracks rule violations

**Meta (4D: indices 53-56)**
- Self-reflection and meta-cognition
- Drift tracks parameter drift from initialization
- ControlBalance measures reflex vs. deliberation ratio

---

## 5. Visualization: Accessing Semantic State Dashboard

### Running the Dashboard

The semantic visualization dashboard is automatically served when a simulation starts with `headless=False`:

```python
from src.simulation.loop import Simulation
from src.config import BASE_CONFIG

sim = Simulation(config=BASE_CONFIG, headless=False)
results = sim.run()
```

This starts:
- **Waitress WSGI server** on `http://localhost:8080`
- **Semantic endpoint** at `http://localhost:8080/semantic/<agent_id>`

### Accessing the Dashboard

1. **Primary visualization:** `http://localhost:8080`
   - Real-time 3D visualization of agent positions, safe zone, RAM nodes
   - Neural metric graphs

2. **Semantic state view:** `http://localhost:8080/semantic`
   - 57D heatmap (8×8 grid visualization)
   - Domain summary stats
   - Per-domain breakdown (bar charts)
   - Reward trace (last 500 ticks)

### Dashboard Components

**Heatmap View (8×8 grid)**
- Each cell represents one semantic dimension
- Color indicates value: blue (low/0) → red (high/1)
- Opacity indicates confidence/magnitude
- Hover to see dimension name and value

**Domain Summary Stats**
- 8 stat cards, one per domain
- Shows mean activation per domain
- Quick overview of agent's "state of mind"

**Reward Trace Canvas**
- Line graph of reward over last 500 ticks
- X-axis: time (left=oldest, right=newest)
- Y-axis: reward magnitude
- Dashed line at 0 for reference

**Per-Domain Breakdown**
- Detailed 8-bar chart for each domain
- Height indicates activation level
- Color matches heatmap

### Query Specific Agent

```
http://localhost:8080/semantic?agent_id=0
```

Use URL parameter to switch between agents (requires simulation with multiple agents).

### API Endpoints

**Get Semantic State:**
```
GET /semantic/<agent_id>
Headers: X-API-Token: <token> (if security enabled)
Response:
{
  "tick": 42,
  "agent_id": 0,
  "semantic_state": [0.12, -0.34, ..., 0.89],  // 57 floats
  "ram": 75.5,
  "position": [50, 30, 10],
  "action": "MOVE"
}
```

**Get Reward Trace:**
```
GET /reward/<agent_id>
Response:
{
  "reward_trace": [
    {"tick": 0, "reward": 0.5},
    {"tick": 1, "reward": 0.6},
    ...
  ]
}
```

---

## 6. Integration Points in the Codebase

### Key Files

| File | Role | Semantic Integration |
|------|------|---------------------|
| `src/lrsg/semantic_encoder.py` | Encode sensors → 57D | Maps vision, lidar, IMU, state to semantic space |
| `src/lrsg/semantic_motor.py` | 57D → actions + motors | Maps semantic state to action bias and motor output |
| `src/simulation/loop.py` | Main simulation loop | Calls `semantic_encoder.forward()` each tick, feeds to LRSG |
| `src/agent/agent.py` | Agent entity | Stores `semantic_state` tensor, provides sensor dict to encoder |
| `src/simulation/visualizer.py` | Web dashboard | Serves semantic state via `/semantic/<agent_id>` endpoints |
| `docs/semantic_state_taxonomy.md` | Reference | Defines all 57 dimensions and their meanings |

### Data Flow in Code

```python
# In simulation/loop.py, each physics tick:

for agent in self.agents:
    # 1. Collect sensors
    sensor_dict = agent.sense_world(...)
    
    # 2. Encode to semantic state
    semantic_state = self.semantic_encoder.forward(sensor_dict)
    agent.semantic_state = semantic_state
    
    # 3. LRSG forward pass (includes semantic state)
    brain_output = agent.forward_pass(semantic_state)
    
    # 4. Semantic-motor mapping
    action_bias = self.semantic_motor_interface.semantic_to_action_bias(semantic_state)
    
    # 5. Select action and apply output modulation
    action = select_action_with_bias(action_bias, brain_output)
    motor_output = self.semantic_motor_interface.semantic_to_motor_output(
        semantic_state, action, brain_output
    )
    
    # 6. Execute action with motor parameters
    agent.act(action, motor_output)
```

---

## 7. Troubleshooting

### Dashboard Not Visible

**Problem:** `http://localhost:8080/semantic` shows blank page
- **Solution:** Ensure simulation started with `headless=False`
- **Check:** Look for "Waitress serving" message in console
- **Port:** If port 8080 in use, check `visualizer.py` for custom port

### Semantic State All Zeros

**Problem:** Semantic state is `[0, 0, ..., 0]`
- **Solution:** Check that agent sensors are populated
- **Debug:** Add print statements in `sensor_dict` construction
- **Verify:** Ensure `vision`, `lidar`, `imu` keys present in dict

### Action Bias Not Changing

**Problem:** Action bias identical across ticks
- **Solution:** Semantic state should vary with agent state
- **Check:** Verify RAM level changes (drives ResourceMargin)
- **Trace:** Print `semantic_state[3]` (ResourceMargin) across ticks

### Visualization API Token Error

**Problem:** "HTTP 401" when accessing `/semantic/<agent_id>`
- **Solution:** If security enabled, pass token via header:
  ```javascript
  fetch('http://localhost:8080/semantic/0', {
    headers: { 'X-API-Token': 'your-token-here' }
  })
  ```

---

## 8. Advanced: Custom Semantic States

To define custom semantic dimensions or modify the encoder:

### Extending the Encoder

```python
# In src/lrsg/semantic_encoder.py

class CustomSemanticEncoder(SemanticEncoder):
    def __init__(self, device='cpu'):
        super().__init__(device)
        # Add custom projections
        self.custom_fc = nn.Linear(32, 10)  # Custom 10D space
    
    def forward(self, sensor_dict):
        # Call parent forward to get base 57D
        base_semantic = super().forward(sensor_dict)
        
        # Compute custom features (e.g., threat level)
        threat_vector = self._compute_threat(sensor_dict)
        custom_embed = torch.tanh(self.custom_fc(threat_vector))
        
        # Blend custom into base (replace some dims or append)
        # Example: replace indices 48-52 (Value domain) with custom
        base_semantic[48:53] = custom_embed[:5]
        
        return base_semantic
```

### Custom Action Mapping

```python
# In src/lrsg/semantic_motor.py

class CustomSemanticMotorInterface(SemanticMotorInterface):
    def semantic_to_action_bias(self, semantic_state):
        # Call parent for baseline
        base_bias = super().semantic_to_action_bias(semantic_state)
        
        # Apply custom rules
        # Example: suppress ATTACK if reputation is high
        reputation = semantic_state[47]
        base_bias[0] *= (1.0 - reputation * 0.5)  # Reduce ATTACK bias by reputation
        
        return base_bias
```

---

## 9. References

- **Semantic State Taxonomy:** `docs/semantic_state_taxonomy.md`
- **Knowledge Base:** `CYNOSURE_KB.md` (Section 1.1 Architecture Map)
- **LRSG Forward Pass:** `src/lrsg/forward.py`
- **Agent Sensing:** `src/agent/agent.py` (method: `sense_world()`)
- **Simulation Loop:** `src/simulation/loop.py` (method: `physics_tick()`)

---

**Last updated:** June 22, 2026
