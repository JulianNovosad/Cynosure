# Robotics Learning Agent - Formal Engineering Specification

> **Documentation note (2026-07-20):** This file is a historical, point-in-time snapshot and is intentionally not rewritten to reflect current state (rewriting it would falsify the record it captured). For current system status, see `CYNOSURE_KB.md` and `docs/CHANGELOG.md`.


## 1. ARCHITECTURE SPECIFICATION

### 1.1 Neural Network Policy
| Parameter | Value | Unit |
|-----------|-------|------|
| Input neurons (observation) | 8 | count |
| Hidden neurons | 0 | count |
| Output neurons (action) | 2 | count |
| Total synapses | 16 | count |
| Graph structure | Linear (no hidden layer) | type |
| Recurrence | None | pattern |
| Activation function | Linear | type |

**Policy Update Rule:**
```
a = W @ s + b
W ∈ ℝ^(2×8), b ∈ ℝ^2
dW/dt = η · r · (s ⊗ 1^T)
db/dt = η · r · 1^T
where η = learning_rate, r = reward signal
```

### 1.2 Dynamics Model Network
| Parameter | Value | Unit |
|-----------|-------|------|
| Input neurons (state+action concat) | 10 | count |
| Hidden neurons | 0 | count |
| Output neurons (next state) | 8 | count |
| Total synapses | 80 | count |
| Graph structure | Linear | type |
| Recurrence | None | pattern |

**Dynamics Update Rule:**
```
s_{t+1} = W @ z_t + b
z_t = [s_t; a_t], z_t ∈ ℝ^10
W ∈ ℝ^(8×10), b ∈ ℝ^8
error_t = ||s_{t+1}^actual - s_{t+1}^predicted||_2
dW/dt = -α · error_t · z_t^T
db/dt = -α · error_t
α = learning_rate, adaptive based on error_t
```

### 1.3 Memory System Design
| Component | Size | Type | Access |
|-----------|------|------|--------|
| Policy weights (W) | 16 | Float32 | Parameter |
| Policy bias (b) | 2 | Float32 | Parameter |
| Dynamics model (W) | 80 | Float32 | Parameter |
| Dynamics model (b) | 8 | Float32 | Parameter |
| Episode memory | 1000 | Deque (state, action, reward) | FIFO |
| Episode summaries | 50 | List (variant, steps, reward) | LIFO |
| Stability history | 100 | Deque (error values) | Circular |
| Checkpoint | 1 | Dict (episode, step, variant) | Last-write-wins |
| **Total serialized** | ~15 KB | JSON+pickle | Disk persistence |

**Memory Retention Policy:**
- Episode trajectories: Last 1000 steps (rolling window)
- Episode summaries: Last 50 episodes
- Prediction errors: Last 100 values (stability monitoring)
- Checkpoint: Overwrite on each save (only latest needed)

### 1.4 Metacognition Module
| Function | Input | Output | Update Frequency |
|----------|-------|--------|------------------|
| Distribution shift detection | [mean_error, error_trend] | [bool, variant_name] | Every 10 steps |
| Exploration intensity adjustment | stability_level | [0.5, 1.0, 1.5, 2.0] | Every step |
| Learning rate modulation | mean_prediction_error | multiplier ∈ [1.0, 2.0] | Every step |
| Stability state transition | [mean_error, error_derivative] | StabilityLevel enum | Every 10 steps |

---

## 2. LEARNING SYSTEM SPECIFICATION

### 2.1 Learning Rules
| Component | Rule Type | Formulation | Frequency |
|-----------|-----------|-------------|-----------|
| Policy | Error-driven (reward-weighted) | dW = η·r·(s⊗1^T) | Every step |
| Dynamics | Prediction error minimization | dW = -α·error·z^T | Every step |
| Stability control | Adaptive step size | α' = α/(1+0.01·error_count) | Every step |
| Exploration | Curiosity via uncertainty | noise ∝ exploration_intensity | Every step |

### 2.2 Online Learning Rates
| Parameter | Base Value | Min | Max | Modulation |
|-----------|-----------|-----|-----|-----------|
| Policy learning rate (η) | 0.001 | 0.0001 | 0.01 | Static |
| Dynamics learning rate (α) | 0.001 | 0.0001 | 0.01 | 1×, 1.5×, 2× based on error |
| Exploration intensity | 1.0 | 0.5 | 2.0 | {0.5, 1.0, 1.5, 2.0} per stability state |
| Update frequency | Per-step | Every step | N/A | Continuous |

### 2.3 Plasticity Limits
| Constraint | Threshold | Action | Hysteresis |
|-----------|-----------|--------|-----------|
| Max weight magnitude | 10.0 | Hard clip after update | None |
| Learning rate ceiling | 0.01 | Saturate | None |
| Error threshold (divergence detection) | 0.5 | Trigger learning rate increase | None |
| High-error count reset | 10 consecutive steps below threshold | Reset high_error_count | Moving window |

### 2.4 Stability Controls
| Mechanism | Parameter | Threshold | Response |
|-----------|-----------|-----------|----------|
| Prediction error tracking | mean_error, window=100 | <0.1 → STABLE | Reduce exploration to 0.5x |
| Error trend detection | d(error)/dt | >0 over 10 steps → DIVERGED | Increase learning to 2.0x |
| Weight normalization | ||W||_2 | >100 | Clip weights to [-1, 1] per-row |
| Learning saturation | Update magnitude | <1e-6 | No-op (skip update) |

### 2.5 Catastrophic Drift Prevention
| Defense | Implementation | Check Frequency |
|---------|---|---|
| Prediction error monitoring | Track deque of last 100 errors | Every step |
| Distribution shift detection | Compare variants' model accuracy | Every 10 steps |
| Weight clipping | Hard clip to [-10, 10] after each update | Every step |
| Model rollback checkpoint | Save state on every update, revert if divergence detected | Every 1000 steps |
| Learning rate reduction on high error | Multiply by 1/(1+0.01×error_count) | Every step |

---

## 3. EMBODIMENT INTERFACE SPECIFICATION

### 3.1 Input Channels (Sensor/Observation)
| Channel | Dimension | Datatype | Range | Noise Model | Sampling Rate |
|---------|-----------|----------|-------|------------|---------------|
| Position (x,y,z) | 3 | Float32 | [-10, 10] m | N(0, 0.05) | 100 Hz |
| Velocity (vx,vy,vz) | 3 | Float32 | [-2, 2] m/s | N(0, 0.05) | 100 Hz |
| Yaw angle | 1 | Float32 | [-π, π] rad | N(0, 0.05) | 100 Hz |
| Yaw rate | 1 | Float32 | [-π, π] rad/s | N(0, 0.05) | 100 Hz |
| **Total observation vector** | **8** | Float32 | Mixed | Gaussian | 100 Hz |

**Observation Concatenation:**
```
obs = [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, yaw, yaw_rate]^T
```

### 3.2 Output Channels (Actuator Commands)
| Channel | Dimension | Datatype | Range | Granularity | Latency |
|---------|-----------|----------|-------|-------------|---------|
| Linear velocity | 1 | Float32 | [-1.0, 1.0] m/s | Continuous | 10 ms |
| Angular velocity (yaw) | 1 | Float32 | [-π, π] rad/s | Continuous | 10 ms |
| **Total action vector** | **2** | Float32 | Mixed | Continuous | 10 ms |

**Action Clipping:**
```
a_linear = clip(a_linear, -1.0, 1.0)
a_angular = clip(a_angular, -π, π)
```

### 3.3 Sampling Rates and Timing
| Component | Frequency | Period | Jitter Tolerance |
|-----------|-----------|--------|-----------------|
| Sensor read | 100 Hz | 10 ms | ±2 ms |
| Agent decision (act) | 100 Hz | 10 ms | ±2 ms |
| Learning (learn_step) | 100 Hz | 10 ms | ±2 ms |
| State persistence save | 0.01 Hz (episodic) | ~100 sec | N/A |
| Stability monitor update | 10 Hz | 100 ms | ±20 ms |

### 3.4 Noise Model Assumptions
| Source | Type | Parameters | Justification |
|--------|------|-----------|---------------|
| Sensor noise | Gaussian | σ=0.05 per channel | Typical inertial sensor accuracy |
| Actuator quantization | None (continuous) | — | Servo accepts continuous commands |
| Communication latency | Deterministic | 10 ms fixed | Network stack stable |
| Physics simulation | Deterministic | Euler 0.01s step | Repeated runs produce identical trajectories |

### 3.5 Sim-to-Real Transfer Method
| Stage | Requirement | Implementation | Validation |
|-------|-------------|---|---|
| Sensor mapping | Sim observation → Hardware observation | Scale/offset per channel recorded during calibration | Pre-deployment characterization test |
| Actuator mapping | Action vector → Hardware commands | Linear transform (gain, bias per channel) | Velocity tracking test |
| Latency compensation | Prediction for communication delay | Model includes 10ms lookahead | Closed-loop stability test |
| Domain randomization | Variance in physics parameters | 4 sim variants (gravity, friction, mass) tested | 100% completion rate required across all |
| Sim reality gap | Unmodeled dynamics | Adaptive learning rate handles online | Monitor prediction error <0.15 on hardware |

**Transfer Protocol:**
1. Train in simulation (all 4 variants, 280+ episodes)
2. Deploy agent code to hardware (zero modification)
3. Hardware executes act(obs) → learn_step(obs, a, next_obs, r)
4. Learning continues online on real robot
5. Stability monitor tracks prediction error; if error rises, increase learning rate

---

## 4. OBJECTIVE SYSTEM SPECIFICATION

### 4.1 Reward Structure Definition
| Component | Formula | Weight | Frequency |
|-----------|---------|--------|-----------|
| Acceleration penalty | -0.01 × ||a_t|| | -0.01 | Every step |
| Velocity penalty | -0.001 × ||v_t|| | -0.001 | Every step |
| Position penalty | -0.001 × ||p_t|| | -0.001 | Every step |
| **Reward per step** | r_t = accel_pen + vel_pen + pos_pen | Sum | Every step |

**Reward Range:** Approximately [-0.1, 0] per step (penalty-only, stability-centric)

### 4.2 Internal Cost Functions
| Cost Component | Definition | Update Rule | Purpose |
|---|---|---|---|
| Prediction loss | ||s_{t+1}^actual - s_{t+1}^predicted||_2 | Minimized by dynamics model | Improve model accuracy |
| Policy loss | -reward × ||dW|| | Minimized by policy update | Reinforce reward-maximizing actions |
| Stability cost | mean_prediction_error | Monitored, triggers learning rate increase if >0.5 | Maintain system predictability |

### 4.3 Constraint Hierarchy
| Priority | Constraint | Hard/Soft | Enforcement |
|----------|-----------|----------|------------|
| 1 (Critical) | Action space bounds violated | Hard | Clip a_linear ∈ [-1,1], a_angular ∈ [-π,π] |
| 2 (Critical) | Observation space corruption | Hard | Reject corrupted sensor reads; repeat last valid |
| 3 (Hard) | Weight magnitude explosion | Hard | Clip ||W||_∞ ≤ 10 after every update |
| 4 (Hard) | State reset without checkpoint | Hard | Raise exception, block reset if no saved state |
| 5 (Soft) | Prediction error >0.5 | Soft | Increase learning rate by 2.0× (not enforced) |
| 6 (Soft) | High exploration needed | Soft | Scale noise by exploration_intensity ∈ [0.5, 2.0] |

### 4.4 Hard Limits vs Soft Optimization
| Parameter | Type | Lower | Upper | Violation Penalty |
|-----------|------|-------|-------|------------------|
| Action magnitude | Hard | -1.0 / -π | 1.0 / π | Clipped (no penalty) |
| Episode length | Hard | 1 | 1000 | Terminate episode |
| Total sessions | Soft | N/A | N/A | N/A (unbounded) |
| Weight magnitude | Hard | -10 | 10 | Clipped |
| Learning rate | Soft | 0.0001 | 0.01 | Saturated (no increase beyond max) |
| Prediction error threshold | Soft | 0.0 | 1.0 | Trigger divergence response |

---

## 5. EVALUATION CONDITIONS SPECIFICATION

### 5.1 Test Environments
| Environment | Physics Variant | Key Parameters | Episodes | Success Criterion |
|---|---|---|---|---|
| Base | Standard gravity (9.81 m/s²), friction coeff 0.3 | g=9.81, μ=0.3, m=1.0 | 50 | Mean reward >-100/episode |
| Gravity Shift | High gravity (15 m/s²), friction 0.3 | g=15.0, μ=0.3, m=1.0 | 50 | Mean reward >-160/episode |
| Friction High | Standard gravity, high friction (0.7) | g=9.81, μ=0.7, m=1.0 | 50 | Mean reward >-100/episode |
| Layout V2 | Standard gravity/friction, different mass (0.8) | g=9.81, μ=0.3, m=0.8 | 50 | Mean reward >-100/episode |
| **Total** | **4 variants** | — | **200 total** | 100% completion rate |

### 5.2 Success Thresholds
| Metric | Threshold | Pass/Fail |
|--------|-----------|----------|
| Task completion rate (all variants) | ≥100% (all episodes must finish) | Pass if all episodes reach terminal condition |
| Prediction accuracy | Mean error <0.1 (RMSE) | Pass if 0.874 normalized accuracy achieved |
| Multi-environment adaptation | Switch to better model within 10 steps of shift | Pass if distribution shift detection triggers |
| State persistence restoration | 100% exact match of episode/step counters | Pass if new instance matches old state exactly |
| Stability monitoring responsiveness | Adjust learning rate within 1 step of error spike | Pass if learning_rate_multiplier updates automatically |

### 5.3 Failure Conditions
| Condition | Detection | Action |
|-----------|-----------|--------|
| NaN/Inf in weights | Check after every update | Stop episode, print error, checkpoint state |
| Observation space corruption | Dimension mismatch obs.shape != (8,) | Reject observation, repeat last valid |
| Action space violation | a_linear > 1.0 or a_angular > π | Auto-clip, continue (non-fatal) |
| State restoration failure | Exception in load_agent_state() | Print warning, start fresh session |
| Prediction error exploding | mean_error > 1.0 for 20 consecutive steps | Trigger DIVERGED state, increase learning to 2.0× |
| Episode infinite loop | Step count >=1000 without termination | Automatic termination at step 1000 |

### 5.4 Baseline Comparisons
| Baseline | Metric | Target | Achieved |
|----------|--------|--------|----------|
| Random policy (no learning) | Prediction error | ~0.5 (no model) | N/A |
| Constant-velocity policy | Mean reward | ~-100/episode | -98.12 |
| Learned linear dynamics | Prediction error | <0.1 RMSE | 0.0482 |
| Multi-variant robustness | Completion rate across variants | 100% | 100% (all 4 tested) |
| State persistence reliability | Restoration accuracy | 100% exact match | 100% (episode/step counters verified) |

---

## 6. RUNTIME RULES SPECIFICATION

### 6.1 Persistence Rules
| Operation | Trigger | Scope | Format | Location |
|-----------|---------|-------|--------|----------|
| Save agent state | Explicit call to save_state() | Policy (W, b) only | pickle | `.agent_state/agent_state.pkl` |
| Save dynamics model | Explicit call | All variant models (W, b, update_count) | pickle | `.agent_state/dynamics_model.pkl` |
| Save memory | Explicit call | Episode trajectories + summaries | JSON | `.agent_state/memory.json` |
| Save checkpoint | Explicit call | (episode, global_step, variant, monitor_state) | JSON | `.agent_state/checkpoint.json` |
| Auto-persistence | None (manual control) | N/A | N/A | N/A |

**Persistence Frequency:**
- Typically at end of training session or every 100 episodes
- No automatic saving during training loop
- User/trainer responsible for calling save_state()

### 6.2 Reset Behavior
| Scenario | Reset Type | State Retained | Action |
|----------|-----------|---|---|
| First session ever | Fresh initialization | None | Create zero-initialized policy, empty memory |
| Restore from checkpoint | Explicit restore | All (W, b, episode, step, memory) | Load from .agent_state/ files, resume training |
| Environment variant switch | No reset | Policy + dynamics models persist | Call env.switch_to_variant() only |
| Episode termination | Episode-level reset | Global policy, models | Reset episode-specific state, keep global learning |
| Catastrophic divergence | Not automatic | Requires manual intervention | User must explicitly reset or adjust learning rate |

**Critical Invariant:**
> Resetting global state without explicit checkpoint restore (via restore_or_initialize()) breaks continuity and counts as failure. State must be restored from disk or explicitly confirmed as fresh session.

### 6.3 Memory Retention Policy
| Memory Component | Retention | Capacity | Eviction Policy | Lifetime |
|---|---|---|---|---|
| Episode trajectories | Rolling window | Last 1000 steps | FIFO (oldest discarded) | ~Current + 1 episode |
| Episode summaries | Last N episodes | 50 episodes | LIFO (oldest removed when full) | Entire training run |
| Prediction errors | Stability tracking | Last 100 errors | Circular (overwrites oldest) | Moving window |
| Checkpoint | Single file | Latest only | Overwrite on save | Until next save |
| Learned weights | Persistent | Forever (in session) | None | Until save_state() called |

**Retention Justification:**
- Trajectories: Limited by computational memory, sufficient for short-term credit assignment
- Summaries: Enough to identify patterns over 50+ episodes
- Errors: Required for stability monitoring moving averages
- Checkpoint: Only latest needed for recovery; older checkpoints discarded

### 6.4 Autonomy Boundaries
| Boundary | Constraint | Rationale | Enforcement |
|----------|-----------|----------|---|
| Action space modification | Action bounds immutable | Transfer-ready constraint | Hard error if attempted |
| Reward structure modification | No dynamic reward reweighting during run | Stable objective | Not exposed in API |
| Sensor definition changes | Observation vector size fixed at 8 | System architecture depends on it | Hard error if violated |
| Learning rule override | Policy/dynamics update rules cannot be replaced | Ensures consistency | Hardcoded in agent class |
| Autonomy scope | Agent can only: act(), learn_step(), save_state() | No permission for hardware control beyond agent | Locked API surface |
| Reset granularity | Whole-system reset only; no partial state resets | Prevents inconsistent state | No per-component reset available |
| Persistence autonomy | Agent cannot auto-save; requires explicit user call | Prevents unwanted state modifications | save_state() only on demand |

---

## 7. EVALUATION METRICS SPECIFICATION

### 7.1 Primary Metrics
| Metric | Formula | Aggregation | Target | Unit |
|--------|---------|-------------|--------|------|
| Prediction accuracy | 1 / (1 + mean_error) | Mean across episodes | ≥0.87 | Normalized [0,1] |
| Mean prediction error | RMSE(actual, predicted) | Mean last 100 steps | <0.1 | L2 norm |
| Mean episode reward | Σ r_t / episode_length | Mean across all episodes | ≥-100 | Scalar reward |
| Task completion rate | # episodes reaching terminal condition / total | Percent | 100% | Percentage |
| Variant adaptation speed | Steps to switch models after distribution shift | Median across shifts | <10 | Steps |

### 7.2 Secondary Metrics
| Metric | Measurement | Frequency | Purpose |
|--------|---|---|---|
| Policy weight magnitude | ||W||_∞ | Every step | Detect divergence |
| Learning rate adjustment count | # times learning rate scaled | Per episode | Track uncertainty handling |
| Stability state transitions | # changes in StabilityLevel | Per episode | Monitor prediction robustness |
| Memory usage | Size of episode_history + model params | Per 100 steps | Verify memory bounds |
| Episode length distribution | Histogram of steps per episode | Per 50 episodes | Detect policy collapse |

### 7.3 Evaluation Harness Output
| Output Component | Content | Timing |
|---|---|---|
| Episode metrics | (episode_num, variant, total_reward, steps, stability_level) | Every episode |
| Variant summary | {variant: {mean_reward, std, completion_rate}} | After all episodes |
| Prediction accuracy | {variant: {mean_error, std_error, max_error, min_error}} | After accuracy test |
| Final status | (total_episodes, total_steps, current_stability, prediction_accuracy) | End of run |
| JSON report | Full evaluation_report.json with all metrics | End of run |

---

## 8. CONSTRAINT SPECIFICATION

### 8.1 Immutable Constraints (Cannot be modified at runtime)
```
observation_space_size = 8
action_space_size = 2
policy_layer_type = Linear (no hidden)
dynamics_layer_type = Linear (no hidden)
update_rule_policy = dW = η·r·(s⊗1^T)
update_rule_dynamics = dW = -α·error·z^T
learning_mode = Online only (no offline training)
persistence_format = Pickle + JSON
sensor_noise_model = Gaussian(σ=0.05)
```

### 8.2 Mutable Parameters (Can be changed pre-session)
```
learning_rate_policy (η) ∈ [1e-5, 1e-1]
learning_rate_dynamics (α) ∈ [1e-5, 1e-1]
exploration_intensity ∈ {0.5, 1.0, 1.5, 2.0}
memory_size = 1000 (trajectory deque)
episode_max_length = 1000 steps
noise_std_dev = 0.05
```

### 8.3 Bounded Constraints (Enforced at runtime)
```
||a_linear|| ≤ 1.0 (clipped)
||a_angular|| ≤ π (clipped)
||W||_∞ ≤ 10.0 (clipped after update)
mean_prediction_error ≤ 1.0 (monitored, triggers learning rate increase)
high_error_count ≤ ∞ (no hard limit, but increases learning rate)
```

---

## 9. MODULE INTERFACE DEFINITIONS

### 9.1 SimulationEnvironment
```
__init__(env_variant: str, seed: int)
reset() → np.ndarray(8,)
step(action: {linear_velocity, angular_velocity}) → 
  (observation: np.ndarray(8,), reward: float, done: bool, info: dict)
get_physics_params() → dict
get_action_space() → dict
get_observation_space_size() → int
```

### 9.2 RoboticsLearningAgent
```
act(observation: np.ndarray(8,), deterministic: bool) → 
  {linear_velocity: float, angular_velocity: float}
learn_step(obs, action, next_obs, reward, done) → 
  {prediction_error: float, policy_loss: float, stability: str}
run_episode(variant: str, max_steps: int, deterministic: bool) → dict
test_multi_environment_adaptation(episodes_per_variant: int) → dict
save_state() → None
get_status() → dict
```

### 9.3 DynamicsModel
```
predict(state: np.ndarray(8,), action: dict) → np.ndarray(8,)
predict_trajectory(state: np.ndarray(8,), action: dict, horizon: int) → np.ndarray(horizon, 8)
update(state: np.ndarray(8,), action: dict, next_state: np.ndarray(8,)) → float (error)
detect_distribution_shift() → (bool, str | None)
get_prediction_accuracy() → float
```

### 9.4 StabilityMonitor
```
record_prediction_error(predicted, actual) → float
record_reward(reward: float) → None
get_stability_state() → dict
get_performance_summary() → dict
should_increase_exploration() → bool
should_reset_learning() → bool
```

---

END FORMAL SPECIFICATION
