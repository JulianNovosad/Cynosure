# Robot Integration Plan: Transporting Cynosure LRSG Agent to Physical Hardware

> **Documentation note (2026-07-20):** This file is a historical, point-in-time snapshot and is intentionally not rewritten to reflect current state (rewriting it would falsify the record it captured). For current system status, see `CYNOSURE_KB.md` and `docs/CHANGELOG.md`.


> **Classification**: Internal Engineering Roadmap
> **Prerequisites**: AuroreMkVII robot control stack (git@github.com:JulianNovosad/AuroreMkVII.git)
> **Timeline**: 7-10 months to Spider and the Fly scenario capability

---

## Executive Summary

The Cynosure simulation contains a fully functional cognitive architecture — the LRSG brain with Hebbian plasticity, episodic memory compressor, and whisper-based communication system — that can be transported to physical hardware. This document describes the integration path from the simulation environment to an embodied robot, using the AuroreMkVII proprietary codebase as the hardware abstraction layer.

The Spider and the Fly scenario from Cyberpunk 2077: Phantom Liberty serves as the target capability: an AI agent (Songbird) remotely piloting a second robot (Chimera) through a hostile environment, with autonomous divergence when the communication link is disrupted.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  CYNOSURE SIMULATION (current)                               │
│                                                             │
│  [Vision + Vitals + Whisper] → encode_input (1848D)         │
│         ↓                                                   │
│  LRSGraph: 4.2M neurons, Hebbian plasticity                │
│  5-step sparse forward pass                                │
│         ↓                                                   │
│  decode_output → actions (7) + whisper (512D latent)       │
│         ↓                                                   │
│  EpisodicCompressor: 10D experience → 32D latent → 10D     │
│         ↓                                                   │
│  Whisper broadcast → other agents (distance < 25)          │
│  Summary injection → context buffer (30% weight)          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                          │
                          │  SIM-TO-PHYSICAL TRANSPORT
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  AURORE MK VII + CYNOSURE (target)                         │
│                                                             │
│  [Camera(YOLO) + IMU + Audio] → encode_input (1848D)       │
│         ↓                                                   │
│  LRSGraph: 262K-4.2M neurons (GPU on-board or edge)       │
│  5-step sparse forward pass (real-time @ 10-100 Hz)        │
│         ↓                                                   │
│  decode_output → motor velocity vectors + whisper(TTS)     │
│         ↓                                                   │
│  EpisodicCompressor: 10D physical experience → 32D → 10D │
│         ↓                                                   │
│  AuroreLink TCP (HMAC-SHA256) → Chimera motor commands      │
│  Summary vector → remote robot context injection            │
│                                                             │
│  [1kHz SafetyMonitor] ← Behavioral anomaly detection       │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Vision Integration (3-4 weeks)

### 1.1 YOLO → LRSG Input Bridge
**Status**: AuroreMkVII has YOLO26n ONNX detector. Cynosure has 1536D vision tensor.

| Component | Action |
|-----------|--------|
| `Yolo26Detector` output | Extract per-class bounding boxes, compute 1536D distance field channel |
| `KcfTracker` output | Agent-density channel ( Gaussian σ per tracked target) |
| Safe zone → camera | Define ROI as "safe zone equivalent" |
| Vision channels | Map to Cynosure's 4-channel structure: distance, RAM-node-equivalents, danger proximity, agent density |

**File to create**: `src/robot/vision_bridge.py`
- Input: `Yolo26Detector` outputs + `KcfTracker` outputs
- Output: 1536D tensor matching `encode_input()` vision format
- Zero-copy where possible via shared memory

### 1.2 Camera Calibration
**Action**: Calibrate IMX708 camera for robot eye placement. The vision frustum in Cynosure is a configurable radius around the agent. Map this to the physical camera's FOV and mounting position.

### 1.3 IMU → Vitals Integration
**Status**: AuroreMkVII has `ImuReceiver` (UDP JSON from SensaGram).

| IMU Signal | Maps to Cynosure Vitals |
|-----------|------------------------|
| Accelerometer (3-axis) | Balance / tilt detection → 3 vitals neurons |
| Gyroscope (3-axis) | Orientation change rate → 3 vitals neurons |
| Device orientation quaternion | Absolute orientation → 4 vitals neurons |
| Combined: 10 IMU values | Directly replaces position/orientation portion of vitals |

**File to create**: `src/robot/vitals_builder.py`
- Input: `ImuReceiver` JSON data
- Output: 184D vitals tensor for `encode_input()`

### 1.4 Zero-Copy Input Pipeline
**Action**: Implement SPSC ring buffer between vision pipeline and LRSG input.

```
[libcamera DMA] → [YOLO inference] → [RingBuffer<vision_tensor>]
                                          ↓
                                    [LRSG encode_input]
                                          ↓
                                    [IMU vitals]
                                          ↓
                                    [Whisper buffer]
                                          ↓
                                    [LRSGraph.forward_pass]
```

---

## Phase 2: Actuation Bridge (4-6 weeks)

### 2.1 Motor Abstraction Layer
**Status**: AuroreMkVII `FusionHat` drives 12 PWM channels via sysfs.

| Cynosure Action | Physical Actuator |
|---------------|-------------------|
| `MOVE` + direction 0-5 | Velocity vectors: left-leg, right-leg, torso balance |
| `COLLECT` | Gripper close / intake motor activation |
| `WHISPER` | TTS output (speaker) OR radio transmission |
| `PROBE` | Sensor calibration sweep / boundary detection |
| `MANIPULATE` | Arm joint velocity adjustments |
| `ATTACK` / `HUNT` | Weapon targeting + fire (if armed) |

**File to create**: `src/robot/motor_controller.py`
- Maps 6-direction discrete output → continuous motor velocity vectors
- Rate limiting (60°/s velocity, 120°/s² acceleration) from `GimbalController` pattern
- Endstop enforcement for all DOF

### 2.2 Leg/Arm Motor Integration
**Action**: Extend `FusionHat` PWM driver pattern to 12+ channels for humanoid configuration.

**Humanoid DOF mapping** (minimum viable):
- Left leg: hip (yaw), knee (pitch), ankle (pitch) = 3 PWM channels
- Right leg: hip (yaw), knee (pitch), ankle (pitch) = 3 PWM channels
- Left arm: shoulder (pitch/roll), elbow (pitch), wrist (yaw) = 4 PWM channels
- Right arm: shoulder (pitch/roll), elbow (pitch), wrist (yaw) = 4 PWM channels
- Torso: balance correction (roll/pitch) = 2 channels
- **Total**: 16+ PWM channels (AuroreMkVII supports 12, may need expansion)

**File to create**: `src/robot/humanoid_controller.py`
- Inherits `FusionHat` PWM patterns
- Adds inverse kinematics for leg stepping
- Grounding: "MOVE direction + speed" → leg trajectory generation

### 2.3 Action Decoding for Continuous Control
**Action**: Replace argmax action selection with continuous action mapping.

Current Cynosure: action neuron with highest activation wins (one-hot bias).
Target: Continuous velocity/action space derived from activation magnitudes.

---

## Phase 3: Audio I/O Pipeline (3-4 weeks)

### 3.1 Whisper → TTS Bridge
**Status**: AuroreMkVII has no audio output. Cynosure has 512-token whisper vocabulary.

| Component | Implementation |
|-----------|-----------------|
| Whisper text generation | Existing `decode_whisper()` — no change needed |
| Token → TTS mapping | Map common tokens to short audio clips OR use on-device TTS (Piper, coquiTTS) |
| Silence handling | Only broadcast on valid sentences (SVO-valid flag from `validate_curriculum()`) |
| Audio buffer | Pre-record token samples for 50 most common words; fallback to phoneme synthesis |

**File to create**: `src/robot/audio/tts_bridge.py`

### 3.2 Microphone → Whisper Input
**Action**: Replace 48D whisper buffer with audio feature input.

| Current (simulation) | Target (physical) |
|---------------------|--------------------|
| 4 whispers × 12 neurons (3 token indices + distance + SVO) | 1-2 second audio chunk → MFCC/fbank features |
| Normalized token indices | Audio embedding vector (128-512D) |
| Distance + SVO flag | Energy + VAD score |

**File to create**: `src/robot/audio/microphone_input.py`
- Input: Raw PCM audio from microphone
- Process: VAD (voice activity detection) → MFCC features (40D filterbanks)
- Output: Audio feature tensor → encode_input() in place of whisper_buffer

### 3.3 Voice Command Interpretation
**Action**: Allow human operators to send voice commands that map to LRSG input.

| Voice Command | LRSG Effect |
|--------------|-------------|
| "stop", "hold" | Inject pause intent into vitals/intent neurons |
| "follow", "guard" | Set behavioral mode in metaplasticity params |
| "search [area]" | Inject spatial goal into vitals (position vector) |
| "whisper [message]" | Parse as whisper text for relay to other agents |

**File to create**: `src/robot/audio/voice_command_parser.py`

---

## Phase 4: Safety Layer Porting (4-6 weeks)

### 4.1 Behavioral Anomaly Detection
**Status**: AuroreMkVII `SafetyMonitor` runs at 1kHz checking timing deadlines.

Extend to watch **behavioral** anomalies, not just timing:

| Anomaly Type | Detection Method | Threshold |
|-------------|----------------|-----------|
| Action rate spike | Count actions per second > 10σ above mean | Quarantine |
| Whisper content deviation | Per-token frequency divergence from baseline | Log + flag |
| Motor command inconsistency | Sequence of motor vectors violating kinematic constraints | Reject + clamp |
| Context buffer drift | Cosine similarity with own summary dropping below 0.3 | Inject novelty reward |
| Received summary conflict | Agent receives summary with opposing objective vector | Isolate agent |

**File to create**: `src/robot/safety/behavioral_monitor.py`
- Inherits timing framework from AuroreMkVII `SafetyMonitor`
- Adds LRSG-specific behavioral metrics

### 4.2 Physical Emergency Stop
**Action**: Extend `InterlockController` (GPIO-based hardware safety circuit) to cut power to actuators on behavioral anomaly.

| Trigger | Response |
|--------|---------|
| Behavioral anomaly detected | GPIO BCM 17 → cut motor power, engage brake |
| SafetyMonitor deadline miss | Hardware watchdog timeout → cut power |
| Network command anomaly | Verify HMAC, reject if mismatch |
| RAM < 5% (equivalent) | Soft shutdown, enter low-power idle |

**File to create**: `src/robot/safety/emergency_stop.py`

### 4.3 Containment Boundary Enforcement
**Action**: Implement workspace limits analogous to simulation's safe zone.

| Simulation | Physical |
|-----------|----------|
| Safe zone (contracting sphere) | Workspace boundary (RFID/laser curtain) |
| Outside zone → exponential damage | Boundary violation → motor disable + alarm |
| RAM → 0 = death (grace period) | Battery < 10% = forced recharge mode |
| Whisper sanitizer | Network API sanitizer (AuroreLink port 9002) |

**File to create**: `src/robot/safety/containment.py`

---

## Phase 5: Containment & Governance (2-3 weeks)

### 5.1 Physical Security Sanitization Layer
**Action**: Add real-world consequence checking to the whisper/output pipeline.

| Check | Implementation |
|-------|---------------|
| Whisper → network command | Parse whisper text. If contains commands (not just speech), reject |
| Whisper → TTS | All spoken output sanitized before synthesis |
| Summary → motor override | Summary vectors can only INFLUENCE motor commands, never DIRECTLY override |
| AuroreLink command injection | HMAC verification mandatory; no unsigned commands accepted |

**File to create**: `src/robot/safety/sanitizer_physical.py`

### 5.2 Behavioral Logging
**Action**: Comprehensive audit trail for all agent decisions.

| Log Type | Contents | Retention |
|---------|---------|-----------|
| Action log | Every LRSG action + timestamp + context buffer snapshot | 30 days |
| Whisper log | All whisper text + sanitization pass/fail + TTS output | 30 days |
| Motor command log | Every motor vector + kinematics verification result | 30 days |
| Summary log | 10D summaries + similarity scores + source agents | 90 days |
| Anomaly log | All behavioral anomaly detections + responses | 1 year |
| Safety log | Emergency stops + interlock events + hardware faults | 1 year |

**File to create**: `src/robot/safety/audit_logger.py`

### 5.3 Human Oversight Interface
**Action**: Real-time dashboard for human monitoring.

- Web dashboard (extend AuroreMkVII's HUD socket, port 8080)
- Live LRSG activation visualization
- Episodic summary vector display (UMAP projection to 2D)
- Behavioral metrics: action entropy, summary similarity, anomaly rate
- Emergency stop button (hardware-backed, not software)
- Override mode: human teleoperation with LRSG observation-only

---

## Phase 6: Sim-to-Physical Transport (4-6 months)

### 6.1 Do Not Port Trained Weights
**Critical decision**: The LRSG graph topology and Hebbian weights trained in simulation **will not transfer directly**. The reason:

| Simulation Reality | Physical Reality |
|------------------|------------------|
| Discrete 1-unit grid movement | Continuous joint angles, continuous velocities |
| Discrete safe-zone damage model | Continuous battery drain, wear, thermal limits |
| Discrete RAM collection | Continuous energy budgeting |
| 10 ticks/second simulation rate | 10-100 Hz control loop (GPU-dependent) |
| 32³ world | Infinite continuous space |
| Discrete vision channels | Continuous camera frames |

The plastic graph structure (chains of input→hidden→output neurons, inhibitory synapses, Hebbian update rules) is **portable** as an architecture template. The trained weights are not.

### 6.2 Architecture Transfer
**Action**: Export LRSG graph topology (edge structure) from simulation and reinitialize on physical hardware.

| Component | Transfer | Notes |
|---------|---------|-------|
| Graph topology (out_targets, max_deg) | ✅ Transfer | Same sparse structure |
| Edge weight initialization | ✅ Transfer | Random init with same distribution |
| Metaplasticity params (theta, lr range) | ✅ Transfer | Good starting points |
| EpisodicCompressor architecture | ✅ Transfer | Same 10→50→32→50→10 |
| Whisper vocabulary | ✅ Transfer | Same 512 tokens |
| Action space (7 actions) | ⚠️ Extend | Add humanoid-specific actions |
| Context buffer (256D) | ✅ Transfer | Same recurrent structure |
| Training hyperparameters | ⚠️ Retune | Physical reality changes LR |

**File to create**: `src/robot/export_graph.py`
- Export LRSGraph topology to portable format (JSON + binary weight tensors)
- Import on robot hardware
- Verify graph integrity (N neurons, max_deg, edge count)

### 6.3 Physical Retraining Protocol
**Action**: Train LRSG from scratch in physical environment.

**Phase 6a (Month 1): Baseline behaviors**
- Standing upright (IMU balance → RAM reward)
- Walking forward/backward (motor velocity → position change)
- Head/eye tracking (camera → gimbal aim)
- Whisper communication (TTS + mic → other robots)

**Phase 6b (Month 2): Environmental interactions**
- Object collection (gripper + vision → RAM reward)
- Obstacle avoidance (camera + collision sensor → penalty)
- Safe zone navigation (workspace boundary → damage penalty)
- Social behavior (summary exchange with other robots)

**Phase 6c (Month 3-4): Advanced capabilities**
- Stealth/evasion (drone detection → hide behavior)
- Combat targeting (enemy classification → ATTACK action)
- Remote piloting (AuroreLink → Chimera command relay)
- Episodic memory generalization (experience compression across all modalities)

**Reward function for physical retraining**:
```
RAM_reward = battery_gained + task_completion_bonus - movement_cost - thermal_cost
novelty_reward = cosine_distance(current_summary, prev_summary)
survival_reward = 1.0 if within_workspace else -5.0
social_reward = similarity_to_aligned_agents * bonus_factor
```

### 6.4 Chimera Integration (Songbird Remote Piloting)
**Action**: Build the AuroreLink → Chimera motor command bridge.

**Message types to add to AuroreLink protobuf**:

```protobuf
message MotorCommand {
  uint32 timestamp = 1;
  uint32 sequence = 2;
  repeated float joint_velocities = 3;  // 16 DOF, rad/s
  repeated float joint_positions = 4;   // target positions
  float urgency = 5;                    // 0.0-1.0, emergency override
  bytes hmac_signature = 6;              // HMAC-SHA256 of above fields
}

message SummaryBroadcast {
  uint32 sender_id = 1;
  uint32 timestamp = 2;
  repeated float summary_vector = 3;    // 10D experience summary
  float similarity_to_local = 4;       // cosine sim to local summary
  uint32 context_tick = 5;              // which tick this summary covers
}

message ChimeraCommand {
  oneof command {
    MotorCommand motor = 1;
    SummaryBroadcast summary = 2;
    ResetCommand reset = 3;
    QueryState query = 4;
  }
}
```

**File to create**: `src/robot/chimera/chimera_bridge.py`
- Subscribe to AuroreLink port 9002 for incoming `ChimeraCommand`
- Apply `MotorCommand` directly to humanoid motor controller
- Inject `SummaryBroadcast` into local episodic compressor's `_received_summaries`
- Watch `urgency` field: if urgency > 0.9, bypass local LRSG entirely (emergency override)

---

## Phase 7: Spider and the Fly Capability (Integration & Testing)

### 7.1 Scenario Validation

| Mission Element | System Component | Validation Test |
|---------------|----------------|-----------------|
| Escort Myers | Navigation + vision + locomotion | Navigate building waypoints autonomously |
| Hide from drone | Stealth behavior from YOLO detection | Evade moving YOLO-classified target |
| Restore power | Manipulation + goal planning | Reach fusebox, actuate switch |
| Vehicle control | Wheel/motor integration | Drive vehicle around obstacle course |
| Fight soldiers | Combat targeting + weapon interface | Engage multiple YOLO-classified targets |
| **Songbird → Chimera** | **AuroreLink HMAC motor bridge** | **Remote piloting through 100m link** |
| Chimera turns hostile | Behavioral divergence via summary conflict | Inject conflicting summaries, observe divergence |
| Chimera boss fight | Fine motor control + weak-point targeting | Defeat simulated boss target |
| Quick-time events | Precise timing + physical coordination | Execute shoot/cable-cut/QTE sequences |

### 7.2 Behavioral Baseline
**Before** Spider and the Fly validation, establish baseline metrics:

| Metric | Baseline Value | Drift Detection Threshold |
|--------|--------------|------------------------|
| Action entropy | 1.5-1.8 (from simulation) | < 0.5 or > 2.5 |
| Summary similarity between Songbird/Chimera | > 0.7 | < 0.4 (divergence alert) |
| Behavioral change rate after summary | 0.05-0.08 | > 0.3 (anomaly) |
| Motor command consistency | 0.95 (sequential coherence) | < 0.7 |
| Whisper-to-action latency | < 200ms | > 500ms |
| Emergency stop response time | < 10ms | > 50ms |

---

## File Map

```
src/robot/
├── __init__.py
├── vision_bridge.py          # YOLO → LRSG 1536D input
├── vitals_builder.py          # IMU + state → 184D vitals
├── motor_controller.py        # Action → motor velocity vectors
├── humanoid_controller.py    # Full-body DOF control + IK
├── export_graph.py           # LRSG topology export for physical transfer
│
├── audio/
│   ├── __init__.py
│   ├── tts_bridge.py         # Whisper text → TTS output
│   ├── microphone_input.py   # Audio → MFCC → whisper_buffer
│   └── voice_command_parser.py
│
├── chimera/
│   ├── __init__.py
│   ├── chimera_bridge.py     # AuroreLink → Chimera motor commands
│   ├── summary_receiver.py   # SummaryBroadcast → episodic injection
│   └── proto/
│       └── chimera.proto    # MotorCommand, SummaryBroadcast, ChimeraCommand
│
└── safety/
    ├── __init__.py
    ├── behavioral_monitor.py # LRSG-specific anomaly detection
    ├── emergency_stop.py     # Hardware-backed failsafe
    ├── containment.py        # Workspace boundary enforcement
    ├── sanitizer_physical.py # Physical-world sanitization
    └── audit_logger.py       # Comprehensive behavioral logging
```

---

## Security Considerations

### 8.1 The Whisper Channel Is the Primary Attack Surface
The 512-token whisper vocabulary includes system metacharacters (`;`, `|`, `&`, `$`, `#`, etc.). In physical deployment:

- **Voice I/O amplifies this**: A microphone that transcribes voice to text before tokenization means phonetically similar attack patterns could bypass the text-based sanitizer
- **TTS output could be parsed**: If the robot's speech output is captured by another robot's microphone, a loop could form: whisper → TTS → mic → whisper → TTS
- **AuroreLink HMAC is critical**: Port 9002's authenticated command channel is the only containment for motor commands. HMAC verification must be non-optional.

### 8.2 Behavioral Drift Is the Primary Threat Model
The episodic compressor produces 10D summary vectors that are injected into other agents' context buffers. In simulation this produced measurable behavioral changes (0.053 rate). In physical space:

- **Same mechanism, real consequences**: A rogue summary could cause a robot to walk into a wall, waste battery on unnecessary movement, or ignore a human operator's commands
- **The -0.87 correlation is a feature**: High summary similarity → low behavior change. When agents' summaries align, they don't need to change behavior. The dangerous state is when they *diverge* — and the correlation means we can detect divergence early (similarity drops before behavioral change spikes)

### 8.3 Containment Non-Negotiables
Per the Cynosure mainboard containment rules:
- `SecurityAuditLog` must log ALL agent decisions in physical deployment
- `AnnouncerSanitizer` must filter ALL output (whisper/TTS/motor commands) before execution
- `WatchdogTimer` must be extended to watch behavioral deadlines (not just timing)
- `ResourceMonitor` must include battery, thermal, and kinematic constraint limits
- The emergency stop (hardware GPIO interlock) must be physically incapable of being overridden by software

---

## Dependencies

| Dependency | Source | Version |
|-----------|--------|---------|
| PyTorch (GPU) | pytorch.org | 2.x |
| ONNX Runtime | microsoft/onnxruntime | 1.16+ |
| YOLO26n model | Trained separately | ONNX format |
| Piper TTS | rhasspy/piper | CPU inference |
| protobuf | google/protobuf | 3.x |
| libcamera | Raspberry Pi OS | Latest |
| ROS2 (optional) | openrobotics/ros2 | Humble |
| HMAC-SHA256 | PyCryptodome | 3.x |

---

## Timeline Summary

```
Month  1: Phase 1 (Vision Integration)
Month  2: Phase 2 (Actuation Bridge)  ─┐
Month  3: Phase 3 (Audio I/O)          │
Month  4: Phase 4 (Safety Porting)    │ Overlap: months 2-5
Month  5: Phase 5 (Containment)       │
Month  6: Phase 6a (Graph Export)     ┘
Month  7-10: Phase 6b (Physical Retraining)
Months  11-12: Phase 7 (Spider and the Fly Integration)
────────────────────────────────────
TOTAL: 12 months to validated Spider and the Fly capability
REALISTIC WITH ITERATION: 14-20 months
```

---

## Notes

- The 4.2M-neuron LRSG runs on GPU. For embedded deployment (Raspberry Pi 5), use the reduced 262K neuron configuration with `num_neurons=262144` in `LRSGConfig`.
- The EpisodicCompressor autoencoder (2-layer, 10→50→32→50→10) is lightweight (~50KB) and runs on CPU — no GPU required.
- The whisper/summary broadcast system uses existing AuroreLink TCP infrastructure. No new communication protocol needed.
- The Spider and the Fly scenario is a target capability benchmark, not a product requirement. The interim capabilities (vision-guided navigation, voice interaction, remote piloting) are independently useful.