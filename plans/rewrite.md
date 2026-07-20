# Cynosure Rewrite Plan — Streaming Cluster Brain

> **Documentation note (2026-07-20):** This file is a historical, point-in-time snapshot and is intentionally not rewritten to reflect current state (rewriting it would falsify the record it captured). For current system status, see `CYNOSURE_KB.md` and `docs/CHANGELOG.md`.


## Core architecture

Robot → encoded state stream → brain server → action stream → robot

The robot is a thin sensorimotor shell. All inference and learning live on the server.

---

## Robot side (thin shell only)

**Hardware: JulianNovosad/AuroreMkVII** — all robot-side code is written in and against that repo. Do not implement standalone robot code here.

Hardware inventory (from AuroreMkVII spec.md):
- **Camera:** Raspberry Pi Camera Module 3 (Sony IMX708, MIPI CSI-2), 1536×864 resolution, 120 fps, accessed via libcamera
- **Laser rangefinder:** UART LRF, boresighted with primary camera
- **Soundboard:** Fusion HAT+ onboard speaker/microphone (audio I/O via SunFounder Fusion HAT+, I2C address 0x17)
- **Servos:** 3 active (channel 0: elevation gimbal, channel 1: azimuth gimbal, channel 2: interlock); hardware capacity is 12-channel PWM (16-bit, 50 Hz) via Fusion HAT+
- **Motor outputs:** 4-channel motor drivers on Fusion HAT+ — **motor control software does not yet exist in AuroreMkVII; it must be added there as part of this project**
- **Compute:** Raspberry Pi 5, I2C on GPIO2/GPIO3

The robot handles three things and nothing else:

- sensor capture + fixed preprocessing
- motor/servo command execution
- state packet transmission at 30–60 Hz

**State packet contents:**
- camera frame → low-dim feature vector via pretrained frozen encoder (libcamera → encoder, no CNN training on device, no gradient)
- rangefinder → scalar distance reading (meters, float32)
- servo positions → normalized float vector (3 active channels)
- soundboard input → optional ambient audio feature (if used)

No inference. No memory. No learning logic on device.

A minimal hardcoded reflex layer handles safety cutoffs (overcurrent, servo limit, rangefinder proximity stop) with sub-10ms latency independent of server round-trip.

---

## Server side (cluster brain)

The server receives state packets, runs the cluster swarm, and returns a motor action vector per tick.

### Cluster spec

- neurons per cluster: 2K–8K
- edges per cluster: 20K–80K (sparse only, no dense fallback)
- edge storage: (source uint16, target uint16, weight float16) = 6 bytes per edge
- neuron state: activation float16 + recurrent state float16 = 4 bytes per neuron
- eligibility trace: float16 accumulator per edge = 2 bytes per edge
- total per cluster: ~500KB–1.2MB

### Scale targets

- active clusters in VRAM: 4–12
- total stored clusters: 10K–50K (disk or RAM)
- active VRAM footprint: ~6–15 MB cluster data + ~200MB compute buffers
- stays well within 8GB on RTX 2070

### Execution mode

Sparse only. No dense 4096×4096 blocks.

Rationale: dense matrices fix memory cost at matrix size regardless of actual activity. Sparse scales with edge count, which you control directly. One representation, no fallback path.

### Scheduler

Event-driven. At each tick:

1. receive state packet → convert to input activations
2. select active clusters based on incoming signal strength (priority queue, top-N by total incoming weight × signal)
3. load clusters into VRAM if not resident
4. run sparse matmul + recurrent update + gating per cluster
5. propagate outputs through cross-cluster routing table
6. evict least-recently-active clusters
7. send motor action vector to robot

Cluster shape is fixed and uniform so the scheduler treats all clusters as interchangeable units. No special-casing.

---

## Learning rule

Reward-modulated Hebbian with eligibility traces. No backprop. No STDP.

### Mechanism

- each synapse holds a short eligibility trace (float16, decays over ~200–500ms)
- on each tick, trace increments when pre- and post-synaptic neurons co-activate
- trace decays multiplicatively each step (decay constant tunable)
- when reward signal arrives (from simulation outcome), apply:
  - `Δw = lr × trace × reward` for all synapses in active clusters
  - strengthen co-active edges, weaken inactive edges
  - apply global weight decay baseline to prevent unbounded growth

### Reward signal

Scalar broadcast from simulation or robot outcome. Not per-neuron. Not per-layer.

The reward signal is the only global operation. Everything else is local to the cluster.

---

## Simulation strategy

Train in simulation first. Do not test on hardware until sim policy is stable.

Simulation must match robot interface exactly:
- identical sensor pipeline (same encoder, same packet format)
- camera simulation at 1536×864, encoder output matches real Camera Module 3 pipeline
- rangefinder simulated as scalar raycast distance
- same 30–60 Hz tick rate (robot transmits at this rate; camera runs at 120 fps, subsampled for state packets)

Sim-to-real transfer works because the interface is frozen. The cluster brain sees the same packet format in sim and on hardware.

---

## What to keep vs replace

### Keep

| Component | Reason |
|---|---|
| Simulation environment (`src/simulation/`, `src/arena/`) | Reuse as-is, freeze sensor/action format |
| Robot interface layer | Already thin, just standardize packet format |
| Evaluation suite | Metrics stay valid across architecture change |

### Replace

| Component | Replace with |
|---|---|
| Global graph model | Cluster runtime (sparse edge lists per cluster) |
| Current learning system (LRSG) | Hebbian + eligibility traces + reward broadcast |
| Execution scheduler | Event-driven cluster scheduler (priority queue) |

### Freeze these interfaces (do not change once set)

- sensor packet format (field names, dtypes, order)
- action vector format (dimension, scale, meaning)
- cluster activation API (input: activation vector → output: action vector)

Everything underneath these interfaces can change without breaking the robot, sim, or tests.

---

## Robot repo: JulianNovosad/AuroreMkVII

All robot-side code lives in that repo. This Cynosure repo owns the server-side cluster brain and simulation only.

Gaps to fill in AuroreMkVII before hardware testing:
- Motor control software (4-channel Fusion HAT+ motor driver interface) — does not exist yet
- State packet serialization/transmission (camera encoder output + rangefinder + servo positions → UDP/TCP to brain server)
- Action packet deserialization + motor/servo command dispatch

Interfaces between repos are the frozen boundary: sensor packet format and action vector format must match exactly between AuroreMkVII (robot side) and Cynosure (server side).

---

## Implementation order

1. Define and freeze packet formats (sensor in, action out) — agreed interface between AuroreMkVII and Cynosure
2. Add motor control software to AuroreMkVII (Fusion HAT+ 4-channel motor driver, 3-servo command dispatch)
3. Build single-cluster runtime: sparse matmul + Hebbian update + eligibility trace (Cynosure)
4. Build cluster scheduler: load/evict, priority queue, routing (Cynosure)
5. Wire scheduler to simulation environment (Cynosure sim matches AuroreMkVII sensor/action formats)
6. Validate learning: agent must improve on a simple navigation task over 1K episodes
7. Scale cluster count and run full evaluation suite
8. Port to robot hardware via AuroreMkVII — interface already matches sim by construction
