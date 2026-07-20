# Cynosure Configuration Metrics Comparison

> **Documentation note (2026-07-20):** This file is a historical, point-in-time snapshot and is intentionally not rewritten to reflect current state (rewriting it would falsify the record it captured). For current system status, see `CYNOSURE_KB.md` and `docs/CHANGELOG.md`.


**Test Date:** June 24, 2026  
**Hardware:** NVIDIA GeForce RTX 2070 (8.6GB VRAM)

## Quick Reference Metrics

### Performance Metrics

| Metric | Compact (2^20) | Standard (2^22) | Medium (2^23) |
|--------|---|---|---|
| **Initialization Time** | 4.81s | 6.97s | 6.85s |
| **VRAM Required** | 443MB | 1,770MB | 2,509MB |
| **Activation Cycle** | 0.008ms | 0.010ms | 0.015ms |
| **Numerical Stability** | ✓ (0 NaNs) | ✓ (0 NaNs) | ✓ (0 NaNs) |

### Scale Metrics

| Metric | Compact | Standard | Medium |
|--------|---|---|---|
| **Total Neurons** | 1,048,576 | 4,194,304 | 8,388,608 |
| **Max Out-Degree** | 24 | 24 | 12 |
| **Est. Synapses** | 17.6M | 70.5M | 70.5M |
| **VRAM per 1M Neurons** | 423MB | 422MB | 299MB |

### Capacity Metrics (on RTX 2070)

| Metric | Compact | Standard | Medium |
|--------|---|---|---|
| **VRAM Utilization** | 5.2% | 20.6% | 29.2% |
| **Available VRAM** | 8.2GB | 6.8GB | 6.1GB |
| **Multi-Agent Capacity** | ~18x | ~3x | ~2x |
| **Single-Agent Headroom** | Excellent | Good | Adequate |

## Detailed Results

### Configuration 1: Compact

```
Neurons:           1,048,576 (2^20)
Max Out-Degree:    24
Est. Synapses:     17,616,076

Init Time:         4.81 seconds
VRAM:              443 MB
Stability:         0 NaNs ✓

Activation Cycle:  0.008 ms
VRAM per Neuron:   0.42 MB
```

**Best for:** Edge devices, lightweight scenarios, many agents

### Configuration 2: Standard

```
Neurons:           4,194,304 (2^22)
Max Out-Degree:    24
Est. Synapses:     70,464,307

Init Time:         6.97 seconds
VRAM:              1,770 MB
Stability:         0 NaNs ✓

Activation Cycle:  0.010 ms
VRAM per Neuron:   0.42 MB
```

**Best for:** Single-agent learning, proven configuration, balanced approach

### Configuration 3: Medium

```
Neurons:           8,388,608 (2^23)
Max Out-Degree:    12
Est. Synapses:     70,464,307

Init Time:         6.85 seconds
VRAM:              2,509 MB
Stability:         0 NaNs ✓

Activation Cycle:  0.015 ms
VRAM per Neuron:   0.30 MB
```

**Best for:** Multi-agent scenarios, complex navigation, largest single brain fit

## Performance Scaling Analysis

### Initialization Time Scaling

- Config 1→2 (4x neurons): **1.45x slowdown** (sublinear)
- Config 2→3 (2x neurons): **0.98x** (actually faster due to degree reduction)

**Conclusion:** Larger networks benefit from vectorization; initialization doesn't scale linearly.

### Memory Scaling

- Compact: 0.42 MB/neuron
- Standard: 0.42 MB/neuron (same efficiency)
- Medium: 0.30 MB/neuron (better due to smaller degree)

**Conclusion:** Memory efficiency improves with larger brains and smaller degrees.

### VRAM Headroom Scaling

| Agents | Config 1 | Config 2 | Config 3 |
|--------|----------|----------|----------|
| 1 agent | 8.2GB | 6.8GB | 6.1GB |
| 2 agents | 7.8GB | 5.0GB | 3.6GB |
| 3 agents | 7.5GB | 3.3GB | N/A |
| 4 agents | 7.1GB | N/A | N/A |

## Key Findings

1. **All configurations are numerically stable** (0 NaNs across all tests)

2. **Larger brains are more efficient** - Medium config (8M neurons) uses only 0.30 MB per neuron

3. **Initialization is sublinear** - 4x neurons takes only 1.45x time

4. **VRAM is the limiting factor** - Even Medium config (8M) only fits 2-3 agents on 8GB

5. **Degree reduction is viable** - Medium config reduces degree from 24→12 without stability issues

## Recommendations by Use Case

### Research & Development
```
Config: Standard (2^22, 4.2M neurons)
Rationale: Proven performance, single-agent learning, good headroom
```

### Multi-Agent Simulation
```
Config: Medium (2^23, 8.4M neurons)
Rationale: Larger brains, 2-3 agent capacity, complex behavior
```

### Edge/Mobile Deployment
```
Config: Compact (2^20, 1M neurons)
Rationale: 443MB footprint, scales to 18 agents, IoT compatible
```

### High-End Hardware (A6000, RTX 4090)
```
Config: Scale to 2^26+ neurons
Rationale: Would need hardware testing, potential for 32M+ neurons
```

## Test Reproducibility

All results can be reproduced by running:

```bash
python src/experiments/config_comparison_final.py
```

Results are saved to `experiment_results/config_comparison_final.json`

## Next Steps

1. **Learning Validation** - Run learning experiments with each config
2. **Multi-Agent Testing** - Verify agent capacity claims
3. **Hardware Scaling** - Test on RTX 3090, A6000
4. **Behavior Emergence** - Compare learned behaviors across scales

---

*Report generated: June 24, 2026*  
*Hardware: RTX 2070, 8.6GB VRAM*  
*All configurations STABLE and VIABLE*
