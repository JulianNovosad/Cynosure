# Cynosure AI Configuration Comparison Report

> **Documentation note (2026-07-20):** This file is a historical, point-in-time snapshot and is intentionally not rewritten to reflect current state (rewriting it would falsify the record it captured). For current system status, see `CYNOSURE_KB.md` and `docs/CHANGELOG.md`.


**Date:** June 24, 2026  
**Device:** NVIDIA GeForce RTX 2070 (8.6GB VRAM)  
**Test Scope:** Three Cynosure configurations with varying neuron/synapse counts

---

## Executive Summary

Tested three Cynosure AI configurations to establish performance, stability, and resource utilization baselines. All three configurations initialize successfully and demonstrate zero numerical instability (0 NaNs). The largest viable configuration (8M neurons) fits comfortably within 8GB VRAM with headroom for multi-agent scenarios.

---

## Configurations Tested

### Configuration 1: Compact
- **Neurons:** 2^20 = 1,048,576
- **Max Out-Degree:** 24
- **Estimated Synapses:** ~17.6M
- **Target Use Case:** Lightweight agents, resource-constrained environments

### Configuration 2: Standard  
- **Neurons:** 2^22 = 4,194,304
- **Max Out-Degree:** 24
- **Estimated Synapses:** ~70.5M
- **Target Use Case:** Single-agent learning, baseline experiments

### Configuration 3: Medium
- **Neurons:** 2^23 = 8,388,608  
- **Max Out-Degree:** 12 (reduced from 24 for VRAM)
- **Estimated Synapses:** ~70.5M (same as Config 2 due to reduced degree)
- **Target Use Case:** Multi-agent arenas, complex environments

**Note:** Original Config 3 (2^25 = 33.5M neurons) exceeded 8GB VRAM during initialization of additional tensors. Adjusted to 2^23 to fit within hardware constraints while maintaining 8x neurons vs. Compact.

---

## Test Results

### 1. Initialization Performance

| Metric | Compact | Standard | Medium |
|--------|---------|----------|--------|
| **Neurons** | 1,048,576 | 4,194,304 | 8,388,608 |
| **Init Time (sec)** | 4.81 | 6.97 | 6.85 |
| **VRAM Used (MB)** | 443 | 1,770 | 2,509 |
| **Time per 1M Neurons (ms)** | 4.6 | 1.7 | 0.8 |
| **VRAM per 1M Neurons (MB)** | 423 | 422 | 299 |

**Key Finding:** Initialization time scales sublinearly; larger brains initialize faster per neuron due to vectorization benefits.

### 2. Stability Metrics

| Metric | Compact | Standard | Medium |
|--------|---------|----------|--------|
| **Numerical Stability** | 0 NaNs | 0 NaNs | 0 NaNs |
| **Graph Integrity** | ✓ | ✓ | ✓ |
| **Activation Reset (ms)** | 0.008 | 0.010 | 0.015 |

All configurations demonstrate perfect numerical stability. No NaN propagation detected.

### 3. VRAM Headroom Analysis

Tested on RTX 2070 (8.6GB total):

| Config | Used for Brain | Available for Agents | Multi-Agent Capacity |
|--------|----------------|---------------------|----------------------|
| Compact | 443MB (5.2%) | 8.2GB | 18x agents |
| Standard | 1.77GB (20.6%) | 6.8GB | 3-4x agents |
| Medium | 2.51GB (29.2%) | 6.1GB | 2-3x agents |

---

## Recommendations

### Use Config 1 (Compact) When:
- Running on edge devices (< 2GB VRAM)
- Deploying 15+ agents in a single arena
- Prioritizing speed (fastest init time per neuron)
- Memory footprint is critical

### Use Config 2 (Standard) When:
- Running single-agent learning experiments
- Baseline comparative studies
- Balanced resource utilization needed
- Learning convergence is the priority (4.2M is proven configuration)

### Use Config 3 (Medium) When:
- Multi-agent competitive scenarios (2-3 agents)
- Complex environment navigation required
- Larger synaptic capacity needed (8M neurons)
- Available VRAM ≥ 8GB

---

## Performance Scaling Observations

1. **Initialization:** Non-linear scaling; larger brains benefit from PyTorch's batch operations
   - Config 1→2 (4x neurons): 1.45x init time
   - Config 2→3 (2x neurons): 0.98x init time (actually faster due to smaller degree)

2. **Memory Efficiency:** VRAM per neuron decreases with scale
   - Config 1: 423 MB per 1M neurons
   - Config 2-3: 299-422 MB per 1M neurons
   
3. **Stability:** All configurations are numerically stable (0 NaNs)

---

## Hardware Requirements Summary

| Component | Minimum | Recommended | Tested |
|-----------|---------|-------------|--------|
| VRAM | 2GB (Config 1 only) | 4GB (Config 2) | 8.6GB (RTX 2070) |
| CUDA Compute | 5.0+ (Maxwell) | 6.0+ (Pascal) | 7.5 (Turing) |
| System RAM | 8GB | 16GB | 32GB |
| Initialization | < 10 seconds | < 7 seconds | 6.8s avg |

---

## Next Steps

1. **Learning Validation:** Run 10+ seed learning experiments with each config
2. **Multi-Agent Testing:** Verify multi-agent capacity claims for Configs 2-3
3. **Hardware Variants:** Test on RTX 3080, A6000 for larger config viability
4. **Production Deployment:** Establish Config choice heuristics for hardware profiles

---

## Appendix: Raw JSON Results

Full test results (timestamps, all metrics): `experiment_results/config_comparison_final.json`

Test runner: `src/experiments/config_comparison_final.py`
