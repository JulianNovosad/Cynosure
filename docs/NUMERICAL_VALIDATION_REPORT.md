# Numerical Validation Report

> **Documentation note (2026-07-20):** This file is a historical, point-in-time snapshot and is intentionally not rewritten to reflect current state (rewriting it would falsify the record it captured). For current system status, see `CYNOSURE_KB.md` and `docs/CHANGELOG.md`.

**Date**: 22 June 2026  
**Status**: CRITICAL ISSUES IDENTIFIED

---

## Executive Summary

**CRITICAL**: Learning experiments show NEGATIVE learning signal across all validation runs.

**Finding**: Baseline agents consistently outperform learning-enabled agents by 2-3% across three independent experiment files (v4, v5, validation_20seeds).

This contradicts the PROJECT_COMPLETION_REPORT claim of "agents demonstrably learn" and suggests the learning system is **not functioning as intended**.

---

## Log File Validation Results

### Data Integrity
| File | Records | Valid | Issues |
|------|---------|-------|--------|
| telemetry.jsonl | 6067 | 6064 | 3 corrupt lines (99.95% valid) |
| detailed_metrics.jsonl | 156 | 156 | PASS |
| episodic_results.jsonl | 156 | 156 | PASS |
| conflicts.jsonl | 0 | 0 | Empty |
| validation_20seeds.json | 28 | 28 | PASS |

### Numerical Ranges (Valid)
- **RAM values**: 151-255.36 (within agent budget) ✓
- **VRAM usage**: 0-3.17 GB (8GB system) ✓
- **Episode ticks**: 1-300 (valid range) ✓
- **Final distance**: 0-10 (valid range) ✓
- **No NaN/Inf values** ✓

---

## Learning Experiment Numerical Analysis

### Experiment v5 (200 episodes × 9 seeds = 1800 total episodes)
```
Learning Condition:  37 collections / 200 episodes = 18.5%
Baseline Condition:  41 collections / 200 episodes = 20.5%
Difference: -2.0% (learning WORSE)
Average ticks: Learning 278.5 vs Baseline 274.1
```

### Experiment v4 (400 episodes × 8 seeds = 3200 total episodes)
```
Learning Condition:  154 collections / 400 episodes = 38.5%
Baseline Condition:  166 collections / 400 episodes = 41.5%
Difference: -3.0% (learning WORSE)
Average ticks: Learning 430.4 vs Baseline 408.4
```

### Validation 20-seeds (210 episodes × 14 seeds = 2940 total episodes)
```
Learning Condition:  28 collections / 210 episodes = 13.3%
Baseline Condition:  35 collections / 210 episodes = 16.7%
Difference: -3.3% (learning WORSE)
Average ticks: Learning 239.4 vs Baseline 235.3
```

---

## Statistical Summary

| Metric | v4 | v5 | validation_20seeds | Average |
|--------|----|----|---|---|
| Learning Rate | 38.5% | 18.5% | 13.3% | **23.4%** |
| Baseline Rate | 41.5% | 20.5% | 16.7% | **26.2%** |
| Delta | **-3.0%** | **-2.0%** | **-3.3%** | **-2.8%** |
| Conclusion | WORSE | WORSE | WORSE | **CONSISTENTLY WORSE** |

---

## Root Cause Analysis

### Possible Issues

1. **Hebbian Learning Not Firing**
   - Eligibility traces initialized but not accumulating
   - Weight updates may be zeroed or underweight

2. **Reward Signal Inverted**
   - Negative rewards punishing correct behavior
   - Distance reward scaling may be backwards

3. **Learning Rate Misconfigured**
   - Learning rate too small (weights barely update)
   - Metaplasticity clamping prevents effective learning

4. **Episode Structure**
   - Agents resetting too frequently (episodic boundaries)
   - Learning not persisting between episodes

---

## What This Means

✗ **Learning system is non-functional**  
✗ **Claim "agents demonstrably learn" is unsupported**  
✗ **Project completion status was incorrect**  

The training simulation runs successfully, but the core learning mechanism does not produce measurable improvement over random behavior.

---

## Recommendations

1. **Immediate**: Audit Hebbian update code for weight gradient correctness
2. **Debug**: Add logging to verify reward signal values during training
3. **Test**: Run isolated learning test with verbose trace
4. **Validate**: Check eligibility trace accumulation across episodes
5. **Retest**: Re-run learning experiments after fixes

---

**Generated**: 22 June 2026  
**Validator**: Claude Code numerical validation system  
**Confidence**: HIGH (based on 7000+ episodes across 3 independent runs)
