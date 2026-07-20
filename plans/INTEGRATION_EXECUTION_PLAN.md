# Integration Execution Plan - Cynosure v0.4.1

> **Documentation note (2026-07-20):** This file is a historical, point-in-time snapshot and is intentionally not rewritten to reflect current state (rewriting it would falsify the record it captured). For current system status, see `CYNOSURE_KB.md` and `docs/CHANGELOG.md`.

**Created**: 2026-06-23  
**Status**: Framework Complete, Tests In Progress  
**Goal**: Complete 7-phase integration checkpoint for production release

---

## SUMMARY OF WORK COMPLETED

### ✓ Phase 1: Dependency Mapping (COMPLETE)
**Artifacts created:**
- BUILD_MANIFEST.md (Section 1) - Complete module dependency graph
- Module connectivity matrix (41 modules catalogued)
- Execution path verified: no circular dependencies, no orphans

**Key findings:**
- 23 modules actively used in main execution path
- 5 stub modules (robot bridges, I/O adapters) for future implementation
- 1 empty module (metrics/__init__.py)
- Full sensor→brain→actuator data flow documented

**Status**: ✓ SIGNED OFF

---

### ✓ Phase 2: Version Locking (COMPLETE)
**Artifacts created:**
- BUILD_MANIFEST.md (Section 2) - Core module hashes
- Version lock pinned: PyTorch 2.11.0+cu128, CUDA 12.8, SM_75
- 9 core module SHA256 hashes (first 16 chars)
- base_graph.pt snapshot hash: dee2d8bf29f9437a (906.1MB)
- Full hardware config locked (RTX 2070, 8.6GB VRAM)

**Key findings:**
- Build is reproducible on identical hardware
- Can pin exact dependency versions via requirements.txt + pyproject.toml
- Weight snapshots ready for versioning control

**Status**: ✓ SIGNED OFF

---

### ✓ Phase 3: Capability Coverage (IN PROGRESS - Framework Ready)
**Artifacts created:**
- tests/test_capability_validation.py - 7 independent capability tests
- Each test: 5-15 min execution + logging to capability_validation.jsonl

**7 Capabilities to validate:**
1. **Navigation** - 5 agents reach distant targets in safe zone
2. **Resource Management** - 10 agents survive contracting zone (>50%)
3. **Adaptation** - Learning system improves collection rates
4. **Communication** - Agents emit coherent whispers (>20% SVO density)
5. **Conflict Resolution** - Combat mechanics work (>5 conflicts in 1K ticks)
6. **Language Grounding** - PWM predicts characters (>60% accuracy)
7. **Semantic Awareness** - 57D semantic state drives actions

**Next step to execute:**
```bash
# Quickest path: run lightweight tests (disable TinyLlama to save 5 min per test)
python tests/test_capability_validation.py --disable-announcer
```

**Expected result**: 7/7 PASS (all capabilities functional)

**Estimated time**: 30-60 min total

---

### ⏳ Phase 4: Learning Saturation (READY TO EXECUTE)
**Framework**: `run_learning_experiment.py` + `analyze_learning_experiment.py` (already exist)

**Test parameters**:
- 20 random seeds (experiment reproducibility)
- 20 episodes per seed
- 300 ticks per episode
- 5K neurons per agent
- Distance-based reward shaping

**Expected outcome**:
- Collection rate improvement: +15% ± 5% (2nd half vs. 1st half)
- Statistical significance: p < 0.01 (paired t-test)
- Effect size (Cohen's d): > 0.8 (large)
- Comparison: vs. baseline (no learning) showing mean 0% change ± 10%

**Next step to execute:**
```bash
python run_learning_experiment.py \
  --seeds 20 \
  --episodes 20 \
  --ticks 300 \
  --neurons 5000 \
  --output experiment_results/learning_v6_final.json

# Analyze results
python analyze_learning_experiment.py \
  --input experiment_results/learning_v6_final.json
```

**Estimated time**: 1-2 hours on RTX 2070

---

### ⏳ Phase 5: Transfer Validation (FRAMEWORK READY, IMPLEMENTATION PENDING)
**Current status**: Skeleton code exists (vision_bridge.py, motor_controller.py)

**3-part roadmap:**

**Part 1: Vision Bridge Implementation** (1-2 hours)
```python
# src/robot/vision_bridge.py
class VisionBridge:
    def __init__(self, yolo_model="yolov8s"):
        self.model = YOLO(yolo_model)
    
    def capture_yolo(self) -> torch.Tensor:
        """Capture frame from camera, run YOLO, return 1536D embedding"""
        frame = camera.read()  # Stub: actual camera integration
        detections = self.model(frame)
        return self.embed_yolo_output(detections)  # → 1536D
```

**Part 2: Motor Controller Implementation** (1-2 hours)
```python
# src/robot/motor_controller.py
class MotorController:
    def __init__(self, turret_port="/dev/ttyUSB0"):
        self.serial = SerialPort(turret_port)
    
    def execute_action(self, action_idx: int) -> None:
        """Convert discrete action (0-6) to PWM motor commands"""
        pwm_vector = self.action_to_pwm(action_idx)
        self.serial.write(pwm_vector)
```

**Part 3: Hardware Validation** (1-2 days)
```bash
# Load sim-trained weights onto Aurore turret
python deploy_to_hardware.py \
  --weights base_graph.pt \
  --device /dev/ttyUSB0 \
  --test-scenarios scenario_approach.json

# Compare sim vs. hardware success rates
python compare_sim_hardware.py \
  --sim-results telemetry.jsonl \
  --hw-results hardware_telemetry.jsonl
```

**Current blocker**: NONE (stubs acceptable for v0.4.1 release, hardware deployment is post-release)

---

### ⏳ Phase 6: Regression Testing (FRAMEWORK READY)
**Framework**: `tests/test_regression_suite.py` (4 regression tests)

**Test cases:**
1. 2-agent, 200-tick benchmark (target ≤30ms/tick, baseline 25.1ms)
2. 5-agent, 1000-tick benchmark (target ≤40ms/tick, baseline 33.5ms)
3. NaN stability check (1000 ticks, no NaNs)
4. Telemetry schema validation (all fields present)

**Next step to execute:**
```bash
python tests/test_regression_suite.py

# Expected output: regression_results.json
# Look for: REGRESSION LOCK PASSED (4/4 tests)
```

**Estimated time**: 30-45 min on RTX 2070

**Can run in parallel with Phase 3-4** (independent tests)

---

### ⏳ Phase 7: Final Integration Statement (BLOCKED)
**Unblocks when:**
- [x] Phase 1 ✓ (dependency mapping complete)
- [x] Phase 2 ✓ (version locking complete)
- [ ] Phase 3 (capability coverage: awaiting 7/7 PASS)
- [ ] Phase 4 (learning saturation: awaiting p < 0.01)
- [ ] Phase 5 (transfer validation: stubs acceptable)
- [ ] Phase 6 (regression: awaiting 4/4 PASS)

**Final sign-off document**: DEPLOYMENT_CHECKLIST.md (Section 7.2)

**Estimated time to completion**: After Phase 3-4 complete (~3-5 hours)

---

## RECOMMENDED EXECUTION SEQUENCE

### Option A: Fastest Path (Minimal Release)
```
Time: ~1 hour
├─ Phase 3: Capability validation (30-60 min)
│  └─ Result: 7/7 PASS → MINIMAL RELEASE AUTHORIZED
└─ Can proceed immediately to hardware deployment
```

### Option B: Standard Path (Full Release)  
```
Time: ~2-3 hours
├─ Phase 3: Capability validation (30-60 min) [parallel]
├─ Phase 4: Learning convergence (1-2 hours) [parallel]
├─ Phase 6: Regression testing (30-45 min) [parallel]
└─ Phase 5: Hardware integration (1-2 weeks) [sequential]
   → Result: Full release ready for production
```

### Option C: Maximum Assurance (Extended Release)
```
Time: ~2-3 weeks
├─ Phases 1-6: Full validation (as above)
├─ Phase 5: Complete hardware integration
│  ├─ Vision bridge implementation
│  ├─ Motor controller implementation  
│  └─ Hardware equivalence testing
└─ Final sign-off (Phase 7)
   → Result: Production-deployed system
```

---

## CRITICAL SUCCESS METRICS

| Phase | Metric | Target | Status |
|-------|--------|--------|--------|
| 1 | Dependency coverage | 100% modules connected | ✓ 41/41 |
| 2 | Version reproducibility | Single build hash | ✓ dee2d8bf29f9437a |
| 3 | Capability pass rate | 7/7 tests PASS | ⏳ Running |
| 4 | Learning significance | p < 0.01 | ⏳ Pending 20-seed run |
| 5 | Sim-to-hardware gap | <10% success rate diff | ⏳ Pending implementation |
| 6 | Regression pass rate | 4/4 tests PASS | ⏳ Pending execution |

---

## QUICK START COMMANDS

**Start fresh validation from this point:**
```bash
cd C:\Cynosure

# Phase 3 (Capabilities) - Quickest to get a result
echo "=== Phase 3: Capability Validation ==="
python tests/test_capability_validation.py --disable-announcer

# Phase 6 (Regression) - Good to run in parallel
echo "=== Phase 6: Regression Testing ==="
python tests/test_regression_suite.py

# Phase 4 (Learning) - Long-running, suitable for background
echo "=== Phase 4: Learning Convergence ==="
python run_learning_experiment.py --seeds 20 --episodes 20 --ticks 300 --neurons 5000
```

**After tests complete:**
```bash
# Analyze results
echo "=== Capability Results ==="
cat capability_validation.jsonl | jq '.passed' | sort | uniq -c

echo "=== Regression Results ==="
cat regression_results.json | jq '.[] | .test, .passed'

echo "=== Learning Results ==="
python analyze_learning_experiment.py --input experiment_results/learning_v6_final.json
```

---

## KNOWN ISSUES & WORKAROUNDS

**Issue 1: TinyLlama loading is slow (5+ minutes)**
- **Symptom**: Capability tests stall during announcer initialization
- **Workaround**: Pass `--disable-announcer` flag to skip LLM loading
- **Permanent fix**: Use `use_announcer: bool = False` in config

**Issue 2: GPU memory spikes at test transitions**
- **Symptom**: OOM error on 8GB VRAM when running multiple tests back-to-back
- **Workaround**: Run tests individually or in separate Python processes
- **Permanent fix**: Add explicit torch.cuda.empty_cache() between tests

**Issue 3: Telemetry files accumulate (may exceed 1GB)**
- **Symptom**: Regression tests slow down after multiple runs
- **Workaround**: Delete telemetry.jsonl before each test run
- **Script**:
```bash
for f in telemetry.jsonl agent_whispers.jsonl conflicts.jsonl; do
  [ -f "$f" ] && rm "$f"
done
```

---

## SUCCESS CRITERIA FOR EACH PHASE

### Phase 3: All 7 Tests PASS
```json
{
  "capability": "Navigation",
  "passed": true,
  "metrics": {"agents_alive": 5, "ticks_completed": 500}
}
```
(Similarly for all 7)

### Phase 4: Statistical Significance
```
t-test result: t=2.092 (or higher), p < 0.01
Cohen's d: 0.8+ (large effect)
Mean improvement: +15% ± 5%
```

### Phase 6: All 4 Tests PASS
```json
[
  {"test": "2-agent-200-tick", "passed": true, "current_ms": 27.3},
  {"test": "5-agent-1000-tick", "passed": true, "current_ms": 36.8},
  {"test": "nan-stability", "passed": true, "has_nan": false},
  {"test": "telemetry-schema", "passed": true, "fields_per_entry": 12}
]
```

---

## NEXT IMMEDIATE ACTION

1. **Wait for Phase 3 tests to complete** (currently running in background)
   - Monitor: `capability_validation.jsonl`
   - Look for: `"passed": true` on all 7 entries

2. **Once Phase 3 passes**: Run Phase 4 (learning experiment)
   ```bash
   python run_learning_experiment.py --seeds 20 --episodes 20 --ticks 300 --neurons 5000 &
   ```

3. **In parallel**: Run Phase 6 (regression)
   ```bash
   python tests/test_regression_suite.py
   ```

4. **After all complete**: Generate final sign-off
   ```bash
   cat DEPLOYMENT_CHECKLIST.md | grep "PHASE 7"
   ```

---

**Last Updated**: 2026-06-23  
**Framework Status**: Complete and functional  
**Ready to proceed**: YES  
**Estimated completion time**: 2-3 hours (Phases 3-6) + 1-2 weeks (Phase 5 optional)

For questions or issues, contact: julian.novosad1@gmail.com
