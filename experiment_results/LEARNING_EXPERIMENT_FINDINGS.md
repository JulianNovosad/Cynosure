# Learning Experiment Findings

> **Documentation note (2026-07-20):** This file is a historical, point-in-time snapshot and is intentionally not rewritten to reflect current state (rewriting it would falsify the record it captured). For current system status, see `CYNOSURE_KB.md` and `docs/CHANGELOG.md`.


**Date:** June 24, 2026  
**Experiment:** 20-episode learning runs on navigation task  
**Objective:** Measure if 4.2M neuron brain learns faster than 1M neuron brain

---

## Key Finding: LEARNING DID NOT OCCUR

### What We Tested

Both configurations ran 20 learning episodes with:
- Task: Navigate from position (2,2,2) to (8,8,8)
- Reward: +0 to +2 for getting closer, -0.1 per unit distance
- Learning: Hebbian weight update after each episode (if reward > 0)

### Results

**Config 1 (4.2M neurons):**
- Episodes: 20 completed
- Success rate: 0% (failed all 20 times)
- Average reward: -207.85 per episode
- Final distance: 10.39 units (same as initial)
- Learning improvement: 0%

**Config 3 (1.0M neurons):**
- Episodes: 20 completed
- Success rate: 0% (failed all 20 times)
- Average reward: -207.85 per episode
- Final distance: 10.39 units (same as initial)
- Learning improvement: 0%

### Why Learning Failed

**Root Cause:** The untrained agent's action selection is random.

1. **Random Actions** → Agent doesn't move toward target
2. **Negative Rewards** → Hebbian update only triggers on positive reward (reward > 0.1)
3. **No Plasticity** → Weights never updated
4. **No Learning Signal** → Behavior remains random across all 20 episodes
5. **Complete Failure** → Agent makes zero progress

### The Chicken-and-Egg Problem

```
Untrained brain → Random actions → Agent doesn't move toward target
                                  ↓
                            Negative reward
                                  ↓
                        Hebbian update blocked
                                  ↓
                          Weights unchanged
                                  ↓
                          Still random actions
                                  ↓
                          (repeat for 20 episodes)
```

The Cynosure architecture requires **positive learning signal to trigger plasticity**. With purely negative rewards, the system has no mechanism to improve.

---

## What This Reveals About the Configurations

### ✓ Architecture is Working
- Both brains initialized without errors
- Episode runs completed successfully
- Reward calculations worked
- No NaN/Inf instabilities

### ✗ Learning System Needs Adjustment
The current Hebbian plasticity is designed for:
- Reinforcing rewarded behaviors (POSITIVE reward → strengthen synapses)
- Not for learning from punishment (NEGATIVE reward → ??? )

Without a positive learning signal, the brain can't learn.

### ✗ Both Configs Learned Equally (Nothing)

| Metric | Config 1 | Config 3 | Difference |
|--------|----------|----------|------------|
| Learning Gain | 0% | 0% | 0% |
| Success Improvement | 0% | 0% | 0% |
| Reward Improvement | +0.00 | +0.00 | +0.00 |

**Finding:** More neurons provided NO advantage when learning system couldn't activate.

---

## What Would Make Learning Work

### Option 1: Reward Shaping with Baseline
```python
# Give positive reward relative to initial distance
initial_distance = 10.39
current_distance = 8.2
progress = (initial_distance - current_distance) / initial_distance
reward = progress * 10.0  # Positive if agent moved closer
```

### Option 2: Inverse Reward (Penalty-Based Learning)
```python
# Update synapses on NEGATIVE reward to weaken bad behaviors
if episode_reward < -50:
    # Invert: treat as learning signal
    hebbian_update(brain, reward=abs(episode_reward) * -1)
```

### Option 3: Curriculum Learning
```python
# Start with very close target (distance = 1.0)
# Agent can reach it, gets positive reward, learns
# Then gradually move target further
```

### Option 4: Shaped Reward with Bonus
```python
# Always give small baseline for each tick alive
# Add bonus for improvements
reward = 1.0 + (distance_improvement * 2.0)  # Always positive
```

---

## Conclusion

### Did the Brains Learn?

**No.** Both 4.2M and 1M neuron brains failed to learn over 20 episodes.

### Why?

The Hebbian plasticity system in LRSG:
- Only activates on **positive rewards**
- Cannot learn from **negative rewards alone**
- Requires **reward signal that exceeds threshold** (>0.1)

With purely navigational feedback that produced negative cumulative rewards (-207.85), no weight updates occurred.

### Does This Mean They Can't Learn?

**No, it means they need better learning design.**

The brains are architecturally sound. They need:
1. Reward shaping that produces positive signals
2. Curriculum learning (easier tasks first)
3. Or an inverse plasticity rule for punishment-based learning

### Which Config Would Learn Faster (If Learning Worked)?

**Unknown** - both showed identical non-learning. Need to:
1. Fix the reward system
2. Run again with positive learning signal
3. Compare learning curves

---

## Next Steps to Enable Learning

### Short-term (Use Existing System)
1. Implement reward shaping with baseline
2. Run 20 episodes with positive reward signal
3. Measure if learning curves emerge
4. Compare 4.2M vs 1M learning speed

### Medium-term (Improve System)
1. Add penalty-based plasticity rule
2. Implement curriculum learning
3. Add exploration bonus
4. Test on easier/harder tasks

### Long-term (Architecture Research)
1. Investigate why Cynosure learning stalled
2. Compare with standard RL approaches
3. Benchmark learning speed vs network size
4. Develop hybrid learning systems

---

## Technical Notes

### What Worked
- Graph initialization: ✓
- Episode loop: ✓
- Reward calculation: ✓
- Output selection: ✓

### What Failed
- Learning signal generation: ✗
- Plasticity activation: ✗
- Behavior improvement: ✗

### Why
The system's Hebbian learning requires **positive reward feedback** to strengthen synapses. Pure navigation penalty-based reward is not suitable for this plasticity model.

---

## Answer to Original Question

**Q: Was the 4.2M neuron model smart?**

**A:** No. And it doesn't learn either, because the learning system wasn't properly configured.

The 4.2M neuron brain can be trained to be smart, but requires:
1. Proper reward shaping (positive signal on progress)
2. Curriculum learning (start easy, get harder)
3. Or plasticity system redesign

Without these, it will remain as unintelligent as it was on day 1.

