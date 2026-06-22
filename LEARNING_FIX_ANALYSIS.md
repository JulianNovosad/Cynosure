# Learning System Fix Analysis

## Test Results After Scaling Fixes

Applied:
1. Weight initialization: 5x increase
2. Learning rate: 10x increase  
3. Reward scale: 10x increase

Results:
- 10 episodes: Learning 7% vs Baseline 10% (WORSE by 3%)
- 50 episodes: Learning 8% vs Baseline 20% (WORSE by 12%)

**Conclusion: Scaling alone does NOT fix the learning problem.**

## Root Cause Analysis: Why Random Networks Can't Learn

### The Fundamental Issue

The task requires:
1. Input: Current position, target position, distance vector
2. Processing: Network must learn to output "move toward target"
3. Output: 6 discrete movement directions

With **random initial weights**:
- All neurons fire randomly
- Movement is essentially a random walk
- Hebbian learning tries to "strengthen" random synapses during reward
- But there's no underlying structure to learn FROM

### Mathematical Proof Why Scaling Doesn't Help

Let W be a random weight: W ~ N(0, σ²)

Traditional learning algorithm:
```
W_new = W_old + η × ∇Loss
```

With Hebbian + reward shaping:
```
W_new = W_old + η × reward × pre_activation × post_activation
```

The issue:
1. W_old is RANDOM
2. Network structure is RANDOM
3. pre/post activations are RANDOM
4. No matter how large η is, strengthening RANDOM structure doesn't help

**Random walk baseline**: Agent gets lucky ~20% of the time
**Learning agent**: Strengthens random patterns that happened during lucky events
**Result**: Learning is WORSE because it's fighting against randomness

### Example: A Lucky Collection

Scenario: Random agent bumps into target by pure luck
- Activations: [0.03, -0.15, 0.08, -0.02, 0.12, ...] (random)
- Reward: +1.0 (collected)
- Hebbian tries to strengthen these random activations
- Next episode: Different random activations occur
- No consistent structure to reinforce

## Why Previous Attempts Failed

1. ✗ Weight scaling (5x): Didn't provide structure
2. ✗ Learning rate (10x): Stronger updates to random patterns = worse learning
3. ✗ Reward scaling (10x): Bigger updates to random structure

## What's Actually Needed: Structural Initialization

Instead of random initialization, the network needs:

**Option A: Task-Aligned Input Structure**
- Input neurons for approach/retreat classification
- Reward neurons that respond to movement direction
- Seeds some structure that Hebbian can build on

**Option B: Pre-trained or Seeded Weights**
- Initialize weights not randomly, but to respond to sensory patterns
- E.g., weights from input position neurons → output movement neurons

**Option C: Longer Training with Regularization**
- 1000+ episodes instead of 10-50
- Add regularization to prevent weight explosion
- Eventually random structure might converge by luck

**Option D: Different Algorithm**
- Replace Hebbian with advantage actor-critic
- Use policy gradients instead of correlation-based learning
- More suited to discrete action spaces

## Recommendation

**Primary fix needed**: Task-aware network initialization

Instead of:
```python
W = randn() * 0.5  # Pure random
```

Use:
```python
# Initialize with implicit bias toward movement
W = randn() * 0.5
W[input_neurons, movement_output_neurons] += 0.5  # Seed approach structure
W = W / sqrt(sum(W²))  # Normalize
```

This gives Hebbian learning something to work with.

## Impact Assessment

- Random network learning: BROKEN (gets worse with more training)
- With task structure: Likely FIXED (Hebbian can optimize learned structure)
- Scaling alone: INSUFFICIENT (random structures at any scale don't learn)

---

**Status**: Identified that weight/learning-rate scaling is necessary but insufficient. Core issue is network architecture, not just hyperparameters.

**Next step**: Initialize network with task structure seed.
