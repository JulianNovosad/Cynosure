import torch
import math
from src.lrsg.graph import LRSGraph
from src.config import LRSGConfig


def update_structural_neurogenesis(graph: LRSGraph, src_indices: torch.Tensor, slot_indices: torch.Tensor):
    """
    Spatially constrained neurogenesis (Vectorized).
    """
    if src_indices.numel() == 0:
        return
    
    device = graph.device
    num_births = src_indices.numel()
    
    # Candidate sampling to avoid O(N) probability matrix
    num_candidates = 256
    
    # 1. Sample candidates
    # Half local (same block), half random
    block_size = 100
    src_blocks = graph.block_ids[src_indices]
    
    local_offsets = torch.randint(0, block_size, (num_births, num_candidates // 2), device=device)
    local_candidates = (src_blocks.unsqueeze(1) * block_size + local_offsets) % graph.N
    
    random_candidates = torch.randint(0, graph.N, (num_births, num_candidates // 2), device=device)
    candidates = torch.cat([local_candidates, random_candidates], dim=1) # [num_births, num_candidates]
    
    # 2. Compute probabilities for candidates
    src_pos = graph.positions[src_indices].unsqueeze(1) # [num_births, 1, 3]
    dst_pos = graph.positions[candidates.view(-1)].view(num_births, num_candidates, 3)
    
    dists = torch.norm(dst_pos - src_pos, dim=2)
    
    cand_blocks = graph.block_ids[candidates.view(-1)].view(num_births, num_candidates)
    c_ij = (src_blocks.unsqueeze(1) == cand_blocks).float()
    
    alpha, beta = 0.8, 0.1
    gamma, delta = 0.1, 0.5
    probs = alpha * torch.exp(-beta * dists) * c_ij + \
            gamma * torch.exp(-delta * dists) * (1.0 - c_ij)
    
    # 3. Sample from candidates
    target_indices_in_cand = torch.multinomial(probs + 1e-9, 1).squeeze(1)
    dst_indices = candidates[torch.arange(num_births, device=device), target_indices_in_cand]
    
    # 4. Update graph
    graph.out_targets[src_indices, slot_indices] = dst_indices.to(torch.int32)
    
    sigma = 1.0 / math.sqrt(graph.max_deg)
    weights = torch.randn(num_births, device=device) * sigma
    graph.out_weights[src_indices, slot_indices] = weights.to(graph.out_weights.dtype)
    graph.synapse_age[src_indices, slot_indices] = 0
    graph._invalidate_edge_cache()

def update_lph_plasticity(graph: LRSGraph):
    """Ensures LPH (Block 9) wiring follows strict spatial and density constraints."""
    device = graph.device
    lph_indices = graph.idx_lph
    
    # 1. ENFORCE DENSITY LIMIT (1:10,000 relative to global N)
    # Global N is ~4.2M, so limit is ~420. Our max_deg (24) is well within this.
    # We strictly enforce max_deg for LPH nodes.
    lph_targets = graph.out_targets[lph_indices]
    valid = lph_targets >= 0
    
    # 2. AUDIT MINICOLUMN CONSTRAINTS
    # LPH nodes should primarily target their own block or adjacent blocks
    # We prune synapses that violate local 3D structural layout
    for i, src in enumerate(lph_indices):
        src_targets = lph_targets[i]
        src_valid = valid[i]
        if not src_valid.any():
            continue
            
        dst_indices = src_targets[src_valid]
        dst_blocks = graph.block_ids[dst_indices]
        
        # Pruning logic: if target is outside LPH neighborhood and weight is weak
        is_outside = dst_blocks != 9
        weights = graph.out_weights[src][src_valid].abs()
        
        # Prune synapses to non-LPH blocks if they are in the bottom 20% of weights
        if is_outside.any():
            threshold = torch.quantile(weights.float(), 0.2)
            to_prune_mask = is_outside & (weights < threshold)
            if to_prune_mask.any():
                prune_slots = torch.where(src_valid)[0][to_prune_mask]
                graph.out_targets[src, prune_slots] = -1
                graph.out_weights[src, prune_slots] = 0.0

def metabolic_maintenance(
    graph: LRSGraph,
    config: LRSGConfig,
    tau: int = 100,
    a_target: float = 0.05,
    eta: float = 0.001,
    alpha: float = 0.8,
    beta: float = 0.1,
    gamma: float = 0.1,
    delta: float = 0.5,
    track_deltas: bool = False,
    age_threshold: int = 50
):
    """
    Implements Homeostatic Synaptic Scaling and Columnar Sparsity Pruning.
    Includes a Scaling Isolation Mask to shield young connections.
    """
    device = graph.device
    
    # 1. HOMEOSTATIC SYNAPTIC SCALING
    # Update rolling average: a_i(t) = (1 - 1/τ)a_i(t-1) + (1/τ)x_{i,t}
    graph.rolling_avg = (1.0 - 1.0/tau) * graph.rolling_avg + (1.0/tau) * graph.activations
    
    # Δs_i = η(A_target - a_i(t))
    delta_s = eta * (a_target - graph.rolling_avg)
    
    # Capture weights before scaling for ΔW_scaling audit
    if track_deltas:
        w_before_scaling = graph.out_weights.clone()

    # Scaling Isolation Mask: Shield young connections (age < age_threshold)
    scaling_factors = (1.0 + delta_s).unsqueeze(1).expand(-1, graph.max_deg)
    
    # CLAMP SCALING FACTORS for stability
    scaling_factors = torch.clamp(scaling_factors, min=0.8, max=1.2)
    
    mask = graph.synapse_age < age_threshold
    scaling_factors = torch.where(mask, torch.ones_like(scaling_factors), scaling_factors)
    
    # W_{ij}(t+1) = W_{ij}(t) * scaling_factors
    graph.out_weights *= scaling_factors.to(graph.out_weights.dtype)
    
    # Increment synapse age
    graph.synapse_age += 1
    
    delta_w_scaling = None
    if track_deltas:
        delta_w_scaling = graph.out_weights - w_before_scaling

    # 2. COLUMNAR SPARSITY (Structural Pruning)
    # OPTIMIZATION: Chunked processing to avoid VRAM thrashing on 8GB cards
    dst_indices_p = graph.out_targets
    valid = dst_indices_p >= 0
    
    if valid.any():
        num_valid = valid.sum().item()
        chunk_size = 1_000_000 # 1M synapses at a time
        
        # Flatten valid indices for easier chunking
        flat_valid_indices = torch.where(valid.view(-1))[0]
        
        for start_idx in range(0, num_valid, chunk_size):
            end_idx = min(start_idx + chunk_size, num_valid)
            curr_chunk_indices = flat_valid_indices[start_idx:end_idx]
            
            # Map back to [N, max_deg] coordinates
            src_idx = curr_chunk_indices // graph.max_deg
            slot_idx = curr_chunk_indices % graph.max_deg
            dst_idx = dst_indices_p.view(-1)[curr_chunk_indices]
            
            src_pos = graph.positions[src_idx]
            dst_pos = graph.positions[dst_idx]
            dist = torch.norm(src_pos - dst_pos, dim=1)
            
            src_blocks = graph.block_ids[src_idx]
            dst_blocks = graph.block_ids[dst_idx]
            c_ij = (src_blocks == dst_blocks).float()
            
            prob = alpha * torch.exp(-beta * dist) * c_ij + gamma * torch.exp(-delta * dist) * (1.0 - c_ij)
            
            # Pruning mask for this chunk
            keep = torch.rand_like(prob) < prob
            
            # Apply pruning in-place
            to_prune = ~keep
            if to_prune.any():
                prune_src = src_idx[to_prune]
                prune_slot = slot_idx[to_prune]
                graph.out_targets[prune_src, prune_slot] = -1
                graph.out_weights[prune_src, prune_slot] = 0.0
        
        graph._invalidate_edge_cache()

    # 2b. LPH STRUCTURAL AUDIT
    update_lph_plasticity(graph)

    # 3. CONTINUOUS STRUCTURAL NEUROGENESIS
    # OPTIMIZATION: Efficient sampling instead of O(N*D) check
    sensory_actual = graph.activations[graph.input_slice]
    graph.prediction_error = torch.mean((sensory_actual - graph.predicted_sensory)**2)
    
    kappa = 1.0
    p_birth = torch.sigmoid(kappa * graph.prediction_error).item()
    
    if p_birth > 0.01: # Only attempt birth if error is significant
        # Sample random candidates
        num_candidates = 2048 # Check 2048 random slots
        src_cand = torch.randint(0, graph.N, (num_candidates,), device=device)
        slot_cand = torch.randint(0, graph.max_deg, (num_candidates,), device=device)
        
        # Check which are empty
        targets_cand = graph.out_targets[src_cand, slot_cand]
        empty_mask = targets_cand == -1
        
        if empty_mask.any():
            empty_srcs = src_cand[empty_mask]
            empty_slots = slot_cand[empty_mask]
            
            # Limit births to 256
            num_to_birth = min(empty_srcs.numel(), 256)
            # We don't even need randperm if we already sampled randomly
            update_structural_neurogenesis(graph, empty_srcs[:num_to_birth], empty_slots[:num_to_birth])
            
    return delta_w_scaling, None # neuro delta not tracked for speed

def hebbian_update(
    graph: LRSGraph,
    config: LRSGConfig,
    reward: float,
    learning_rate: float = None,
    theta: float = None,
):
    if learning_rate is None:
        learning_rate = 0.01
    if theta is None:
        theta = config.threshold_theta
    device = graph.device

    # Use eligibility trace to find pre-synaptic neurons that fired recently
    # This enables temporal credit assignment across multiple ticks.
    trace_threshold = theta * 0.1
    pre_mask = graph.eligibility_trace > trace_threshold
    active_indices = torch.where(pre_mask)[0]
    
    if active_indices.numel() == 0:
        return

    pre = active_indices.unsqueeze(1)
    post = graph.out_targets[active_indices]
    valid = post >= 0

    pre_expanded = pre.expand_as(post)
    # pre_act uses the trace magnitude to represent recent activity level
    pre_act = graph.eligibility_trace[pre_expanded]

    # post_act uses eligibility trace of target neurons (temporal credit for post side too)
    post_act = torch.zeros_like(post, dtype=torch.float32, device=device)
    post_act[valid] = graph.eligibility_trace[post[valid]]

    coactive = valid & (pre_act > trace_threshold) & (post_act > trace_threshold)
    if not coactive.any():
        return

    # REMOVED O(N*D) sanity checks from hot path
    # graph.out_weights.nan_to_num_(...)
    # graph.out_weights.clamp_(...)

    # Remove signal choke point: use real reward magnitude, not cutoff that cancels learning
    # Reward is the direction and magnitude of the learning signal
    pre_final = pre_act[coactive].clamp(max=graph.trace_cap)
    post_final = post_act[coactive].clamp(max=graph.trace_cap)

    # We need to map back the coactive indices correctly to [N, max_deg]
    # batch_idx is the index into active_indices, slot_idx is the target slot
    batch_idx, slot_idx = torch.where(coactive)
    global_src_indices = active_indices[batch_idx]

    weights = graph.out_weights[global_src_indices, slot_idx].float()
    # Use real learning speed: learning_rate directly, reward as magnitude + direction
    # Don't multiply learning_rate by reward; reward shapes the update sign and strength
    delta_w = learning_rate * reward * pre_final * post_final

    new_weights = weights + delta_w
    new_weights = torch.clamp(new_weights, min=-10.0, max=10.0)
    new_weights.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    graph.out_weights[global_src_indices, slot_idx] = new_weights.to(graph.out_weights.dtype)


def structural_plasticity(
    graph: LRSGraph,
    config: LRSGConfig,
    tick: int = 0,
    birth_rate: float = 0.001,
    death_rate: float = 0.0005,
    min_weight: float = 0.01,
    active_window: int = 100,
    growth_cap: float = 1.2,
    growth_window: int = 1000,
    theta: float = None,
) -> tuple:
    if theta is None:
        theta = config.threshold_theta
    device = graph.device
    N = graph.N
    deg = graph.max_deg

    deaths = 0
    births = 0

    currently_active = graph.activations > (theta * 0.5)
    graph.activity_history[currently_active] = tick

    recently_active = (tick - graph.activity_history) <= active_window
    active_indices = torch.where(recently_active)[0]
    if active_indices.numel() == 0:
        return births, deaths

    active_targets = graph.out_targets[active_indices]
    active_weights = graph.out_weights[active_indices].float()
    valid = active_targets >= 0
    weak = valid & (active_weights.abs() < min_weight)
    if weak.any():
        num_deaths = weak.sum().item()
        deaths += num_deaths
        graph.out_targets[active_indices] = torch.where(
            weak, -1, graph.out_targets[active_indices]
        )
        zero_weights = torch.zeros(
            active_weights.shape, dtype=graph.out_weights.dtype, device=device
        )
        graph.out_weights[active_indices] = torch.where(weak, zero_weights, graph.out_weights[active_indices])
        graph._invalidate_edge_cache()

    empty_mask = graph.out_targets[active_indices] == -1
    num_empty = empty_mask.sum().item()
    if num_empty == 0:
        return births, deaths

    if hasattr(graph, "_edge_history") and len(graph._edge_history) >= growth_window:
        initial = graph._edge_history[0]
        recent = graph._edge_history[-growth_window:]
        if max(recent) > initial * growth_cap:
            return births, deaths

    birth_mask = torch.bernoulli(
        torch.full((active_indices.numel(),), birth_rate, dtype=torch.float32, device=device)
    ).bool()

    if not birth_mask.any():
        return births, deaths

    birth_srcs = active_indices[birth_mask]
    
    if birth_srcs.numel() > 0:
        # Find first empty slot for each source in birth_srcs
        # graph.out_targets[birth_srcs] is [num_births, max_deg]
        targets_slice = graph.out_targets[birth_srcs]
        empty_mask = targets_slice == -1
        
        # Get first empty slot index for each source
        # This is a bit tricky to vectorize if they have different empty slots
        # But we can use argmax on the mask
        has_empty = empty_mask.any(dim=1)
        if has_empty.any():
            valid_birth_srcs = birth_srcs[has_empty]
            # argmax returns the first index of the max value (True > False)
            first_empty_slots = empty_mask[has_empty].float().argmax(dim=1)
            
            update_structural_neurogenesis(graph, valid_birth_srcs, first_empty_slots)
            births = valid_birth_srcs.numel()

    graph.out_weights.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    graph.out_weights.clamp_(min=-10.0, max=10.0)

    return births, deaths
