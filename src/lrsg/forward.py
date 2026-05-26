import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from src.lrsg.graph import LRSGraph
from src.config import LRSGConfig
from src.lrsg.triton_kernels import propagation_step


def apply_lph_fatigue(graph: LRSGraph):
    """Subtract adaptive fatigue from LPH activations before decoding."""
    if hasattr(graph, 'lph_fatigue') and graph.lph_fatigue.numel() == graph.idx_lph.numel():
        graph.activations[graph.idx_lph] = graph.activations[graph.idx_lph] - graph.lph_fatigue
        graph.activations[graph.idx_lph].clamp_(min=0.0)


def update_lph_fatigue(graph: LRSGraph):
    """Penalize the winning LPH node and decay the entire fatigue tensor."""
    if not hasattr(graph, 'lph_fatigue'):
        return

    decoded_char = graph._decode_raw_char()
    if decoded_char and len(decoded_char) > 0:
        char_code = ord(decoded_char[0]) % 256
        winner_global = graph.char_to_idx[char_code]
        winner_local_mask = graph.idx_lph == winner_global
        if winner_local_mask.any():
            winner_local_idx = torch.where(winner_local_mask)[0][0]
            graph.lph_fatigue[winner_local_idx] += 5.0  # Reduced from 10.0

    graph.lph_fatigue *= 0.85  # Slower decay (was 0.75)


def pwm_character_prediction(graph: LRSGraph) -> torch.Tensor:
    """PWM → character distribution prediction head.
    Returns logits over 256 ASCII characters from current PWM state."""
    pwm_state = graph.activations[graph.idx_pwm].half()
    logits = torch.matmul(graph.W_char_pred, pwm_state)
    return logits.float()


def update_char_prediction_hebbian(graph: LRSGraph, target_char_code: int):
    """Hebbian update to W_char_pred: strengthen PWM→char weights that
    correctly predicted the observed character."""
    pwm_state = graph.activations[graph.idx_pwm].half()
    logits = graph.W_char_pred @ pwm_state
    probs = torch.softmax(logits.float(), dim=0)

    error = torch.zeros(256, device=graph.device, dtype=torch.float16)
    error[target_char_code] = 1.0
    error.sub_(probs.half())

    lr = graph.char_pred_lr
    # In-place rank-1 update: W += lr * error ⊗ pwm
    graph.W_char_pred.data.add_(error.unsqueeze(1) * pwm_state.unsqueeze(0), alpha=lr)

    graph.char_prediction_loss = torch.nn.functional.cross_entropy(
        logits.float().unsqueeze(0), torch.tensor([target_char_code], device=graph.device)
    ).detach()


def compute_bigram_coherence_bonus(graph: LRSGraph, current_char_code: int) -> float:
    if graph._prev_char_code < 0:
        return 1.0

    prev = graph._prev_char_code % 256
    curr = current_char_code % 256

    count = graph.bigram_counts[prev, curr]
    total = graph.bigram_counts[prev].sum()

    if total < 1e-6:
        return 1.0

    freq = count / total.clamp(min=1.0)
    scaled = (freq * 10.0).clamp(max=1.0)
    bonus = (0.5 + 1.5 * scaled).clamp(0.5, 2.0)

    graph.bigram_counts[prev, curr] += 0.1

    return bonus.item()


def trace_lph_activation(graph: LRSGraph):
    """Store current LPH activations in a rolling trace for Hebbian reward shaping."""
    if not hasattr(graph, '_lph_activation_trace'):
        return
    slot = graph._trace_cursor % 20
    graph._lph_activation_trace[slot] = graph.activations[graph.idx_lph].detach().clone()
    graph._trace_cursor += 1


def apply_language_hebbian_reward(graph: LRSGraph, reward: torch.Tensor, bigram_bonus: float):
    if abs(reward) < 1e-6:
        return

    effective_lr = 0.001 * reward * bigram_bonus

    trace_len = min(20, graph._trace_cursor)
    if trace_len < 2:
        graph._language_reward_accumulator += reward * bigram_bonus
        return

    if graph._trace_cursor >= 20:
        recent_trace = graph._lph_activation_trace
    else:
        recent_trace = graph._lph_activation_trace[:graph._trace_cursor]

    if recent_trace.numel() == 0:
        graph._language_reward_accumulator += reward * bigram_bonus
        return

    avg_lph_activation = recent_trace.mean(dim=0)
    active_mask = avg_lph_activation > graph.config.threshold_theta

    if not active_mask.any():
        graph._language_reward_accumulator += reward * bigram_bonus
        return

    active_lph = graph.idx_lph[active_mask]
    for node_idx in active_lph:
        targets = graph.out_targets[node_idx]
        weights = graph.out_weights[node_idx].float()
        valid = targets >= 0
        if not valid.any():
            continue

        node_act = avg_lph_activation[active_mask][list(active_lph).index(node_idx)]
        target_acts = graph.activations[targets[valid]]
        coactive = target_acts > graph.config.threshold_theta

        if coactive.any():
            delta = effective_lr * node_act * target_acts[coactive]
            valid_slots = torch.where(valid)[0]
            coactive_slots = valid_slots[coactive]
            graph.out_weights[node_idx, coactive_slots] += delta.to(graph.out_weights.dtype)

    graph._language_reward_accumulator += reward * bigram_bonus


def clear_language_buffer(graph: LRSGraph):
    """Temporal blanking mask: zero-mask winning LPH neuron + spatial neighbors for 1 tick."""
    lph_acts = graph.activations[graph.idx_lph]
    decoded_char = graph.decode_lph()

    if decoded_char and len(decoded_char) > 0:
        winner_local = torch.argmax(lph_acts)
        winner_global = graph.idx_lph[winner_local]
        winner_pos = graph.positions[winner_global]

        lph_positions = graph.positions[graph.idx_lph]
        dists = torch.norm(lph_positions - winner_pos.unsqueeze(0), dim=1)
        neighbor_mask = dists <= 0.03  # Reduced from 0.05

        blank_indices = graph.idx_lph[neighbor_mask]
        graph.activations[blank_indices] = 0.0

        non_blank = graph.idx_lph[~neighbor_mask]
        graph.activations[non_blank] *= 0.3  # Reduced suppression (was 0.1)
    else:
        graph.activations[graph.idx_lph] *= 0.3  # Reduced suppression (was 0.1)

# TRITON_FALLBACK — kept as reference; Triton kernel in triton_kernels.py replaces this
def _fused_propagation_step(activations, out_targets, out_weights, frontier, theta, gamma_t, p_start, p_end):
    if frontier.numel() == 0:
        return activations, frontier

    src_targets = out_targets[frontier]
    src_weights = out_weights[frontier]
    src_vals = activations[frontier].unsqueeze(1)

    valid = src_targets >= 0
    if not valid.any():
        return activations, frontier

    act_before_threshold = activations[frontier].clone()

    excitatory = src_weights > 0
    inhibitory = src_weights < 0

    if excitatory.any():
        all_targets_e = src_targets[valid & excitatory]
        contrib_e = (src_vals * src_weights)[valid & excitatory]
        
        is_pcz_target = (all_targets_e >= p_start) & (all_targets_e < p_end)
        contrib_e[is_pcz_target] *= gamma_t

        contrib_e.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        activations.scatter_add_(0, all_targets_e, contrib_e)

    if inhibitory.any():
        all_targets_i = src_targets[valid & inhibitory]
        contrib_i = (src_vals * src_weights)[valid & inhibitory]
        contrib_i.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        activations.scatter_add_(0, all_targets_i, contrib_i)

    activations.clamp_(min=0.0, max=1e4)
    activations.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    crossed_now_vals = activations[frontier] > theta
    crossed_before = act_before_threshold > theta
    crossed_frontier = crossed_now_vals & ~crossed_before
    new_frontier = frontier[crossed_frontier]
    
    return activations, new_frontier

def forward_pass(graph: LRSGraph, config: LRSGConfig, theta: float = None, last_action: int = None):
    T = config.propagation_steps
    if theta is None:
        theta = config.threshold_theta

    # REMOVED O(N) sanity checks from hot path for speed.
    # Logic should ensure stability without per-tick O(N) cleaning.
    # graph.activations.nan_to_num_(...) 
    # graph.out_weights.nan_to_num_(...)
    # graph.out_weights.clamp_(...)

    # ... (PWM and scaling logic remains same)
    # 1. PREDICTIVE WORLD MODEL (PWM) TRANSITION & ACTIVE INFERENCE
    if last_action is not None:
        a_onehot = torch.zeros(7, device=graph.device, dtype=torch.float16)
        if 0 <= last_action < 7: a_onehot[last_action] = 1.0

        if last_action == 4: # WHISPER
            lph_winner_idx = torch.argmax(graph.activations[graph.idx_lph])
            pwm_sub_idx = lph_winner_idx % graph.idx_pwm.numel()
            graph.activations[graph.idx_pwm[pwm_sub_idx]] += 1.0

        x_pwm_prev = graph.activations[graph.idx_pwm].half()
        raw_next = torch.matmul(graph.W_pwm, x_pwm_prev) + torch.matmul(graph.W_act, a_onehot)
        x_pwm_next = torch.tanh(raw_next.float()) if config.pwm_activation == "tanh" else torch.nn.functional.leaky_relu(raw_next.float(), 0.01)
        graph.activations[graph.idx_pwm] = x_pwm_next
        graph.predicted_sensory = torch.sigmoid(torch.matmul(graph.W_dec, x_pwm_next.half()).float())

    active_slice = slice(0, config.h_window)
    activations_active = graph.activations[active_slice]

    h_rest_active_mask = (graph.idx_h_rest < config.h_window)
    if h_rest_active_mask.any():
        v_t = graph.activations[graph.idx_h_rest[h_rest_active_mask]].var()
    else:
        v_t = torch.tensor(0.0, device=graph.device)
    gamma_t = torch.sigmoid(graph.W_v * v_t)

    # FP16 fused bmm: W_proj + W_fb in one call instead of two separate matmuls
    act_half = activations_active.half().contiguous()                # (h_window,)
    w_proj_slice = graph.W_proj[:, active_slice].contiguous()        # (out_dim, h_window)
    w_fb_slice = graph.W_fb[:, active_slice].contiguous()            # (out_dim, h_window)
    stacked_W = torch.stack([w_proj_slice, w_fb_slice], dim=0)      # (2, out_dim, h_window)
    act_expanded = act_half.unsqueeze(0).unsqueeze(-1).expand(2, -1, -1)  # (2, h_window, 1)
    x_a, x_b_raw = torch.bmm(stacked_W, act_expanded).squeeze(-1)   # (2, out_dim)
    inhib = torch.matmul(graph.W_inh.contiguous(), x_a.unsqueeze(-1)).squeeze(-1)
    x_b = torch.relu(x_b_raw - torch.exp(inhib))

    # Back to float for sparse propagation
    input_active = graph.activations > 0
    frontier = torch.where(input_active)[0]
    
    p_start = graph.p_start
    p_end = graph.p_end

    for _ in range(T):
        graph.activations, frontier = propagation_step(
            graph.activations, graph.out_targets, graph.out_weights, 
            frontier, theta, gamma_t, p_start, p_end,
            use_triton=True,
        )
        if frontier.numel() == 0: break
    
    # ... (Rest of maintenance remains same)

    # Apply BG gate to output activations
    if x_b.numel() > 0:
        gate_out = torch.sigmoid(torch.nn.functional.interpolate(x_b.view(1, 1, -1), size=graph.output_slice.stop - graph.output_slice.start, mode='linear').view(-1))
        graph.activations[graph.output_slice] *= gate_out

    # Final global clamp
    graph.activations.clamp_(min=0.0, max=100.0)

    # 3. POST-PROPAGATION: CHARACTER PREDICTION + HEBBIAN REWARD
    if not config.disable_language:
        trace_lph_activation(graph)

        # Populate lph_history BEFORE decoding so _decode_raw_char() has data
        hist_slot = graph._lph_history_cursor % 5
        graph.lph_history[hist_slot] = graph.activations[graph.idx_lph].detach().clone()
        graph._lph_history_cursor += 1

        char_logits = pwm_character_prediction(graph)
        char_probs = torch.softmax(char_logits, dim=0)

        decoded_char = graph._decode_raw_char()
        if decoded_char and len(decoded_char) > 0:
            char_code = ord(decoded_char[0]) % 256
            update_char_prediction_hebbian(graph, char_code)

            bigram_bonus = compute_bigram_coherence_bonus(graph, char_code)

            char_surprise = 1.0 - char_probs[char_code]
            reward_signal = 0.1 + 0.9 * char_surprise
            apply_language_hebbian_reward(graph, reward_signal, bigram_bonus)

            graph._prev_char_code = char_code

        # 4. BUFFER MAINTENANCE
        apply_lph_fatigue(graph)
        clear_language_buffer(graph)
        update_lph_fatigue(graph)
