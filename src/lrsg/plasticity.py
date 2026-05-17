import torch
from src.lrsg.graph import LRSGraph
from src.config import LRSGConfig


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

    both_fired = graph.activations > theta
    active_indices = torch.where(both_fired)[0]
    if active_indices.numel() == 0:
        return

    pre = active_indices.unsqueeze(1)
    post = graph.out_targets[active_indices]
    valid = post >= 0

    pre_expanded = pre.expand_as(post)
    pre_act = graph.activations[pre_expanded]

    post_clamped = graph.activations.clamp(min=0)
    post_act = torch.zeros_like(post, dtype=torch.float32, device=device)
    post_act[valid] = post_clamped[post[valid]]

    coactive = valid & (pre_act > theta) & (post_act > theta)
    if not coactive.any():
        return

    pre_final = pre_act[coactive]
    post_final = post_act[coactive]
    weights = graph.out_weights[active_indices][coactive].float()
    learning_rate_eff = learning_rate * reward

    delta_w = learning_rate_eff * pre_final * post_final
    batch_idx, slot_idx = torch.where(coactive)
    new_weights = weights + delta_w
    new_weights = torch.clamp(new_weights, min=-10.0, max=10.0)
    graph.out_weights[active_indices[batch_idx], slot_idx] = new_weights.to(graph.out_weights.dtype)


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
    placed = torch.zeros(birth_srcs.shape[0], dtype=torch.bool, device=device)
    dsts = torch.randint(0, N, (birth_srcs.shape[0],), device=device)
    for slot in range(deg):
        slot_empty = graph.out_targets[birth_srcs, slot] == -1
        need_placed = slot_empty & ~placed
        if need_placed.any():
            num_births = need_placed.sum().item()
            births += num_births
            graph.out_targets[birth_srcs[need_placed], slot] = dsts[need_placed]
            graph.out_weights[birth_srcs[need_placed], slot] = (
                torch.randn(need_placed.sum().item(), device=device, dtype=torch.float16) * 0.05
            )
            placed = placed | need_placed
        if placed.all():
            break

    return births, deaths
