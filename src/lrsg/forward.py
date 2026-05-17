import torch
from src.lrsg.graph import LRSGraph
from src.config import LRSGConfig


def forward_pass(graph: LRSGraph, config: LRSGConfig, theta: float = None):
    T = config.propagation_steps
    if theta is None:
        theta = config.threshold_theta

    input_active = graph.activations > 0
    frontier = torch.where(input_active)[0]

    for _ in range(T):
        if frontier.numel() == 0:
            break

        src_targets = graph.out_targets[frontier]
        src_weights = graph.out_weights[frontier]
        src_vals = graph.activations[frontier]

        valid = src_targets >= 0
        if not valid.any():
            break

        act_before = graph.activations.clone()

        all_targets = src_targets[valid]
        contributions = (src_vals.unsqueeze(1) * src_weights)[valid]

        graph.activations.scatter_add_(0, all_targets, contributions)
        graph.activations.clamp_(min=0.0)

        crossed_now = (graph.activations > theta) & ~(act_before > theta)
        frontier = torch.where(crossed_now)[0]
