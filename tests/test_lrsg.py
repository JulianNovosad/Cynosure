import torch
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import LRSGConfig
from src.lrsg import LRSGraph, forward_pass
from src.lrsg.io import encode_input, decode_output


def test_graph_creation():
    config = LRSGConfig(num_neurons=10000, num_input_neurons=100, num_output_neurons=12)
    graph = LRSGraph(config)
    assert graph.N == 10000
    assert graph.out_targets.shape == (10000, 12)
    assert graph.out_weights.shape == (10000, 12)
    assert graph.num_edges > 0


@pytest.mark.cuda
def test_forward_pass():
    config = LRSGConfig(num_neurons=10000, num_input_neurons=100, num_output_neurons=12)
    graph = LRSGraph(config)

    input_tensor = torch.randn(100) * 2.0
    graph.inject_input(input_tensor)

    forward_pass(graph, config)

    output = graph.read_output()
    assert output.shape == (12,)


def test_encode_decode():
    config = LRSGConfig(num_input_neurons=1848, num_output_neurons=527)
    vision = torch.randn(8, 8, 24)
    vitals = torch.randn(184)
    announcer_emb = torch.randn(128)

    encoded = encode_input(vision, vitals, announcer_emb, config)
    assert encoded.shape == (1848,)

    output = torch.randn(527)
    decoded = decode_output(output)
    assert "action" in decoded
    assert "direction" in decoded
    assert "intent" in decoded
    assert 0 <= decoded["action"] < 7
    assert 0 <= decoded["direction"] < 6
    assert 0 <= decoded["intent"] < 2


@pytest.mark.cuda
def test_sparse_propagation():
    N = 1000
    config = LRSGConfig(
        num_neurons=N,
        num_input_neurons=10,
        num_output_neurons=4,
    )
    graph = LRSGraph(config)
    graph.reset_activations()

    graph.activations[0] = 1.0
    forward_pass(graph, config)

    output = graph.read_output()
    assert output.shape == (4,)


def test_hebbian_uses_post_trace():
    """Post-synaptic eligibility trace is used (not instantaneous activations)."""
    from src.lrsg import hebbian_update
    config = LRSGConfig(num_neurons=5000, num_input_neurons=100, num_output_neurons=12)
    graph = LRSGraph(config)
    graph.trace_decay = 0.95
    graph.trace_cap = 50.0

    # Build up trace over several ticks
    for _ in range(15):
        graph.reset_activations()
        graph.activations[:100] = torch.rand(100)
        forward_pass(graph, config, theta=config.threshold_theta, last_action=-1, energy=200.0)

    # Zero out current activations so instantaneous post_act = 0
    graph.activations.zero_()

    # With the old code (post_act from activations), no coactive pairs → no update
    # With the new code (post_act from trace), trace has accumulated → updates happen
    w_before = graph.out_weights.clone()
    hebbian_update(graph, config, 1.0, learning_rate=0.05)
    w_after = graph.out_weights
    delta = (w_after.float() - w_before.float()).abs().sum().item()
    assert delta > 0, "hebbian_update must use post eligibility trace (not instantaneous activations)"


def test_hebbian_negative_reward_depresses():
    """Negative reward depresses co-active synapses."""
    from src.lrsg import hebbian_update
    config = LRSGConfig(num_neurons=5000, num_input_neurons=100, num_output_neurons=12)
    graph = LRSGraph(config)
    graph.trace_decay = 0.95

    for _ in range(10):
        graph.reset_activations()
        graph.activations[:100] = torch.rand(100) * 2.0
        forward_pass(graph, config, theta=config.threshold_theta, last_action=3, energy=200.0)

    w_before = graph.out_weights.float().clone()
    hebbian_update(graph, config, -0.5, learning_rate=0.05)
    w_after = graph.out_weights.float()
    # Some positive weights should have decreased (been depressed)
    positive_before = w_before > 0.001
    decreased = (w_after - w_before) < 0
    assert (positive_before & decreased).any(), "Negative reward should depress positive weights"


if __name__ == "__main__":
    test_graph_creation()
    print("[PASS] test_graph_creation")
    test_encode_decode()
    print("[PASS] test_encode_decode")
    test_hebbian_uses_post_trace()
    print("[PASS] test_hebbian_uses_post_trace")
    test_hebbian_negative_reward_depresses()
    print("[PASS] test_hebbian_negative_reward_depresses")
    print("All tests passed!")
