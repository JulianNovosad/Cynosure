import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import LRSGConfig
from src.lrsg import LRSGraph, forward_pass
from src.lrsg.io import encode_input, decode_output


def test_graph_creation():
    config = LRSGConfig(num_neurons=10000, num_input_neurons=100, num_output_neurons=12)
    graph = LRSGraph(config)
    assert graph.N == 10000
    assert graph.out_targets.shape == (10000, 4)
    assert graph.out_weights.shape == (10000, 4)
    assert graph.num_edges > 0


def test_forward_pass():
    config = LRSGConfig(num_neurons=10000, num_input_neurons=100, num_output_neurons=12)
    graph = LRSGraph(config)

    input_tensor = torch.randn(100) * 2.0
    graph.inject_input(input_tensor)

    forward_pass(graph, config)

    output = graph.read_output()
    assert output.shape == (12,)


def test_encode_decode():
    config = LRSGConfig(num_input_neurons=1848, num_output_neurons=12)
    vision = torch.randn(8, 8, 24)
    vitals = torch.randn(184)
    announcer_emb = torch.randn(128)

    encoded = encode_input(vision, vitals, announcer_emb, config)
    assert encoded.shape == (1848,)

    output = torch.randn(12)
    decoded = decode_output(output)
    assert "action" in decoded
    assert "direction" in decoded
    assert "intent" in decoded
    assert 0 <= decoded["action"] < 4
    assert 0 <= decoded["direction"] < 6
    assert 0 <= decoded["intent"] < 2


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


if __name__ == "__main__":
    test_graph_creation()
    print("[PASS] test_graph_creation")
    test_forward_pass()
    print("[PASS] test_forward_pass")
    test_encode_decode()
    print("[PASS] test_encode_decode")
    test_sparse_propagation()
    print("[PASS] test_sparse_propagation")
    print("All tests passed!")
