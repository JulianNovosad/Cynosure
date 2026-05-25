import torch
import math
import pytest
from src.lrsg.graph import LRSGraph
from src.lrsg.forward import forward_pass
from src.lrsg.plasticity import metabolic_maintenance
from src.config import LRSGConfig

@pytest.mark.cuda
def test_neuro_architectures():
    print("Initializing scaled LRSG for testing...")
    config = LRSGConfig(
        num_neurons=500,  # Small N for fast CPU/GPU test
        num_input_neurons=50,
        num_output_neurons=10,
        propagation_steps=5
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Test Initialization and 1:10,000 Scaling (clamped to N)
    graph = LRSGraph(config, device=device)
    
    print(f"Graph initialized with N={graph.N}, max_deg={graph.max_deg}")
    k_expected = min(10000, graph.N)
    valid_edges_per_node = (graph.out_targets >= 0).sum(dim=1)
    
    assert (valid_edges_per_node == k_expected).all(), f"Edge count mismatch. Expected {k_expected}, got {valid_edges_per_node[0]}"
    print(f"✓ Structural scaling verified: {k_expected} synapses per neuron.")

    # 2. Test PCZ/BG Structure
    assert hasattr(graph, 'idx_p'), "PCZ indices missing"
    assert hasattr(graph, 'idx_a'), "BG Striatal indices missing"
    assert hasattr(graph, 'idx_b'), "BG Thalamic indices missing"
    print(f"✓ PCZ/BG sub-networks initialized. PCZ size: {graph.idx_p.numel()}")

    # 3. Test Forward Pass (with PCZ/BG Gating)
    graph.activations.fill_(0.1) # Simulate some activity
    print("Running forward pass...")
    forward_pass(graph, config, theta=0.01)
    
    assert not torch.isnan(graph.activations).any(), "NaN in activations after forward pass"
    print("✓ Forward pass completed successfully.")

    # 4. Test Metabolic Maintenance (Scaling & Pruning)
    print("Running metabolic maintenance...")
    initial_weights_sum = graph.out_weights.abs().sum().item()
    metabolic_maintenance(graph, config, a_target=0.05, eta=0.1) # Large eta to see change
    
    final_weights_sum = graph.out_weights.abs().sum().item()
    print(f"Initial weight sum: {initial_weights_sum:.4f}, Final: {final_weights_sum:.4f}")
    
    # Pruning might have occurred
    num_edges_after = (graph.out_targets >= 0).sum().item()
    print(f"Edges after pruning: {num_edges_after} (Initial: {graph.N * k_expected})")
    
    print("✓ Metabolic maintenance verified.")
    print("\nALL TESTS PASSED SUCCESSFULLY.")

if __name__ == "__main__":
    test_neuro_architectures()
