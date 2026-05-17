import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.config import AgentConfig, LRSGConfig, ArenaConfig
from src.lrsg import LRSGraph
from src.arena import Environment, RAMNode
from src.agent import Agent


def test_agent_creation():
    lrsg_cfg = LRSGConfig(num_neurons=10000, num_input_neurons=100, num_output_neurons=12)
    brain = LRSGraph(lrsg_cfg)
    agent = Agent(0, (10, 10, 10), brain, AgentConfig(), lrsg_cfg)
    assert agent.alive
    assert agent.ram == 100.0
    assert agent.position == (10, 10, 10)
    assert agent.id == 0


def test_agent_vitals():
    lrsg_cfg = LRSGConfig(num_neurons=10000, num_input_neurons=100, num_output_neurons=12)
    brain = LRSGraph(lrsg_cfg)
    agent = Agent(0, (128, 128, 128), brain, AgentConfig(), lrsg_cfg)
    v = agent.vitals
    assert v.shape == (184,)
    assert v[0] == 0.5


def test_agent_sense_and_act():
    lrsg_cfg = LRSGConfig(num_neurons=10000, num_input_neurons=1848, num_output_neurons=12, vision_dims=1536)
    brain = LRSGraph(lrsg_cfg)
    agent = Agent(0, (50, 50, 50), brain, AgentConfig(), lrsg_cfg)
    env = Environment(ArenaConfig(grid_size=(100, 100, 100)))
    announcer_emb = torch.zeros(128)
    agent.sense_and_act(env, announcer_emb)
    assert 0 <= agent.last_action < 4
    assert 0 <= agent.last_direction < 6
    assert 0 <= agent.last_intent < 2


def test_agent_move():
    lrsg_cfg = LRSGConfig(num_neurons=10000, num_input_neurons=100, num_output_neurons=12)
    brain = LRSGraph(lrsg_cfg)
    agent = Agent(0, (50, 50, 50), brain, AgentConfig(), lrsg_cfg)
    env = Environment(ArenaConfig(grid_size=(100, 100, 100)))
    agent.last_action = 3
    agent.last_direction = 0
    agent.execute_action(env)
    assert agent.position[0] > 50


def test_agent_ram_decay():
    lrsg_cfg = LRSGConfig(num_neurons=10000, num_input_neurons=100, num_output_neurons=12)
    brain = LRSGraph(lrsg_cfg)
    agent = Agent(0, (50, 50, 50), brain, AgentConfig(), lrsg_cfg)
    env = Environment(ArenaConfig(grid_size=(100, 100, 100)))
    agent.last_action = 3
    initial = agent.ram
    agent.execute_action(env)
    assert agent.ram < initial


def test_agent_collect_ram():
    lrsg_cfg = LRSGConfig(num_neurons=10000, num_input_neurons=100, num_output_neurons=12)
    brain = LRSGraph(lrsg_cfg)
    agent = Agent(0, (10, 10, 10), brain, AgentConfig(), lrsg_cfg)
    env = Environment(ArenaConfig(grid_size=(100, 100, 100)))
    env.ram_nodes.clear()
    env.ram_nodes.append(RAMNode((10, 10, 10), 15))
    collected = agent.collect_ram(env)
    assert collected == 15 + ArenaConfig().ram_collect_bonus
    assert agent.ram == 100 + 15 + ArenaConfig().ram_collect_bonus


def test_agent_dies_when_ram_zero():
    lrsg_cfg = LRSGConfig(num_neurons=10000, num_input_neurons=100, num_output_neurons=12)
    brain = LRSGraph(lrsg_cfg)
    agent = Agent(0, (50, 50, 50), brain, AgentConfig(), lrsg_cfg)
    env = Environment(ArenaConfig(grid_size=(100, 100, 100)))
    agent.ram = 0.001
    agent.last_action = 3
    agent.execute_action(env)
    assert not agent.alive


def test_agent_bounds():
    lrsg_cfg = LRSGConfig(num_neurons=10000, num_input_neurons=100, num_output_neurons=12)
    brain = LRSGraph(lrsg_cfg)
    agent = Agent(0, (0, 0, 0), brain, AgentConfig(), lrsg_cfg)
    env = Environment(ArenaConfig(grid_size=(100, 100, 100)))
    agent.last_action = 3
    agent.last_direction = 1
    for _ in range(10):
        agent.execute_action(env)
    assert agent.position[0] >= 0


def test_agent_vision():
    lrsg_cfg = LRSGConfig(num_neurons=10000, num_input_neurons=1848, num_output_neurons=12, vision_dims=1536)
    brain = LRSGraph(lrsg_cfg)
    agent = Agent(0, (50, 50, 50), brain, AgentConfig(), lrsg_cfg)
    env = Environment(ArenaConfig(grid_size=(100, 100, 100)))
    vision = agent._compute_vision(env)
    assert vision.shape == (1536,)


if __name__ == "__main__":
    test_agent_creation()
    print("[PASS] test_agent_creation")
    test_agent_vitals()
    print("[PASS] test_agent_vitals")
    test_agent_sense_and_act()
    print("[PASS] test_agent_sense_and_act")
    test_agent_move()
    print("[PASS] test_agent_move")
    test_agent_ram_decay()
    print("[PASS] test_agent_ram_decay")
    test_agent_collect_ram()
    print("[PASS] test_agent_collect_ram")
    test_agent_dies_when_ram_zero()
    print("[PASS] test_agent_dies_when_ram_zero")
    test_agent_bounds()
    print("[PASS] test_agent_bounds")
    test_agent_vision()
    print("[PASS] test_agent_vision")
    print("All agent tests passed!")
