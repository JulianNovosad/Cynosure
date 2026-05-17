from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class LRSGConfig:
    num_neurons: int = 4_194_304
    max_out_degree: int = 24
    num_input_neurons: int = 1848
    num_output_neurons: int = 25
    propagation_steps: int = 5
    threshold_theta: float = 0.01
    announcer_embed_dim: int = 128

    vision_dims: int = 1536
    vitals_dims: int = 184

    num_actions: int = 7
    num_directions: int = 6
    num_intents: int = 2

    context_size: int = 256
    theta_range: Tuple[float, float] = (0.008, 0.015)
    learning_rate_range: Tuple[float, float] = (0.005, 0.02)
    exploration_bias_range: Tuple[float, float] = (0.2, 0.8)

    whisper_vocab_size: int = 512
    whisper_latent_dim: int = 10

    probe_base_reward: float = 0.3
    whisper_base_reward: float = 0.2
    manipulate_base_reward: float = 0.1


@dataclass
class ArenaConfig:
    grid_size: Tuple[int, int, int] = (32, 32, 32)
    safe_zone_initial_radius: float = 28.0
    safe_zone_contract_rate: float = 0.995
    safe_zone_contract_interval_ticks: int = 100
    safe_zone_contract_fixed: float = 1.0
    safe_zone_min_radius: float = 5.0
    safe_zone_damage_rate: float = 0.005
    ram_node_spawn_interval: int = 50
    ram_node_yield: int = 7
    ram_collect_bonus: int = 25
    ram_collect_range: float = 15.0
    max_ram_nodes: int = 5
    agent_spawn_radius: float = 15.0
    safe_zone_weight_corruption_rate: float = 0.002


@dataclass
class AgentConfig:
    initial_ram: int = 200
    ram_decay_rate: float = 0.005
    vision_range: int = 8
    sensory_radius: float = 10.0
    memory_decay_ticks: int = 50


@dataclass
class SimulationConfig:
    physics_tps: int = 10
    linguistic_tps: int = 1
    num_agents: int = 10
    max_ticks: int = 10_000
    use_llm_announcer: bool = False
    announcer_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@dataclass
class Config:
    lrsg: LRSGConfig = field(default_factory=LRSGConfig)
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
