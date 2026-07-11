from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class LRSGConfig:
    num_neurons: int = 1_048_576
    max_out_degree: int = 12
    num_input_neurons: int = 1848
    num_output_neurons: int = 527
    h_window: int = 4096
    propagation_steps: int = 5
    threshold_theta: float = 0.01
    whisper_buffer_dim: int = 48

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
    whisper_latent_dim: int = 512

    probe_base_reward: float = 0.3
    whisper_base_reward: float = 0.2
    manipulate_base_reward: float = 0.1
    pwm_activation: str = "tanh" # options: "tanh", "leaky_relu"

    # Experiment control flags
    disable_language: bool = False
    disable_structural_plasticity: bool = False

    # PWM character prediction parameters
    char_pred_lr: float = 0.002
    language_hebbian_lr: float = 0.001
    bigram_bootstrap_weight: float = 0.1
    char_surprise_reward_scale: float = 0.9


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
class LocomotionConfig:
    """Configuration for differential-drive locomotion."""
    wheelbase: float = 0.15  # Distance between wheels (meters)
    wheel_radius: float = 0.05  # Wheel radius (meters)
    max_motor_speed: float = 1.0  # Max motor command magnitude
    friction_coeff: float = 0.8  # Friction scaling
    max_linear_velocity: float = 0.04  # Max forward speed (m/s)
    max_angular_velocity: float = 2.0  # Max turning speed (rad/s)


@dataclass
class EmbodiedConfig:
    """Configuration for embodied simulation environment."""
    grid_size: float = 2.0  # Grid dimensions (meters) — sized to be reachable within episode_length at max_linear_velocity
    num_obstacles: int = 3  # Obstacles per episode
    num_enemies: int = 3  # Enemies per episode (also collidable)
    episode_length: int = 300  # Ticks per episode
    physics_freq: int = 240  # Hz
    lidar_rays: int = 8  # Number of LiDAR rays
    goal_reach_threshold: float = 0.3  # Distance to goal to count as reached
    exploration_epsilon: float = 0.3  # Probability of random action (exploration)
    exploration_decay: float = 0.995  # Per-episode multiplicative decay of epsilon
    min_exploration_epsilon: float = 0.05  # Floor for exploration_epsilon
    collision_grace_ticks: int = 10  # Consecutive colliding ticks tolerated before ending the episode


@dataclass
class EpisodicCompressorConfig:
    input_dim: int = 10
    hidden_dim: int = 50
    latent_dim: int = 32
    buffer_max_size: int = 100
    train_interval: int = 10
    summary_interval: int = 10
    context_inject_weight: float = 0.3
    summary_history_size: int = 5


@dataclass
class SimulationConfig:
    physics_tps: int = 10
    linguistic_tps: int = 1
    num_agents: int = 10
    max_ticks: int = 10_000
    use_llm_announcer: bool = True
    announcer_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    headless: bool = False
    use_announcer: bool = False
    use_semantic: bool = True
    use_physics: bool = False
    semantic_encoder_device: str = 'cpu'


@dataclass
class Config:
    lrsg: LRSGConfig = field(default_factory=LRSGConfig)
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    episodic: EpisodicCompressorConfig = field(default_factory=EpisodicCompressorConfig)
