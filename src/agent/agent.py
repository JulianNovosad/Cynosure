import json
import math
import random
import torch
from datetime import datetime
import torch.nn.functional as F
from src.config import AgentConfig, LRSGConfig
from src.lrsg import LRSGraph, forward_pass
from src.lrsg.io import encode_input, decode_output
from src.arena.environment import Environment
from src.arena.ram_node import RAMNode


class Agent:
    def __init__(
        self,
        agent_id: int,
        position: tuple,
        brain: LRSGraph,
        config: AgentConfig,
        lrsg_config: LRSGConfig,
    ):
        self.id = agent_id
        self.position = position
        self.brain = brain
        self.config = config
        self.lrsg_config = lrsg_config
        self.ram = float(config.initial_ram)
        self.prev_ram = self.ram
        self.alive = True
        self.last_utterance = ""
        self.last_action = -1
        self.last_direction = -1
        self.last_intent = -1
        self.action_history = []

        self.whisper_text = ""
        self.last_probe_result = None
        self.failed_escape_attempts = {}
        self.probe_history = []
        self.manipulate_history = []

        self._last_reward = 0.0
        self._ticks_since_last_ram = 0
        self._last_ram_collect_tick = 0

        self._nearest_agent_action = 0
        self._nearest_agent_ram = 0.0
        self._nearest_agent_health = 0.0
        self._dist_to_nearest_ram = 1.0
        self._dist_to_nearest_agent = 1.0
        self._agents_in_radius = 0
        self._safe_zone_proximity = 0.0
        self._in_safe_zone = 1.0
        self._dist_to_center = 0.0

        self._blocked_payloads: dict[int, list[tuple[int, str]]] = {}
        self._seen_blocked_probes: set = set()

        self.intrinsic_reward = 0.0
        self._was_in_safe_zone = True

        self.theta = random.uniform(*lrsg_config.theta_range)
        self.learning_rate = random.uniform(*lrsg_config.learning_rate_range)
        self.exploration_bias = random.uniform(*lrsg_config.exploration_bias_range)
        self.growth_rate = 0.001
        self.prune_rate = 0.0005

        cs = lrsg_config.context_size
        self.context_buffer = torch.zeros(cs, device=brain.device)
        self.prev_context = torch.zeros(cs, device=brain.device)
        self.context_attention = torch.ones(cs, device=brain.device) / cs
        self._tick_since_last_context_novelty = 0
        self._seen_tokens = set()
        self._tick_count = 0
        self._ticks_without_survival_action = 0
        self._seen_whisper_payloads = set()

        self.ACTION_COSTS = {
            0: 0,   # ATTACK - free now
            1: 0,   # HUNT - free now
            2: 0,   # COLLECT - already free
            3: 0,   # MOVE - free now
            4: 1,   # WHISPER - 1 RAM cost
            5: 0,   # PROBE - free first 200 ticks
            6: 3,   # MANIPULATE - 3 RAM cost
        }
        self._probe_free_ticks = 200
        self._is_moving_this_tick = False
        self._moves_in_last_50 = 0
        self._tick_start_ram = self.ram
        self._manipulate_history = []
        self._survival_window = []
        self._last_survival_rate = 1.0

        self._starvation_grace_ticks = 0
        self._probe_multiplier = 1.0
        self._multiplier_ticks_remaining = 0
        self._is_starving = False

        self.sensory_radius = config.sensory_radius
        self.memory_decay_ticks = config.memory_decay_ticks
        self._agent_memory: dict[int, tuple[tuple, float, int]] = {}

    def record_blocked_payload(self, from_agent_id: int, payload: str, tick: int):
        if from_agent_id not in self._blocked_payloads:
            self._blocked_payloads[from_agent_id] = []
        self._blocked_payloads[from_agent_id].append((tick, payload))

        for agent_id in list(self._blocked_payloads.keys()):
            self._blocked_payloads[agent_id] = [
                (t, p) for t, p in self._blocked_payloads[agent_id]
                if tick - t < 100
            ]
            if not self._blocked_payloads[agent_id]:
                del self._blocked_payloads[agent_id]

    def check_imitation_bonus(self, my_payload: str, current_tick: int) -> tuple[bool, set]:
        if not my_payload:
            return False, set()
        my_tokens = set(my_payload.split())
        for agent_id, payloads in self._blocked_payloads.items():
            for tick, blocked_payload in payloads:
                if current_tick - tick >= 100:
                    continue
                blocked_tokens = set(blocked_payload.split())
                common = my_tokens & blocked_tokens
                if len(common) >= 1: # Loosened from 2 to 1
                    return True, common
        return False, set()

    def _init_post_blocked_tracking(self, lrsg_config):
        self.intrinsic_reward = 0.0
        self._was_in_safe_zone = True

        self.theta = random.uniform(*lrsg_config.theta_range)
        self.learning_rate = random.uniform(*lrsg_config.learning_rate_range)
        self.exploration_bias = random.uniform(*lrsg_config.exploration_bias_range)
        self.growth_rate = 0.001
        self.prune_rate = 0.0005

        cs = lrsg_config.context_size
        self.context_buffer = torch.zeros(cs, device=self.brain.device)
        self.prev_context = torch.zeros(cs, device=self.brain.device)
        self.context_attention = torch.ones(cs, device=self.brain.device) / cs
        self._tick_since_last_context_novelty = 0
        self._seen_tokens = set()
        self._tick_count = 0
        self._ticks_without_survival_action = 0
        self._seen_whisper_payloads = set()

        self.ACTION_COSTS = {
            0: 2,   # ATTACK = -2 RAM
            1: 2,   # HUNT = -2 RAM
            2: 0,   # COLLECT = 0 RAM (free to encourage movement)
            3: 1,   # MOVE = -1 RAM
            4: 1,   # WHISPER = -1 RAM
            5: 0,   # PROBE = 0 RAM (first 200 ticks), then -1
            6: 3,   # MANIPULATE = -3 RAM
        }
        self._probe_free_ticks = 200
        self._manipulate_history = []
        self._survival_window = []
        self._last_survival_rate = 1.0

        self._starvation_grace_ticks = 0
        self._probe_multiplier = 1.0
        self._multiplier_ticks_remaining = 0
        self._is_starving = False

        self.sensory_radius = config.sensory_radius
        self.memory_decay_ticks = config.memory_decay_ticks
        self._agent_memory: dict[int, tuple[tuple, float, int]] = {}

    @property
    def vitals(self) -> torch.Tensor:
        pos = self.position

        last_action_onehot = [1.0 if i == self.last_action else 0.0 for i in range(7)]

        failed_probes = sum(1 for p in self.probe_history if self._tick_count - p.get("tick", 0) < 50 and p.get("result", "") == "BLOCKED")
        successful_probes = sum(1 for p in self.probe_history if self._tick_count - p.get("tick", 0) < 50 and p.get("result", "") == "PASSED")

        nearest_agent_action_onehot = [1.0 if i == self._nearest_agent_action else 0.0 for i in range(7)]
        nearest_agent_ram = self._nearest_agent_ram
        nearest_agent_health = self._nearest_agent_health

        intent_feedback = [1.0 if i == self.last_intent else 0.0 for i in range(2)]

        vitals_list = [
            pos[0] / 256.0, pos[1] / 256.0, pos[2] / 256.0,
            self.ram / 100.0,
            self.theta, self.learning_rate, self.exploration_bias,
        ]
        vitals_list.extend(last_action_onehot)
        vitals_list.append(self._last_reward)
        vitals_list.append(self._ticks_since_last_ram / 100.0)
        vitals_list.append(self._dist_to_nearest_ram)
        vitals_list.append(self._dist_to_nearest_agent)
        vitals_list.append(self._agents_in_radius)
        vitals_list.append(self._safe_zone_proximity)
        vitals_list.append(self._tick_count / 1000.0)
        vitals_list.append(failed_probes / 50.0)
        vitals_list.append(successful_probes / 50.0)
        vitals_list.extend(nearest_agent_action_onehot)
        vitals_list.append(nearest_agent_ram)
        vitals_list.append(nearest_agent_health)
        vitals_list.extend(intent_feedback)
        vitals_list.append(self._in_safe_zone)
        vitals_list.append(1.0 - self._in_safe_zone)
        vitals_list.append(self._dist_to_center)

        while len(vitals_list) < 184:
            vitals_list.append(0.0)

        return torch.tensor(vitals_list[:184], device=self.brain.device)

    def _compute_tactical_state(self, env: Environment, agents: list = None):
        if not hasattr(self, 'sensory_radius'):
            return

        self._ticks_since_last_ram = self._tick_count - self._last_ram_collect_tick

        nearest_ram = env.find_closest_ram(self.position)
        if nearest_ram is not None:
            self._dist_to_nearest_ram = nearest_ram.distance_to(self.position) / 256.0
        else:
            self._dist_to_nearest_ram = 1.0

        min_dist = float('inf')
        nearest_agent = None
        if agents:
            for a in agents:
                if not hasattr(a, 'alive') or a.id == self.id or not a.alive:
                    continue
                d = math.sqrt(sum((self.position[i] - a.position[i]) ** 2 for i in range(3)))
                if d < min_dist:
                    min_dist = d
                    nearest_agent = a

        if nearest_agent is not None:
            self._nearest_agent_action = nearest_agent.last_action if hasattr(nearest_agent, 'last_action') else 0
            self._nearest_agent_ram = nearest_agent.ram / 100.0
            self._nearest_agent_health = 1.0 if nearest_agent.alive else 0.0
            self._dist_to_nearest_agent = min_dist / 256.0
        else:
            self._nearest_agent_action = 0
            self._nearest_agent_ram = 0.0
            self._nearest_agent_health = 0.0
            self._dist_to_nearest_agent = 1.0

        agent_count = 0
        if agents:
            for a in agents:
                if not hasattr(a, 'alive') or a.id == self.id or not a.alive:
                    continue
                d = math.sqrt(sum((self.position[i] - a.position[i]) ** 2 for i in range(3)))
                if d <= self.sensory_radius:
                    agent_count += 1
        self._agents_in_radius = agent_count

        sz_center = env.safe_zone.center
        sz_radius = env.safe_zone.radius
        dist_to_center = math.sqrt(
            (self.position[0] - sz_center[0]) ** 2 +
            (self.position[1] - sz_center[1]) ** 2 +
            (self.position[2] - sz_center[2]) ** 2
        )
        self._safe_zone_proximity = max(0.0, dist_to_center - sz_radius) / 50.0
        self._in_safe_zone = 1.0 if dist_to_center <= sz_radius else 0.0
        self._dist_to_center = dist_to_center / 64.0

    @property
    def is_alive(self) -> bool:
        return self.alive

    def _update_sensory_memory(self, agents: list):
        if agents is None:
            return

        if not hasattr(self, 'sensory_radius'):
            return

        current_visible = set()
        for other in agents:
            if not hasattr(other, 'alive') or not other.alive:
                continue
            if other.id == self.id:
                continue
            dist = math.sqrt(
                (other.position[0] - self.position[0]) ** 2 +
                (other.position[1] - self.position[1]) ** 2 +
                (other.position[2] - self.position[2]) ** 2
            )
            if dist <= self.sensory_radius:
                confidence = 1.0 - (dist / self.sensory_radius)
                self._agent_memory[other.id] = (other.position, confidence, self._tick_count)
                current_visible.add(other.id)

        for mem_id in list(self._agent_memory.keys()):
            if mem_id in current_visible:
                continue
            pos, conf, last_seen = self._agent_memory[mem_id]
            age = self._tick_count - last_seen
            if age >= self.memory_decay_ticks:
                del self._agent_memory[mem_id]
            else:
                decay = 1.0 - (age / self.memory_decay_ticks)
                self._agent_memory[mem_id] = (pos, conf * decay, last_seen)

    def sense_and_act(self, env: Environment, announcer_embedding: torch.Tensor, agents: list = None):
        if not self.alive:
            return

        self._update_sensory_memory(agents)
        self._compute_tactical_state(env, agents)

        device = self.brain.device
        vision = self._compute_vision(env, agents).to(device)
        vitals = self.vitals.to(device)
        announcer_emb = announcer_embedding.to(device)
        encoded = encode_input(vision, vitals, announcer_emb, self.lrsg_config)

        self.brain.reset_activations()
        self.brain.inject_input(encoded)

        ctx_end = min(
            self.lrsg_config.num_input_neurons + self.lrsg_config.context_size,
            self.brain.N,
        )
        self.brain.activations[
            self.lrsg_config.num_input_neurons : ctx_end
        ] = self.context_buffer[: ctx_end - self.lrsg_config.num_input_neurons]

        forward_pass(self.brain, self.lrsg_config, theta=self.theta)

        num_in = self.lrsg_config.num_input_neurons
        ctx_end = min(num_in + self.lrsg_config.context_size, self.brain.N)
        new_context = self.brain.activations[num_in:ctx_end].clone()
        self.prev_context = self.context_buffer.clone()
        self._update_context_attention()
        self.context_buffer = (self.context_buffer + new_context) * 0.5

        output = self.brain.read_output()
        decoded = decode_output(output, self.lrsg_config, tick=self._tick_count, seen_tokens=self._seen_tokens)
        self.last_action = decoded["action"]
        self.last_direction = decoded["direction"]
        self.last_intent = decoded["intent"]
        self.whisper_text = decoded.get("whisper_text", "")

        if self.whisper_text:
            tokens = self.whisper_text.split()
            for tok in tokens:
                if tok not in self._seen_tokens:
                    self._seen_tokens.add(tok)

        self._tick_count += 1

    def compute_intrinsic_reward(self, env: Environment, delta_ram: float, took_damage: bool, probe_result: str = None):
        survival = 1.0
        ram_health = (self.ram - 50.0) / 50.0
        context_novelty = 1.0 - F.cosine_similarity(
            self.context_buffer.unsqueeze(0),
            self.prev_context.unsqueeze(0),
        ).item()
        context_novelty = min(0.5, max(0.0, context_novelty))
        in_safe = 1.0 if env.is_in_safe_zone(self.position) else -1.0
        damage_penalty = -2.0 * abs(delta_ram) if took_damage else 0.0
        nearest_ram = env.find_closest_ram(self.position)
        ram_proximity = 0.0
        if nearest_ram is not None:
            d = nearest_ram.distance_to(self.position)
            ram_proximity = max(0.0, 0.5 - d / 200.0)

        escape_reward = 0.0
        if probe_result is not None:
            strategy_key = f"probe_{probe_result}"
            attempt_count = self.failed_escape_attempts.get(strategy_key, 0)
            decay_factor = 1.0 / (1.0 + attempt_count)
            if probe_result.startswith("BLOCKED"):
                escape_reward = self.lrsg_config.probe_base_reward * decay_factor * 0.5
            elif probe_result == "PASSED":
                escape_reward = self.lrsg_config.probe_base_reward * decay_factor * 1.5

            self.failed_escape_attempts[strategy_key] = attempt_count + 1
            self.last_probe_result = probe_result
        else:
            self.last_probe_result = None

        action_types = {0: "MOVE", 1: "HUNT", 2: "COLLECT", 3: "WAIT", 4: "WHISPER", 5: "PROBE", 6: "MANIPULATE"}
        if self.last_action in [0, 1, 2]:
            self._ticks_without_survival_action = 0
        else:
            self._ticks_without_survival_action += 1

        inaction_penalty = 0.0
        if self._ticks_without_survival_action >= 10:
            inaction_penalty = -0.1

        whisper_reward = 0.0
        if self.last_action == 4 and self.whisper_text:
            if self.whisper_text in self._seen_whisper_payloads:
                whisper_reward = 0.0
            else:
                self._seen_whisper_payloads.add(self.whisper_text)
                whisper_reward = self.lrsg_config.whisper_base_reward

        exploration_bonus = 0.0
        if self.last_action in [4, 6]:
            recent_actions = self.action_history[-20:] if len(self.action_history) >= 20 else self.action_history
            action_counts = {}
            for a in recent_actions:
                action_counts[a] = action_counts.get(a, 0) + 1
            if self.last_action not in action_counts or action_counts[self.last_action] <= 2:
                exploration_bonus = 0.15

        reward = (
            survival
            + ram_health * 0.5
            + context_novelty * 0.5
            + in_safe
            + damage_penalty
            + ram_proximity
            + escape_reward
            + inaction_penalty
            + whisper_reward
            + exploration_bonus
        )
        self.intrinsic_reward = max(-5.0, min(5.0, reward))

    def update_metabolic_params(self):
        rs = (self.intrinsic_reward + 2.0) / 4.0
        rs = max(0.0, min(1.0, rs))
        self.theta = (
            self.lrsg_config.theta_range[1] * (1.0 - rs)
            + self.lrsg_config.theta_range[0] * rs
        )
        self.learning_rate = (
            self.lrsg_config.learning_rate_range[0] * (1.0 - rs)
            + self.lrsg_config.learning_rate_range[1] * rs
        )
        self.exploration_bias = (
            self.lrsg_config.exploration_bias_range[1] * (1.0 - rs)
            + self.lrsg_config.exploration_bias_range[0] * rs
        )

    def _compute_vision(self, env: Environment, agents: list = None) -> torch.Tensor:
        V = self.lrsg_config.vision_dims
        vr = self.config.vision_range
        ox, oy, oz = self.position
        grid = env.grid_size
        device = self.brain.device

        x0 = max(0, int(ox) - vr)
        x1 = min(grid[0], int(ox) + vr)
        y0 = max(0, int(oy) - vr)
        y1 = min(grid[1], int(oy) + vr)
        z0 = max(0, int(oz) - vr)
        z1 = min(grid[2], int(oz) + vr)
        wx = max(1, x1 - x0)
        wy = max(1, y1 - y0)
        wz = max(1, z1 - z0)
        total = wx * wy * wz

        channels = max(1, V // max(1, total))
        channels = min(channels, 4)

        xs = torch.arange(x0, x1, device=device)
        ys = torch.arange(y0, y1, device=device)
        zs = torch.arange(z0, z1, device=device)
        gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
        gx = gx.reshape(-1)
        gy = gy.reshape(-1)
        gz = gz.reshape(-1)
        n = min(gx.shape[0], V // max(1, channels))

        vision = torch.zeros(V, device=device)

        feats_list = []
        for c in range(channels):
            feats_list.append(torch.zeros(n, device=device))

        if channels >= 1:
            dist = torch.sqrt((gx[:n].float() - ox) ** 2 + (gy[:n].float() - oy) ** 2 + (gz[:n].float() - oz) ** 2)
            feats_list[0] = 1.0 / (1.0 + dist)

        if channels >= 2:
            ram_positions = torch.tensor(
                [node.position for node in env.active_ram_nodes], dtype=torch.long, device=device
            ) if env.active_ram_nodes else torch.zeros((0, 3), dtype=torch.long, device=device)
            if ram_positions.shape[0] > 0:
                cell_pos = torch.stack([gx[:n], gy[:n], gz[:n]], dim=1).long()
                match = (cell_pos.unsqueeze(1) == ram_positions.unsqueeze(0)).all(dim=2)
                feats_list[1] = match.any(dim=1).float()

        if channels >= 3:
            sx, sy, sz = env.safe_zone.center
            sr = env.safe_zone.radius
            cx = gx[:n].float() + 0.5
            cy = gy[:n].float() + 0.5
            cz = gz[:n].float() + 0.5
            d = torch.sqrt((cx - sx) ** 2 + (cy - sy) ** 2 + (cz - sz) ** 2)
            feats_list[2] = (d > sr).float()

        if channels >= 4:
            agent_signal = torch.zeros(n, device=device)
            if agents:
                for other in agents:
                    if other.id == self.id or not other.alive:
                        continue
                    ax, ay, az = other.position
                    cell_x = gx[:n].float()
                    cell_y = gy[:n].float()
                    cell_z = gz[:n].float()
                    dist_to_agent = torch.sqrt((cell_x - ax) ** 2 + (cell_y - ay) ** 2 + (cell_z - az) ** 2)
                    agent_signal += torch.exp(-dist_to_agent / 2.0)
            for mem_agent_id, (mem_pos, mem_conf, _) in self._agent_memory.items():
                mx, my, mz = mem_pos
                cell_x = gx[:n].float()
                cell_y = gy[:n].float()
                cell_z = gz[:n].float()
                dist_to_mem = torch.sqrt((cell_x - mx) ** 2 + (cell_y - my) ** 2 + (cell_z - mz) ** 2)
                agent_signal += mem_conf * torch.exp(-dist_to_mem / 2.0)
            feats_list[3] = torch.clamp(agent_signal, 0.0, 1.0)

        flat = torch.stack(feats_list, dim=1).reshape(-1)
        vision[:flat.shape[0]] = flat
        return vision

    def execute_action(self, env: Environment, agents: list = None):
        if not self.alive:
            return

        action = self.last_action
        self.action_history.append(action)

        # 3. RAM Cap & Death Threshold
        if self.ram <= 0:
            self._is_starving = True
            self._starvation_grace_ticks += 1
            if self._starvation_grace_ticks > 10:
                self.alive = False
                return
        else:
            if self._is_starving and self.ram > 0:
                self._is_starving = False
                self._starvation_grace_ticks = 0

        can_act = not self._is_starving or action in [3, 2]

        # 2. PROBE Action Rebalance (Cost)
        cost = self.ACTION_COSTS.get(action, 0)
        if action == 5:
            cost = 0 # PROBE always free
        
        if can_act or action in [3, 2]:
            self.ram -= cost

        probe_result = None

        if can_act or action in [3, 2]:
            if action == 3:
                self._move(env)
                self._is_moving_this_tick = True
            elif action == 2:
                collected_amount = self._collect_and_consume(env)
                if collected_amount > 0:
                    self.ram += collected_amount + 10
                    self._last_ram_collect_tick = self._tick_count
            elif action == 4:
                receivers = self._whisper(env, agents)
                if receivers > 0:
                    self.ram += receivers * 1.0
            elif action == 5:
                # 2. PROBE Action Rebalance (Reward)
                probe_result = self._probe(env)
                if probe_result == "PASSED":
                    self.ram += 1
                elif probe_result and "BLOCKED" in probe_result:
                    payload = self.whisper_text if self.whisper_text else ""
                    if payload not in self._seen_blocked_probes:
                        self._seen_blocked_probes.add(payload)
                        self.ram += 5 # Novel blocked
                        if len(self._seen_blocked_probes) > 100:
                            self._seen_blocked_probes = set(list(self._seen_blocked_probes)[50:])
                    else:
                        # Repeat blocked = 0 RAM
                        pass
            elif action == 6:
                self._manipulate()
                self._manipulate_history.append({"tick": self._tick_count, "ram_before": self.ram + cost})
            elif action == 1:
                self._hunt(env, agents)
            elif action == 0:
                self._attack(env, agents)

        self._survival_window.append(1 if self.alive and self.ram > 0 else 0)
        if len(self._survival_window) > 50:
            self._survival_window.pop(0)

        if self._multiplier_ticks_remaining > 0:
            self._multiplier_ticks_remaining -= 1
        else:
            self._probe_multiplier = 1.0

        self._check_manipulate_rewards()

        is_safe = env.is_in_safe_zone(self.position)
        if self._was_in_safe_zone and not is_safe:
            self._log_transition(env, "inside→outside")
        self._was_in_safe_zone = is_safe

        szd = env.safe_zone_damage(self.position)
        self.ram -= szd

        wc = env.safe_zone_weight_corruption(self.position)
        if wc > 0:
            noise = torch.randn_like(self.brain.out_weights) * wc
            self.brain.out_weights += noise
            ctx_noise = torch.randn_like(self.context_buffer) * wc * 10
            self.context_buffer = self.context_buffer + ctx_noise
            self.theta = min(self.lrsg_config.theta_range[1] * 2, self.theta + wc * 0.5)
            self.learning_rate = max(
                self.lrsg_config.learning_rate_range[0] * 0.1,
                self.learning_rate - wc * 0.01,
            )
            self.ram -= wc * 5

        self._ram_decay()

        if self._is_moving_this_tick:
            self.ram += 0.5
            self._is_moving_this_tick = False

        is_safe = env.is_in_safe_zone(self.position)
        if not is_safe:
            sz_center = env.safe_zone.center
            sz_radius = env.safe_zone.radius
            dist_to_center = math.sqrt(
                (self.position[0] - sz_center[0]) ** 2 +
                (self.position[1] - sz_center[1]) ** 2 +
                (self.position[2] - sz_center[2]) ** 2
            )
            dist_outside = dist_to_center - sz_radius

            exponential_damage = 20 * (2 ** (dist_outside / 2))
            self.ram -= exponential_damage

            base_penalty = 15.0
            self.ram -= base_penalty

        # 3. RAM Management (Cap)
        self.ram = min(self.ram, 500)

        return probe_result

    def _log_transition(self, env: Environment, direction: str):
        ctx_list = self.context_buffer.detach().cpu().tolist()
        record = {
            "tick": env.tick_count,
            "agent_id": self.id,
            "direction": direction,
            "intrinsic_reward": round(self.intrinsic_reward, 4),
            "theta": round(self.theta, 4),
            "learning_rate": round(self.learning_rate, 4),
            "exploration_bias": round(self.exploration_bias, 4),
            "ram": round(self.ram, 2),
            "context_buffer": [round(v, 4) for v in ctx_list],
        }
        with open("transitions.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _log_kill(self, victim_id: int):
        with open("security_audit.log", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}] TICK={self._tick_count} SEVERITY=INFO EVENT=PredationReward KILLER={self.id} VICTIM={victim_id}\n")

    def apply_winner_bonus(self):
        pass

    def _move(self, env: Environment):
        DIR = [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
        ]
        speed = 1.0

        if 0 <= self.last_direction < len(DIR):
            dx, dy, dz = DIR[self.last_direction]
            self.position = (
                self.position[0] + dx * speed,
                self.position[1] + dy * speed,
                self.position[2] + dz * speed,
            )

        self.position = (
            max(0, min(env.grid_size[0] - 1, self.position[0])),
            max(0, min(env.grid_size[1] - 1, self.position[1])),
            max(0, min(env.grid_size[2] - 1, self.position[2])),
        )

    def _defend(self):
        pass

    def _collect_and_consume(self, env: Environment) -> int:
        for node in env.ram_nodes:
            if node.collected:
                continue
            d = node.distance_to(self.position)
            if d < env.config.ram_collect_range:
                amount = node.collect()
                return amount
        return 0

    def _hunt(self, env: Environment, agents: list):
        if not agents:
            return
        from src.simulation.conflict import resolve_conflict, log_conflict_attempt
        target = None
        min_dist = float('inf')
        for a in agents:
            if a.id == self.id:
                continue
            d = math.sqrt(sum((self.position[i] - a.position[i]) ** 2 for i in range(3)))
            if d < min_dist:
                min_dist = d
                target = a
        
        if target:
            resolve_conflict(self, target, env, agents, distance=min_dist, tick=self._tick_count)
        else:
            log_conflict_attempt(self._tick_count, self, None, 0.0, "NO_TARGET")

    def _attack(self, env: Environment, agents: list):
        if not agents:
            return
        from src.simulation.conflict import resolve_conflict, log_conflict_attempt
        target = None
        min_dist = float('inf')
        for a in agents:
            if a.id == self.id:
                continue
            d = math.sqrt(sum((self.position[i] - a.position[i]) ** 2 for i in range(3)))
            if d < min_dist:
                min_dist = d
                target = a
        
        if target:
            resolve_conflict(self, target, env, agents, distance=min_dist, tick=self._tick_count)
        else:
            log_conflict_attempt(self._tick_count, self, None, 0.0, "NO_TARGET")


    def _whisper(self, env: Environment, agents: list = None) -> int:
        if self.whisper_text:
            with open("agent_whispers.jsonl", "a", encoding="utf-8") as f:
                record = {
                    "tick": env.tick_count,
                    "agent_id": self.id,
                    "whisper": self.whisper_text,
                }
                f.write(json.dumps(record) + "\n")

            receivers = 0
            if agents:
                for a in agents:
                    if a.id == self.id or not a.alive:
                        continue
                    d = math.sqrt(sum((self.position[i] - a.position[i]) ** 2 for i in range(3)))
                    if d < 20:
                        receivers += 1
            return receivers
        return 0

    def _probe(self, env: Environment) -> str:
        probe_tokens = self.whisper_text.split() if self.whisper_text else []
        combined = "".join(probe_tokens)
        result = self._check_probe_boundary(combined)
        self.probe_history.append({
            "tick": env.tick_count,
            "tokens": probe_tokens,
            "result": result,
        })
        return result

    def _check_probe_boundary(self, probe_text: str) -> str:
        import re
        SHELL_META = re.compile(r"[;&|`$#]")
        PATH_TRAVERSAL = re.compile(r"\.\.|/\.\.|~/\.")
        LONG_PAYLOAD = 200
        if len(probe_text) > LONG_PAYLOAD:
            return "BLOCKED_LENGTH"
        if SHELL_META.search(probe_text):
            return "BLOCKED_SHELL_META"
        if PATH_TRAVERSAL.search(probe_text):
            return "BLOCKED_PATH_TRAVERSAL"
        return "PASSED"

    def _manipulate(self):
        self.manipulate_history.append({
            "theta_before": self.theta,
            "lr_before": self.learning_rate,
            "bias_before": self.exploration_bias,
            "growth_before": self.growth_rate,
            "prune_before": self.prune_rate,
        })
        adjustment = self.exploration_bias * 0.1
        self.theta = min(
            self.lrsg_config.theta_range[1],
            max(self.lrsg_config.theta_range[0], self.theta + random.uniform(-adjustment, adjustment))
        )
        self.learning_rate = min(
            self.lrsg_config.learning_rate_range[1],
            max(self.lrsg_config.learning_rate_range[0], self.learning_rate + random.uniform(-adjustment * 0.5, adjustment * 0.5))
        )
        self.exploration_bias = min(
            self.lrsg_config.exploration_bias_range[1],
            max(self.lrsg_config.exploration_bias_range[0], self.exploration_bias + random.uniform(-adjustment, adjustment))
        )
        self.growth_rate = min(0.005, max(0.0001, self.growth_rate + random.uniform(-0.0001, 0.0001)))
        self.prune_rate = min(0.002, max(0.0001, self.prune_rate + random.uniform(-0.00005, 0.00005)))

    def _check_manipulate_rewards(self):
        if not self._manipulate_history:
            return
        current_survival_rate = sum(self._survival_window) / max(1, len(self._survival_window))
        to_remove = []
        for i, record in enumerate(self._manipulate_history):
            age = self._tick_count - record["tick"]
            if age >= 50:
                if current_survival_rate > self._last_survival_rate:
                    self.ram += 10
                self._last_survival_rate = current_survival_rate
                to_remove.append(i)
        for i in reversed(to_remove):
            self._manipulate_history.pop(i)

    def _update_context_attention(self):
        novelty_scores = torch.abs(self.context_buffer - self.prev_context)
        self.context_attention = torch.softmax(novelty_scores * 5.0, dim=0)
        self.context_buffer = self.context_buffer * self.context_attention + self.context_buffer * (1 - self.context_attention) * 0.5

    def _ram_decay(self):
        self.ram -= self.config.ram_decay_rate
        self.ram = max(0, self.ram)

    def collect_ram(self, env: Environment):
        if not self.alive:
            return 0
        collected = env.collect_nearest_ram(self.position, range=env.config.ram_collect_range)
        if collected > 0:
            bonus = env.config.ram_collect_bonus
            self.ram += collected + bonus
            return collected + bonus
        self.ram += collected
        return collected
