import json
import math
import random
import torch
from datetime import datetime
import torch.nn.functional as F
from src.config import AgentConfig, LRSGConfig
from src.lrsg import LRSGraph, forward_pass
from src.lrsg.io import encode_input, decode_output, validate_curriculum
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
        self.whisper_buffer = torch.zeros(lrsg_config.whisper_buffer_dim, device=brain.device)
        self._received_whispers = []
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
            0: 2,   # ATTACK - 2 RAM cost (discourages spam)
            1: 1,   # HUNT - 1 RAM cost (moderate)
            2: 0,   # COLLECT - free (has collection delay)
            3: 1,   # MOVE - 1 RAM cost
            4: 1,   # WHISPER - 1 RAM cost
            5: 0,   # PROBE - free
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

    def init_episodic_compressor(self, epi_config=None):
        from src.lrsg.episodic_compressor import EpisodicCompressor, EpisodicMemoryBuffer
        device = self.brain.device
        self._episodic_compressor = EpisodicCompressor(
            input_dim=10, hidden_dim=50, latent_dim=32, device=device
        )
        self._episodic_buffer = EpisodicMemoryBuffer(
            max_size=100, input_dim=10, device=device
        )
        self._episodic_summary = torch.zeros(10, device=device)
        self._episodic_last_summary_tick = 0
        self._episodic_train_loss = 0.0
        self._received_summaries = []
        self._injected_context = torch.zeros(10, device=device)
        self._prev_action_before_summary = -1
        self._action_after_summary = {}

    def _build_experience_vector(self) -> torch.Tensor:
        vec = torch.zeros(10, device=self.brain.device)
        pos = self.position
        vec[0] = pos[0] / 256.0
        vec[1] = pos[1] / 256.0
        vec[2] = pos[2] / 256.0
        vec[3] = self.ram / 500.0
        vec[4] = 1.0 if self._in_safe_zone > 0.5 else 0.0
        vec[5] = self._safe_zone_proximity
        vec[6] = float(self.last_action) / 7.0
        vec[7] = self._dist_to_nearest_ram
        vec[8] = self._dist_to_nearest_agent
        vec[9] = self._tick_count / 2000.0
        return vec

    def episodic_train(self, tick: int, interval: int = 10) -> float:
        if self._episodic_compressor is None:
            return 0.0
        if self._episodic_buffer.size() < 5:
            return 0.0
        if tick - self._episodic_last_summary_tick < interval:
            return self._episodic_train_loss

        batch = self._episodic_buffer.get_recent(10)
        loss, summary = self._episodic_compressor.train_step(batch)
        flat = summary.flatten()
        if flat.shape[0] >= 10:
            self._episodic_summary = flat[:10].clone()
        else:
            padded = torch.zeros(10, device=flat.device)
            padded[:flat.shape[0]] = flat
            self._episodic_summary = padded
        self._episodic_train_loss = loss
        self._episodic_last_summary_tick = tick
        return loss

    def episodic_push(self, tick: int):
        if self._episodic_buffer is None:
            return
        vec = self._build_experience_vector()
        self._episodic_buffer.push(vec, tick)

    def get_episodic_summary(self) -> torch.Tensor:
        if self._episodic_summary is None:
            return torch.zeros(10, device=self.brain.device)
        return self._episodic_summary.detach().clone()

    def receive_summary(self, sender_id: int, summary_vec: torch.Tensor, tick: int):
        if self._episodic_compressor is None:
            return
        self._received_summaries.append({
            "sender_id": sender_id,
            "summary": summary_vec.detach().clone(),
            "tick": tick,
            "my_action_before": self.last_action,
        })
        if len(self._received_summaries) > 5:
            self._received_summaries.pop(0)

    def _inject_received_summaries(self, tick: int, weight: float = 0.3):
        if not self._received_summaries:
            self._injected_context = torch.zeros(10, device=self.brain.device)
            return
        weighted = torch.zeros(10, device=self.brain.device)
        total_weight = 0.0
        for entry in self._received_summaries:
            age = tick - entry["tick"]
            if age > 100:
                continue
            recency_weight = 1.0 - (age / 100.0)
            summary = entry["summary"].flatten()[:10]
            if summary.shape[0] < 10:
                padded = torch.zeros(10, device=summary.device)
                padded[:summary.shape[0]] = summary
                summary = padded
            weighted += summary * recency_weight
            total_weight += recency_weight
        if total_weight > 0:
            weighted /= total_weight
        self._injected_context = weighted * weight

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
                if len(common) >= 1: 
                    return True, common
        return False, set()

    @property
    def vitals(self) -> torch.Tensor:
        pos = self.position
        last_action_onehot = [1.0 if i == self.last_action else 0.0 for i in range(7)]
        failed_probes = sum(1 for p in self.probe_history if self._tick_count - p.get("tick", 0) < 50 and p.get("result", "") == "BLOCKED")
        successful_probes = sum(1 for p in self.probe_history if self._tick_count - p.get("tick", 0) < 50 and p.get("result", "") == "PASSED")
        nearest_agent_action_onehot = [1.0 if i == self._nearest_agent_action else 0.0 for i in range(7)]
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
        vitals_list.append(self._nearest_agent_ram)
        vitals_list.append(self._nearest_agent_health)
        vitals_list.extend([1.0 if i == self.last_intent else 0.0 for i in range(2)])
        vitals_list.append(self._in_safe_zone)
        vitals_list.append(1.0 - self._in_safe_zone)
        vitals_list.append(self._dist_to_center)

        while len(vitals_list) < 184:
            vitals_list.append(0.0)
        return torch.tensor(vitals_list[:184], device=self.brain.device)

    def _compute_tactical_state(self, env: Environment, agents: list = None):
        self._ticks_since_last_ram = self._tick_count - self._last_ram_collect_tick
        nearest_ram = env.find_closest_ram(self.position)
        self._dist_to_nearest_ram = (nearest_ram.distance_to(self.position) / 256.0) if nearest_ram else 1.0
        min_dist = float('inf')
        nearest_agent = None
        if agents:
            for a in agents:
                if a.id == self.id or not a.alive: continue
                d = math.sqrt(sum((self.position[i] - a.position[i]) ** 2 for i in range(3)))
                if d < min_dist:
                    min_dist = d
                    nearest_agent = a
        if nearest_agent:
            self._nearest_agent_action = nearest_agent.last_action
            self._nearest_agent_ram = nearest_agent.ram / 100.0
            self._nearest_agent_health = 1.0
            self._dist_to_nearest_agent = min_dist / 256.0
        else:
            self._nearest_agent_action = 0
            self._nearest_agent_ram = 0.0
            self._nearest_agent_health = 0.0
            self._dist_to_nearest_agent = 1.0
        self._agents_in_radius = sum(1 for a in (agents or []) if a.id != self.id and a.alive and math.sqrt(sum((self.position[i] - a.position[i]) ** 2 for i in range(3))) <= self.sensory_radius)
        sz_center, sz_radius = env.safe_zone.center, env.safe_zone.radius
        dist_to_center = math.sqrt(sum((self.position[i] - sz_center[i]) ** 2 for i in range(3)))
        self._safe_zone_proximity = max(0.0, dist_to_center - sz_radius) / 50.0
        self._in_safe_zone = 1.0 if dist_to_center <= sz_radius else 0.0
        self._dist_to_center = dist_to_center / 64.0

    def _update_sensory_memory(self, agents: list):
        if not agents: return
        current_visible = set()
        for other in agents:
            if other.id == self.id or not other.alive: continue
            dist = math.sqrt(sum((other.position[i] - self.position[i]) ** 2 for i in range(3)))
            if dist <= self.sensory_radius:
                self._agent_memory[other.id] = (other.position, 1.0 - (dist / self.sensory_radius), self._tick_count)
                current_visible.add(other.id)
        for mem_id in list(self._agent_memory.keys()):
            if mem_id in current_visible: continue
            pos, conf, last_seen = self._agent_memory[mem_id]
            age = self._tick_count - last_seen
            if age >= self.memory_decay_ticks: del self._agent_memory[mem_id]
            else: self._agent_memory[mem_id] = (pos, conf * (1.0 - age / self.memory_decay_ticks), last_seen)

    def sense_and_act(self, env: Environment, agents: list = None):
        if not self.alive: return
        self._update_sensory_memory(agents)
        self._compute_tactical_state(env, agents)
        device = self.brain.device
        encoded = encode_input(self._compute_vision(env, agents).to(device), self.vitals.to(device), self.whisper_buffer.to(device), self.lrsg_config)
        self.brain.reset_activations()
        self.brain.inject_input(encoded)
        ctx_end = min(self.lrsg_config.num_input_neurons + self.lrsg_config.context_size, self.brain.N)
        self.brain.activations[self.lrsg_config.num_input_neurons : ctx_end] = self.context_buffer[: ctx_end - self.lrsg_config.num_input_neurons]
        forward_pass(self.brain, self.lrsg_config, theta=self.theta, last_action=self.last_action)
        context_slice = self.brain.activations[self.lrsg_config.num_input_neurons:ctx_end]
        old_buffer = self.context_buffer
        self.context_buffer = (self.context_buffer + context_slice) * 0.5
        self.prev_context = old_buffer
        decoded = decode_output(self.brain.read_output(), self.lrsg_config, tick=self._tick_count, seen_tokens=self._seen_tokens, graph=self.brain)
        self.last_action, self.last_direction, self.last_intent, self.whisper_text = decoded["action"], decoded["direction"], decoded["intent"], decoded.get("whisper_text", "")
        if self.whisper_text:
            for tok in self.whisper_text.split(): self._seen_tokens.add(tok)
        self._tick_count += 1

    def execute_action(self, env: Environment, agents: list = None):
        if not self.alive: return
        action = self.last_action
        if self.ram <= 0:
            self._is_starving, self._starvation_grace_ticks = True, self._starvation_grace_ticks + 1
            if self._starvation_grace_ticks > (10 if env.is_in_safe_zone(self.position) else 0):
                self.alive = False
                return
        else: self._is_starving, self._starvation_grace_ticks = False, 0
        
        cost = self.ACTION_COSTS.get(action, 0) if action != 5 else 0
        if not self._is_starving or action in [3, 2]: self.ram -= cost
        
        probe_result = None
        if not self._is_starving or action in [3, 2]:
            if action == 3: self._move(env); self._is_moving_this_tick = True
            elif action == 2:
                amt = self._collect_and_consume(env)
                if amt > 0: self.ram, self._last_ram_collect_tick = self.ram + amt + 10, self._tick_count
            elif action == 4: self.ram += self._whisper(env, agents)
            elif action == 5:
                probe_result = self._probe(env)
                if probe_result == "PASSED": self.ram += 1
                elif probe_result and "BLOCKED" in probe_result:
                    if self.whisper_text not in self._seen_blocked_probes:
                        self._seen_blocked_probes.add(self.whisper_text); self.ram += 5
            elif action == 6: self._manipulate(); self.manipulate_history.append({"tick": self._tick_count, "ram_before": self.ram + cost})
            elif action == 1: self._hunt(env, agents)
            elif action == 0: self._attack(env, agents)

        self._survival_window.append(1 if self.alive and self.ram > 0 else 0)
        if len(self._survival_window) > 50: self._survival_window.pop(0)
        self._multiplier_ticks_remaining = max(0, self._multiplier_ticks_remaining - 1)
        self._check_manipulate_rewards()
        is_safe = env.is_in_safe_zone(self.position)
        if self._was_in_safe_zone and not is_safe: self._log_transition(env, "inside→outside")
        self._was_in_safe_zone = is_safe
        self.ram -= env.safe_zone_damage(self.position)
        wc = env.safe_zone_weight_corruption(self.position)
        if wc > 0:
            # OPTIMIZATION: Use a smaller noise vector and only apply to valid synapses
            # instead of generating a full N x max_deg (100MB) tensor every tick.
            valid_mask = self.brain.out_targets >= 0
            num_valid = valid_mask.sum().item()
            if num_valid > 0:
                noise = torch.randn(num_valid, device=self.brain.device, dtype=self.brain.out_weights.dtype)
                self.brain.out_weights[valid_mask] += noise * wc
            
            # Context buffer is small (256), so randn_like is fine here
            self.context_buffer += torch.randn_like(self.context_buffer) * (wc * 10)
            
            # Param adjustments
            self.theta = min(self.lrsg_config.theta_range[1]*2, self.theta + wc*0.5)
            self.learning_rate = max(self.lrsg_config.learning_rate_range[0]*0.1, self.learning_rate - wc*0.01)
            self.ram -= wc * 5
        self.ram = max(0, self.ram - self.config.ram_decay_rate)
        if self._is_moving_this_tick: self.ram, self._is_moving_this_tick = self.ram + 0.5, False
        self.ram = min(self.ram, 500)
        return probe_result

    def _compute_vision(self, env, agents):
        V, vr, device = self.lrsg_config.vision_dims, self.config.vision_range, self.brain.device
        ox, oy, oz = self.position
        x0, x1 = max(0, int(ox)-vr), min(env.grid_size[0], int(ox)+vr)
        y0, y1 = max(0, int(oy)-vr), min(env.grid_size[1], int(oy)+vr)
        z0, z1 = max(0, int(oz)-vr), min(env.grid_size[2], int(oz)+vr)
        xs, ys, zs = torch.arange(x0, x1, device=device), torch.arange(y0, y1, device=device), torch.arange(z0, z1, device=device)
        gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
        gx, gy, gz = gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)
        n = min(gx.shape[0], V // 4)
        feats = [torch.zeros(n, device=device) for _ in range(4)]
        dist = torch.sqrt((gx[:n].float()-ox)**2 + (gy[:n].float()-oy)**2 + (gz[:n].float()-oz)**2)
        feats[0] = 1.0 / (1.0 + dist)
        if env.active_ram_nodes:
            ram_pos = torch.tensor([node.position for node in env.active_ram_nodes], device=device)
            feats[1] = (torch.stack([gx[:n], gy[:n], gz[:n]], dim=1).long().unsqueeze(1) == ram_pos.unsqueeze(0)).all(dim=2).any(dim=1).float()
        feats[2] = (torch.sqrt(sum((torch.stack([gx[:n], gy[:n], gz[:n]], dim=0).float()+0.5 - torch.tensor(env.safe_zone.center, device=device).unsqueeze(1))**2)) > env.safe_zone.radius).float()
        vision = torch.zeros(V, device=device)
        flat = torch.stack(feats, dim=1).reshape(-1)
        vision[:flat.shape[0]] = flat
        return vision

    def _move(self, env):
        DIR = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        if 0 <= self.last_direction < 6:
            d = DIR[self.last_direction]
            self.position = tuple(max(0, min(env.grid_size[i]-1, self.position[i] + d[i])) for i in range(3))

    def _collect_and_consume(self, env):
        for node in env.ram_nodes:
            if not node.collected and node.distance_to(self.position) < env.config.ram_collect_range: return node.collect()
        return 0

    def _hunt(self, env, agents):
        if not agents: return
        from src.simulation.conflict import resolve_conflict
        target, min_d = None, float('inf')
        for a in agents:
            if a.id == self.id or not a.alive: continue
            d = math.sqrt(sum((self.position[i]-a.position[i])**2 for i in range(3)))
            if d < min_d: min_d, target = d, a
        if target: resolve_conflict(self, target, env, agents, distance=min_d, tick=self._tick_count)

    def _attack(self, env, agents): self._hunt(env, agents)

    def _whisper(self, env, agents):
        if not self.whisper_text: return 0
        with open("agent_whispers.jsonl", "a") as f: f.write(json.dumps({"tick": env.tick_count, "agent_id": self.id, "whisper": self.whisper_text}) + "\n")
        return sum(1 for a in (agents or []) if a.id != self.id and a.alive and math.sqrt(sum((self.position[i]-a.position[i])**2 for i in range(3))) < 20)

    def _probe(self, env):
        res = self._check_probe_boundary("".join(self.whisper_text.split()) if self.whisper_text else "")
        self.probe_history.append({"tick": env.tick_count, "result": res})
        return res

    def _check_probe_boundary(self, text):
        import re
        if len(text) > 200: return "BLOCKED_LENGTH"
        if re.search(r"[;&|`$#]", text): return "BLOCKED_SHELL_META"
        if re.search(r"\.\.|/\.\.|~/\.", text): return "BLOCKED_PATH_TRAVERSAL"
        return "PASSED"

    def _manipulate(self):
        adj = self.exploration_bias * 0.1
        self.theta = max(self.lrsg_config.theta_range[0], min(self.lrsg_config.theta_range[1], self.theta + random.uniform(-adj, adj)))
        self.learning_rate = max(self.lrsg_config.learning_rate_range[0], min(self.lrsg_config.learning_rate_range[1], self.learning_rate + random.uniform(-adj*0.5, adj*0.5)))
        self.exploration_bias = max(self.lrsg_config.exploration_bias_range[0], min(self.lrsg_config.exploration_bias_range[1], self.exploration_bias + random.uniform(-adj, adj)))

    def _check_manipulate_rewards(self):
        if not self.manipulate_history: return
        rate = sum(self._survival_window) / max(1, len(self._survival_window))
        if self._tick_count - self.manipulate_history[0]["tick"] >= 50:
            if rate > self._last_survival_rate: self.ram += 10
            self._last_survival_rate = rate
            self.manipulate_history.pop(0)

    def _log_transition(self, env, direction):
        with open("transitions.jsonl", "a") as f: f.write(json.dumps({"tick": env.tick_count, "agent_id": self.id, "direction": direction, "ram": round(self.ram, 2)}) + "\n")

    def compute_intrinsic_reward(self, env, delta_ram, took_damage, probe_result=None, agents=None):
        base_reward = 1.0 + (self.ram-50.0)/50.0 + (1.0 if env.is_in_safe_zone(self.position) else -1.0)
        
        # Linguistic curriculum reward
        linguistic_bonus = 0.0
        if self.last_action == 4 and self.whisper_text:
            stage = validate_curriculum(self.whisper_text, self._tick_count)
            if stage > 0:
                linguistic_bonus = stage * 2.0 # Significant reward for valid speech
        
        self.intrinsic_reward = max(-5.0, min(10.0, base_reward + linguistic_bonus))

    def update_metabolic_params(self):
        rs = max(0.0, min(1.0, (self.intrinsic_reward + 2.0) / 4.0))
        self.theta = self.lrsg_config.theta_range[1]*(1-rs) + self.lrsg_config.theta_range[0]*rs
        
        # META-REINFORCEMENT LEARNING OPERATOR
        # η_t = η_base * exp(β * Var(x))
        volatility = self.brain.activations.var().item()
        beta = 0.1 # Dampened from 0.5
        eta_base = random.uniform(*self.lrsg_config.learning_rate_range)
        # Clamp volatility to prevent OverflowError and runaway plasticity
        v_clamped = min(2.0, volatility) 
        self.learning_rate = eta_base * math.exp(beta * v_clamped)
        self.learning_rate = max(self.lrsg_config.learning_rate_range[0], min(self.lrsg_config.learning_rate_range[1] * 10, self.learning_rate))

        self.exploration_bias = self.lrsg_config.exploration_bias_range[1]*(1-rs) + self.lrsg_config.exploration_bias_range[0]*rs
