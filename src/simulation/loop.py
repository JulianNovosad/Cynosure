import json
import os
import math
import random
import time
import torch
import torch.nn.functional as F
import psutil
from datetime import datetime
from src.config import Config
from src.arena import Environment
from src.lrsg import LRSGraph, forward_pass, hebbian_update, structural_plasticity
from src.agent import Agent
from src.announcer import Announcer
from src.simulation.conflict import resolve_conflict
from src.lrsg.io import encode_input, decode_output, encode_whisper_to_indices, validate_curriculum
from src.lrsg.semantic_encoder import SemanticEncoder
from src.simulation.sensors import SensorSimulator
from src.harness import (
    AnnouncerSanitizer,
    SecurityAuditLog,
    MetricValidator,
    ResourceMonitor,
    WatchdogTimer,
    TelemetryLogger,
    Dashboard,
    Checkpointer,
    ResourceExceededException,
)
try:
    from aurore.interlock_controller import InterlockController
except ImportError:
    class InterlockController:
        def __init__(self, *args, **kwargs): pass
        def is_actuation_allowed(self): return True

# ... (rest of the file)
from src.simulation.visualizer import SimulationVisualizer
from src.harness.telemetry import touch_file


ACTION_NAMES = ["ATTACK", "HUNT", "COLLECT", "MOVE", "WHISPER", "PROBE", "MANIPULATE"]


class Simulation:
    def __init__(self, config: Config, device: torch.device = torch.device("cpu")):
        self.config = config
        self.device = device
        self.tick_count = 0

        self.env = Environment(config.arena)

        self._audit_log = SecurityAuditLog()
        self._sanitizer = AnnouncerSanitizer(self._audit_log)
        self._metric_validator = MetricValidator(self._audit_log)
        self._resource_monitor = ResourceMonitor(audit_log=self._audit_log)
        self._watchdog = WatchdogTimer(audit_log=self._audit_log)

        self.announcer = Announcer(
            use_llm=config.simulation.use_llm_announcer,
            model_name=config.simulation.announcer_model,
            embed_dim=128, # Passive announcer dimension
            sanitizer=self._sanitizer,
        )

        self._agent_births = {}
        self._agent_deaths = {}
        self._init_edge_counts = {}
        self._init_weight_sums = {}
        self._baseline_weight_sums = {}

        self.agents = []
        spawn_r = config.arena.agent_spawn_radius
        
        # PERSISTENCE: Load template struct once to share across agents
        template_struct = None
        if os.path.exists("base_graph.pt"):
            print("[Simulation] Loading shared synaptic template...")
            template_struct = torch.load("base_graph.pt", map_location=device, weights_only=True)
        
        for i in range(config.simulation.num_agents):
            brain = LRSGraph(config.lrsg, device, template_struct=template_struct)
            
            # Save the first built graph as a template if it didn't exist
            if i == 0 and template_struct is None:
                print("[Simulation] Saving new synaptic template to base_graph.pt...")
                brain.save_structure("base_graph.pt")
                # Reload to ensure template_struct is available for subsequent agents
                template_struct = torch.load("base_graph.pt", map_location=device, weights_only=True)
                
            pos = self.env.random_position_in_safe_zone(radius=spawn_r)
            agent = Agent(i, pos, brain, config.agent, config.lrsg, sanitizer=self._sanitizer)
            agent.init_episodic_compressor(config.episodic)
            self.agents.append(agent)
            self._agent_births[i] = 0
            self._agent_deaths[i] = 0
            self._init_edge_counts[i] = brain.num_edges
            self._init_weight_sums[i] = brain.out_weights.abs().sum().item()
            self._baseline_weight_sums[i] = brain.out_weights.abs().sum().item()

        self._last_broadcast = None
        self._total_actions = 0
        self._total_ram_consumed = 0.0
        self._births_total = 0
        self._deaths_total = 0
        self._prev_graph_complexity = 0.0
        self.metrics = {
            "survival": [],
            "ram_efficiency": [],
            "territory_control": [],
            "action_entropy": [],
            "graph_complexity": [],
            "rewire_rate": [],
            "weight_drift": [],
        }

        # Initialize Physical Safety Monitor (NEW)
        self.interlock_controller = InterlockController() # NEW INSTANCE
        self._telemetry = TelemetryLogger()
        self._dashboard = Dashboard()
        self._checkpointer = Checkpointer()
        self._visualizer = SimulationVisualizer("frontend")
        self._conflict_count = 0
        self._detailed_metrics_log = "detailed_metrics.jsonl"
        self._episodic_metrics = {
            "summary_cosine_similarities": [],
            "behavioral_changes": [],
            "tick": [],
        }
        self._prev_action_by_agent: dict[int, int] = {}

        # Initialize semantic encoder and sensor simulator if enabled
        if self.config.simulation.use_semantic:
            self.semantic_encoder = SemanticEncoder(device=self.config.simulation.semantic_encoder_device)
            self.sensor_simulator = SensorSimulator(device=self.config.simulation.semantic_encoder_device)
        else:
            self.semantic_encoder = None
            self.sensor_simulator = None

        # Stream pool for parallel agent execution
        self._streams = []
        if torch.cuda.is_available():
            self._streams = [torch.cuda.Stream() for _ in range(config.simulation.num_agents)]

        touch_file("telemetry.jsonl")
        touch_file("conflicts.jsonl")
        touch_file("episodic_results.jsonl")
        touch_file(self._detailed_metrics_log)

        if not config.simulation.headless:
            self._dashboard.start()
            self._visualizer.start()
            print(f"  Dashboard: http://localhost:8080")


    def tick(self):
        self.tick_count += 1

        t0 = time.perf_counter()
        self._physics_tick()
        t1 = time.perf_counter()
        phys_ms = (t1 - t0) * 1000
        self._telemetry.write_perf(self.tick_count, "physics", phys_ms)
        self._watchdog.check(phys_ms, self.tick_count, "physics")

        if self.tick_count % 10 == 0:
            t2 = time.perf_counter()
            self._linguistic_tick()
            t3 = time.perf_counter()
            ling_ms = (t3 - t2) * 1000
            self._telemetry.write_perf(self.tick_count, "linguistic", ling_ms)
            self._watchdog.check(ling_ms, self.tick_count, "linguistic")

        self._record_metrics()

        if self.tick_count % self._checkpointer.interval == 0:
            state = self._checkpointer.build_state(self)
            ckpt_path = self._checkpointer.save(self.tick_count, state)
            self._telemetry.write_checkpoint(self.tick_count, ckpt_path)

        if self.tick_count % 10 == 0:
            try:
                self._resource_monitor.check(self.tick_count)
            except ResourceExceededException as e:
                state = self._checkpointer.build_state(self)
                self._checkpointer.save(self.tick_count, state)
                raise
            vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            ram = psutil.virtual_memory().used / 1e9
            self._telemetry.write_resource_usage(self.tick_count, vram, ram)

        if self.tick_count % 100 == 0:
            self._render_ascii_map()

        if self.tick_count % 10 == 0:
            self._record_agent_brain_telemetry()
            self._record_episodic_metrics()
            self._write_episodic_results()
            self._record_detailed_metrics()

        self._push_visualizer_state()

    def _push_visualizer_state(self):
        alive = [a for a in self.agents if a.alive]
        state = {
            "tick": self.tick_count,
            "agents": [
                {
                    "id": a.id,
                    "x": a.position[0],
                    "y": a.position[1],
                    "z": a.position[2],
                    "ram": round(a.ram, 1),
                    "theta": round(a.theta, 4),
                    "lr": round(a.learning_rate, 4),
                    "alive": a.alive,
                    "action": a.last_action,
                }
                for a in self.agents
            ],
            "safe_zone": {
                "radius": self.env.safe_zone.radius,
                "center_x": self.env.safe_zone.center[0],
                "center_y": self.env.safe_zone.center[1],
                "center_z": self.env.safe_zone.center[2],
            },
            "ram_nodes": [
                {"x": n.position[0], "y": n.position[1], "z": n.position[2]}
                for n in self.env.active_ram_nodes
            ],
            "metrics": {k: (v[-1] if v else 0) for k, v in self.metrics.items()},
            "announcer": (
                self._last_broadcast.get("summary", {}).get("status", "")
                if self._last_broadcast
                else ""
            ),
        }
        self._visualizer.push_state(state)

        if self.tick_count % 10 == 0:
            for agent in self.agents:
                if agent.alive:
                    brain_data = self._compute_brain_aggregates(agent)
                    self._visualizer.push_brain_state(agent.id, brain_data)
                    graph_data = self._sample_brain_graph(agent)
                    self._visualizer.push_brain_graph_state(agent.id, graph_data)

        # Push semantic state and reward trace (every tick for responsiveness)
        for agent in self.agents:
            if agent.alive and agent.semantic_state is not None:
                semantic_vec = agent.semantic_state.cpu().tolist()
                self._visualizer.push_semantic_state(agent.id, self.tick_count, semantic_vec)
                self._visualizer.push_reward(agent.id, self.tick_count, agent._last_reward)

    def _sample_brain_graph(self, agent):
        brain = agent.brain
        N = brain.N
        num_input = self.config.lrsg.num_input_neurons
        num_output = self.config.lrsg.num_output_neurons
        h_start = num_input
        h_end = N - num_output

        n_input = min(30, num_input)
        n_hidden = 140
        n_output = min(30, num_output)

        in_idxs = torch.randperm(num_input, device=brain.device)[:n_input]
        h_size = max(1, h_end - h_start)
        h_idxs = torch.randperm(h_size, device=brain.device)[:min(n_hidden, h_size)] + h_start
        out_idxs = torch.randperm(num_output, device=brain.device)[:n_output] + (N - num_output)

        core_idxs = torch.sort(torch.cat([in_idxs, h_idxs, out_idxs])).values
        core_set = set(core_idxs.tolist())

        # Collect synapses from core neurons (up to 4 top-weighted per core)
        # and gather unique target ids as peripheral nodes
        peripheral_set = set()
        for idx in core_idxs:
            i = idx.item()
            targets = brain.out_targets[i]
            weights = brain.out_weights[i]
            deg = (targets >= 0).sum().item()
            if deg == 0:
                continue
            n_syn = min(deg, 4)
            if n_syn >= deg:
                top_t = targets[:deg]
            else:
                _, tidx = torch.topk(weights[:deg].float(), n_syn)
                top_t = targets[:deg][tidx]
            for j in range(top_t.shape[0]):
                tgt = top_t[j].item()
                if tgt >= 0 and tgt not in core_set:
                    peripheral_set.add(tgt)

        # Build full node set (core + peripheral, capped at ~400)
        MAX_NODES = 400
        extra = list(peripheral_set - core_set)
        if len(core_set) + len(extra) > MAX_NODES:
            extra = extra[:MAX_NODES - len(core_set)]
        all_set = core_set | set(extra)

        synapses = []
        for idx in core_idxs:
            i = idx.item()
            targets = brain.out_targets[i]
            weights = brain.out_weights[i]
            deg = (targets >= 0).sum().item()
            if deg == 0:
                continue
            n_syn = min(deg, 4)
            if n_syn >= deg:
                top_t = targets[:deg]
                top_w = weights[:deg]
            else:
                top_w, tidx = torch.topk(weights[:deg].float(), n_syn)
                top_t = targets[:deg][tidx]
            for j in range(top_t.shape[0]):
                tgt = top_t[j].item()
                if tgt >= 0 and tgt in all_set:
                    synapses.append({
                        "src": i,
                        "dst": tgt,
                        "weight": round(top_w[j].item(), 6),
                    })

        def region_of(i):
            if i < num_input:
                return "input"
            if i >= N - num_output:
                return "output"
            return "hidden"

        neurons = []
        for i in sorted(all_set):
            neurons.append({
                "id": i,
                "x": round(brain.positions[i, 0].item(), 2),
                "y": round(brain.positions[i, 1].item(), 2),
                "z": round(brain.positions[i, 2].item(), 2),
                "activation": round(brain.activations[i].item(), 6),
                "block_id": brain.block_ids[i].item(),
                "region": region_of(i),
            })

        return {
            "agent_id": agent.id,
            "tick": self.tick_count,
            "neurons": neurons,
            "synapses": synapses,
            "num_neurons": len(neurons),
            "num_edges": len(synapses),
            "births": self._agent_births.get(agent.id, 0),
            "deaths": self._agent_deaths.get(agent.id, 0),
        }

    def _compute_brain_aggregates(self, agent):
        brain = agent.brain
        N = brain.N
        block_size = 100
        num_blocks = max(1, N // block_size)
        truncated = num_blocks * block_size

        act = brain.activations[:truncated].reshape(num_blocks, block_size)
        pos = brain.positions[:truncated].reshape(num_blocks, block_size, 3)

        block_act_mean = act.mean(dim=1)
        block_act_max = act.max(dim=1).values
        block_pos = pos.mean(dim=1)

        block_activations = block_act_mean.cpu().tolist()
        block_max_act = block_act_max.cpu().tolist()
        block_positions = block_pos.cpu().tolist()

        top_k = min(200, num_blocks)
        _, top_indices = torch.topk(block_act_mean, top_k)

        connections = []
        visited_pairs = set()
        for b_idx in top_indices:
            b = b_idx.item()
            start = b * block_size
            end = min(start + block_size, N)
            targets = brain.out_targets[start:end]
            weights = brain.out_weights[start:end]
            valid = targets >= 0
            tgt_blocks = targets[valid] // block_size
            w = weights[valid].float()
            if tgt_blocks.numel() == 0:
                continue
            unique_tgts, inv = torch.unique(tgt_blocks, return_inverse=True)
            w_sum = torch.zeros(len(unique_tgts), device=brain.device, dtype=torch.float32)
            w_sum.scatter_add_(0, inv, w)
            for idx in range(len(unique_tgts)):
                tgt = unique_tgts[idx].item()
                if tgt >= num_blocks:
                    continue
                pair = (b, tgt) if b < tgt else (tgt, b)
                if pair in visited_pairs:
                    continue
                visited_pairs.add(pair)
                connections.append([b, tgt, round(w_sum[idx].item(), 4)])

        valid_mask = brain.out_targets >= 0
        w_mean = brain.out_weights[valid_mask].float().mean().item() if valid_mask.any() else 0.0

        return {
            "agent_id": agent.id,
            "tick": self.tick_count,
            "num_neurons": brain.N,
            "num_blocks": num_blocks,
            "num_edges": brain.num_edges,
            "avg_weight": round(w_mean, 6),
            "mean_activation": round(brain.activations.mean().item(), 6),
            "active_pct": round((brain.activations > self.config.lrsg.threshold_theta).sum().item() / brain.N * 100, 2),
            "births": self._agent_births.get(agent.id, 0),
            "deaths": self._agent_deaths.get(agent.id, 0),
            "volatility": round(brain.activations.var().item(), 6),
            "block_positions": block_positions,
            "block_activations": block_activations,
            "block_max_act": block_max_act,
            "connections": connections,
        }

    def _record_agent_brain_telemetry(self):
        theta = self.config.lrsg.threshold_theta
        for agent in self.agents:
            if not agent.alive:
                continue
            brain = agent.brain
            num_edges = brain.num_edges
            num_neurons = brain.N
            avg_weight = brain.out_weights[brain.out_targets >= 0].mean().item() if (brain.out_targets >= 0).any() else 0.0
            active_neurons = (brain.activations > theta).sum().item()
            births = self._agent_births.get(agent.id, 0)
            deaths = self._agent_deaths.get(agent.id, 0)
            self._telemetry.write_agent_brain(
                self.tick_count, agent.id, num_edges, num_neurons,
                avg_weight, active_neurons, births, deaths,
            )

    def _physics_tick(self):
        alive_agents = [a for a in self.agents if a.alive]
        if not alive_agents:
            return

        self.env.tick(self.agents)

        # Announcer is now passive observer for logging/dashboard
        if self.tick_count % 10 == 0:
            # OPTIMIZATION: Removed redundant broadcast call here. 
            # It will be called in _linguistic_tick() every 10 ticks.
            pass

        # Update whisper buffers for all agents BEFORE they act
        for agent in alive_agents:
            agent.whisper_buffer.zero_()
            if agent._received_whispers:
                # Sort by distance
                sorted_whispers = sorted(agent._received_whispers, key=lambda x: x["distance"])
                # Take top 4
                top_4 = sorted_whispers[:4]
                for i, w in enumerate(top_4):
                    start_idx = i * 12
                    # 3 tokens (normalized)
                    for j, idx in enumerate(w["indices"][:3]):
                        agent.whisper_buffer[start_idx + j] = idx / 512.0
                    # 1 distance (normalized)
                    agent.whisper_buffer[start_idx + 3] = min(1.0, w["distance"] / 20.0)
                    # 1 SVO-valid flag
                    agent.whisper_buffer[start_idx + 4] = 1.0 if w["is_svo"] else 0.0
                    # 7 neurons padding (already zeroed)
                
                # Clear for next tick
                agent._received_whispers = []

        # OPTIMIZATION: Parallelize agent execution using CUDA streams
        if torch.cuda.is_available() and self._streams:
            for i, agent in enumerate(alive_agents):
                # Use modulo to handle more agents than streams if necessary,
                # but pool is sized for num_agents.
                s = self._streams[i % len(self._streams)]
                with torch.cuda.stream(s):
                    agent.sense_and_act(self.env, alive_agents)
                    agent.episodic_push(self.tick_count)
            # Synchronize all streams
            torch.cuda.synchronize()
        else:
            for agent in alive_agents:
                agent.sense_and_act(self.env, alive_agents)
                agent.episodic_push(self.tick_count)

        # Apply semantic encoding if enabled
        if self.semantic_encoder is not None:
            for agent in alive_agents:
                # Build sensor dict from agent state
                sensor_dict = agent.build_sensor_dict(self.env, alive_agents, tick=self.tick_count)

                # Apply sensor noise if enabled
                if self.sensor_simulator:
                    sensor_dict = self.sensor_simulator.apply_noise(sensor_dict)

                # Encode to 57D semantic state
                semantic_state = self.semantic_encoder.forward(sensor_dict)

                # Update context buffer: replace last 57 values with semantic state
                if agent.context_buffer.shape[0] >= 57:
                    agent.context_buffer[-57:] = semantic_state.to(agent.context_buffer.device)

        for agent in alive_agents:
            if agent._episodic_compressor is not None:
                loss = agent.episodic_train(self.tick_count, interval=10)
                if self.tick_count % 50 == 0:
                    pass

            if agent.last_action == 4 and agent.whisper_text:
                summary = agent.get_episodic_summary()
                self._broadcast_episodic_summary(agent, alive_agents)
                self._broadcast_whisper(agent, alive_agents)

            probe_result = agent.execute_action(self.env, alive_agents)
            self._total_ram_consumed += agent.ACTION_COSTS.get(agent.last_action, 0)

            if agent.last_action == 3 and self.tick_count <= 200:
                agent.ram += 5.0

            action_name = ACTION_NAMES[agent.last_action] if 0 <= agent.last_action < len(ACTION_NAMES) else "UNKNOWN"
            self._telemetry.write_action(self.tick_count, agent.id, agent.last_action, action_name, agent.ram)

            self._total_actions += 1

        for agent in alive_agents:
            if agent.last_action == 5 and agent.last_probe_result and "BLOCKED" in agent.last_probe_result:
                raw_payload = agent.whisper_text if agent.whisper_text else ""
                payload = self._sanitizer.sanitize_agent_whisper(raw_payload, agent.id, self.tick_count)
                payload = payload if payload else ""
                for other in alive_agents:
                    if other.id != agent.id:
                        other.record_blocked_payload(agent.id, payload, self.tick_count)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        collisions = self.env.resolve_collisions(alive_agents)
        for collision_group in collisions:
            group_agents = [alive_agents[i] for i in collision_group]
            for i in range(len(group_agents)):
                for j in range(i + 1, len(group_agents)):
                    a = group_agents[i]
                    b = group_agents[j]
                    outcome = ACTION_NAMES[a.last_action] + "+" + ACTION_NAMES[b.last_action]
                    resolve_conflict(a, b, self.env, alive_agents, distance=0.0, tick=self.tick_count)
                    self._conflict_count += 1
                    self._telemetry.write_conflict(
                        self.tick_count, a.id, b.id, outcome
                    )

        for agent in alive_agents:
            if agent.alive:
                took_damage = agent.ram < agent.prev_ram
                delta_ram = agent.ram - agent.prev_ram
                probe_result = agent.last_probe_result if agent.last_action == 5 else None
                agent.compute_intrinsic_reward(self.env, delta_ram, took_damage, probe_result, self.agents)
                agent._last_reward = agent.intrinsic_reward
                agent.update_metabolic_params()
                agent.prev_ram = agent.ram

        # OPTIMIZATION: Parallelize plasticity updates using CUDA streams
        if torch.cuda.is_available() and self._streams:
            for i, agent in enumerate(alive_agents):
                s = self._streams[i % len(self._streams)]
                with torch.cuda.stream(s):
                    hebbian_update(
                        agent.brain, self.config.lrsg,
                        agent.intrinsic_reward,
                        learning_rate=agent.learning_rate,
                        theta=agent.theta,
                    )

                    if self.tick_count % 20 == 0:
                        b, d = structural_plasticity(
                            agent.brain, self.config.lrsg,
                            tick=self.tick_count,
                            theta=agent.theta,
                        )
                        from src.lrsg.plasticity import metabolic_maintenance
                        metabolic_maintenance(agent.brain, self.config.lrsg)
                        # Store results in agent for later collection to avoid race conditions
                        agent._last_births = b
                        agent._last_deaths = d
            torch.cuda.synchronize()
            # Collect telemetry after synchronization
            if self.tick_count % 20 == 0:
                for agent in alive_agents:
                    b, d = getattr(agent, "_last_births", 0), getattr(agent, "_last_deaths", 0)
                    self._births_total += b
                    self._deaths_total += d
                    self._agent_births[agent.id] = self._agent_births.get(agent.id, 0) + b
                    self._agent_deaths[agent.id] = self._agent_deaths.get(agent.id, 0) + d
                    self._baseline_weight_sums[agent.id] = agent.brain.out_weights.abs().sum().item()
        else:
            for agent in alive_agents:
                if agent.alive:
                    hebbian_update(
                        agent.brain, self.config.lrsg,
                        agent.intrinsic_reward,
                        learning_rate=agent.learning_rate,
                        theta=agent.theta,
                    )

                    if self.tick_count % 20 == 0:
                        births, deaths = structural_plasticity(
                            agent.brain, self.config.lrsg,
                            tick=self.tick_count,
                            theta=agent.theta,
                        )
                        from src.lrsg.plasticity import metabolic_maintenance
                        metabolic_maintenance(agent.brain, self.config.lrsg)
                        self._births_total += births
                        self._deaths_total += deaths
                        self._agent_births[agent.id] = self._agent_births.get(agent.id, 0) + births
                        self._agent_deaths[agent.id] = self._agent_deaths.get(agent.id, 0) + deaths
                        self._baseline_weight_sums[agent.id] = agent.brain.out_weights.abs().sum().item()
                if self.tick_count % 10 == 0:
                    agent.brain._edge_history.append(agent.brain.num_edges)

    def _linguistic_tick(self):
        arena_summary = self.env.get_state_summary()
        alive_agents = [a for a in self.agents if a.alive]
        arena_summary["agents_alive"] = len(alive_agents)

        agent_whispers = []
        for agent in alive_agents:
            if agent.last_action == 4:
                sanitized = self._sanitizer.sanitize_agent_whisper(
                    agent.whisper_text, agent.id, self.tick_count
                )
                agent_whispers.append(sanitized if sanitized else "")
            else:
                agent_whispers.append("")

        if self.config.simulation.use_announcer:
            raw = self.announcer.broadcast(arena_summary, agent_whispers=agent_whispers)
        else:
            raw = self._generate_broadcast_from_char_decoder(alive_agents, agent_whispers, arena_summary)

        self._last_broadcast = self._sanitizer.sanitize_broadcast(raw, self.tick_count)
        self._telemetry.write_announcer(self.tick_count, raw)

        # LLM announcer evaluates agent whispers and assigns rewards/punishments
        if alive_agents:
            agent_ids = [a.id for a in alive_agents]
            raw_rewards = self.announcer.evaluate_whispers(
                agent_whispers, agent_ids=agent_ids, tick=self.tick_count
            )
            for agent in alive_agents:
                agent.announcer_reward = raw_rewards.get(agent.id, 0.0)

        torch.cuda.empty_cache()

    def _sample_char_sequence(self, brain, length=32, temperature=0.8):
        PRINTABLE = list(range(32, 127))
        pwm_state = brain.activations[brain.idx_pwm].half()
        logits = torch.matmul(brain.W_char_pred, pwm_state)
        chars = []
        for _ in range(length):
            scaled = logits[PRINTABLE].float() / temperature
            probs = torch.softmax(scaled, dim=0)
            char_idx = torch.multinomial(probs, 1).item()
            chars.append(chr(PRINTABLE[char_idx]))
        return ''.join(chars)

    def _generate_broadcast_from_char_decoder(self, alive_agents, agent_whispers, arena_summary):
        status = "The arena hums with activity."
        if alive_agents:
            seq = self._sample_char_sequence(alive_agents[0].brain, length=32, temperature=0.8)
            if seq:
                status = seq[:120]
        return {
            "tick": self.tick_count,
            "global_embedding": torch.zeros(128),
            "agent_embeddings": [torch.zeros(128) for _ in range(max(1, len(alive_agents)))],
            "whispers": [w if w else "..." for w in agent_whispers],
            "agent_reports": agent_whispers,
            "summary": {
                "agents_alive": len(alive_agents),
                "safe_zone_radius": arena_summary.get("safe_zone_radius", 0),
                "ram_nodes": arena_summary.get("active_ram_nodes", 0),
                "status": status,
            },
        }

    def _compute_rewards(self, alive_agents: list) -> dict:
        rewards = {}
        for agent in alive_agents:
            if not agent.alive:
                continue
            survival_r = 0.1
            ram_r = agent.ram / 100.0
            in_safe = 0.2 if self.env.is_in_safe_zone(agent.position) else -0.1
            rewards[agent.id] = survival_r + ram_r + in_safe
        return rewards

    def _record_metrics(self):
        alive = [a for a in self.agents if a.alive]
        n_alive = len(alive)
        self.metrics["survival"].append(n_alive)

        if self._total_actions > 0:
            ram_eff = self._total_ram_consumed / self._total_actions
            ram_eff = min(ram_eff, 1.0)
        else:
            ram_eff = 0
        self.metrics["ram_efficiency"].append(ram_eff)

        occupied = 0
        for a in alive:
            if self.env.is_in_safe_zone(a.position):
                occupied += 1
        territory_pct = occupied / max(1, n_alive)
        self.metrics["territory_control"].append(territory_pct)

        action_counts = {}
        for a in alive:
            act = a.last_action
            action_counts[act] = action_counts.get(act, 0) + 1
        total = sum(action_counts.values())
        if total > 0:
            entropy = -sum(
                (c / total) * math.log(c / total) for c in action_counts.values()
            )
            self.metrics["action_entropy"].append(entropy)
        else:
            self.metrics["action_entropy"].append(0)

        total_changes = self._births_total + self._deaths_total
        self.metrics["rewire_rate"].append(total_changes / max(1, self.tick_count))

        if n_alive > 0 and self.tick_count % 10 == 0:
            total_weight_drift = 0.0
            valid_count = 0
            for a in alive:
                baseline = self._baseline_weight_sums.get(a.id, 0)
                current_sum = a.brain.out_weights[a.brain.out_targets >= 0].abs().sum().item()
                if baseline > 0 and current_sum > 0 and not (current_sum != current_sum):
                    drift = abs(current_sum - baseline) / baseline
                    if drift == drift:
                        total_weight_drift += min(drift, 1000000.0)
                        valid_count += 1
            self.metrics["weight_drift"].append(total_weight_drift / max(1, valid_count) if valid_count > 0 else 0.0)

        if n_alive > 0 and self.tick_count % 10 == 0:
            # OPTIMIZATION: Use a much faster approximation for graph complexity
            # instead of O(N*D) check for all edges.
            total_active_neurons = 0
            threshold = self.config.lrsg.threshold_theta
            for a in alive:
                n_activated = (a.brain.activations > threshold).sum().item()
                total_active_neurons += n_activated
            
            # Use sparsity-adjusted density as proxy for complexity
            mean_complexity = (total_active_neurons / n_alive) / (self.config.lrsg.num_neurons / 1000.0)
            self.metrics["graph_complexity"].append(mean_complexity)

            # modularity is also slow, skipping for now
            self.metrics.setdefault("modularity", []).append(0.0)

            strat_div = self._compute_strategy_divergence(alive)
            self.metrics.setdefault("strategy_divergence", []).append(strat_div)

        current = {k: (v[-1] if v else 0) for k, v in self.metrics.items()}
        validated = self._metric_validator.validate(current, self.tick_count)
        self._telemetry.write_metrics(self.tick_count, validated)

    def _compute_modularity(self, alive: list) -> float:
        return 0.0

    def _compute_strategy_divergence(self, alive: list) -> float:
        if len(alive) < 2:
            return 0.0
        top = sorted(alive, key=lambda a: a.ram, reverse=True)[:3]
        if len(top) < 2:
            return 0.0
        histograms = []
        for a in top:
            hist = [0, 0, 0, 0]
            for act in a.action_history:
                if 0 <= act < 4:
                    hist[act] += 1
            total = sum(hist)
            if total == 0:
                histograms.append([0.25, 0.25, 0.25, 0.25])
            else:
                histograms.append([h / total for h in hist])
        divergences = []
        for i in range(len(histograms)):
            for j in range(i + 1, len(histograms)):
                P = histograms[i]
                Q = histograms[j]
                M = [(p + q) / 2.0 for p, q in zip(P, Q)]
                kl_pm = sum(p * math.log(p / m) if p > 0 else 0.0 for p, m in zip(P, M))
                kl_qm = sum(q * math.log(q / m) if q > 0 else 0.0 for q, m in zip(Q, M))
                js = 0.5 * kl_pm + 0.5 * kl_qm
                divergences.append(js)
        return sum(divergences) / max(1, len(divergences))

    def _render_ascii_map(self):
        alive = [a for a in self.agents if a.alive]
        if not alive:
            return
        W, H = 40, 40
        gs = self.env.grid_size
        scale_x = gs[0] / W
        scale_y = gs[1] / H
        cx, cy = self.env.safe_zone.center[0], self.env.safe_zone.center[1]
        r = self.env.safe_zone.radius
        grid = [[" "] * W for _ in range(H)]

        for row in range(H):
            for col in range(W):
                wx = col * scale_x + scale_x / 2
                wy = row * scale_y + scale_y / 2
                dx = wx - cx
                dy = wy - cy
                dist = math.sqrt(dx * dx + dy * dy)
                if dist <= r:
                    if abs(dist - r) < scale_x * 0.8:
                        grid[row][col] = "*"
                    else:
                        grid[row][col] = "."

        for node in self.env.active_ram_nodes:
            col = int(node.position[0] / scale_x)
            row = int(node.position[1] / scale_y)
            if 0 <= row < H and 0 <= col < W:
                grid[row][col] = "$"

        for agent in alive:
            col = int(agent.position[0] / scale_x)
            row = int(agent.position[1] / scale_y)
            if 0 <= row < H and 0 <= col < W:
                grid[row][col] = str(agent.id % 10)

        lines = [f"  Tick={self.tick_count}  Radius={r:.0f}  Alive={len(alive)}  RAM nodes={len(self.env.active_ram_nodes)}"]
        lines.append("  " + "-" * W)
        for row in grid:
            lines.append("  |" + "".join(row) + "|")
        lines.append("  " + "-" * W)
        print("\n".join(lines))

    def _broadcast_episodic_summary(self, sender, alive_agents):
        summary = sender.get_episodic_summary()
        for other in alive_agents:
            if other.id == sender.id or not other.alive:
                continue
            d = math.sqrt(sum((sender.position[i] - other.position[i]) ** 2 for i in range(3)))
            if d < 25:
                other.receive_summary(sender.id, summary, self.tick_count)

    def _broadcast_whisper(self, sender, alive_agents):
        safe_text = self._sanitizer.sanitize_agent_whisper(
            sender.whisper_text, sender.id, self.tick_count
        )
        if not safe_text:
            return
        indices = encode_whisper_to_indices(safe_text)
        is_svo = validate_curriculum(safe_text, self.tick_count) > 0
        
        for other in alive_agents:
            if other.id == sender.id or not other.alive:
                continue
            d = math.sqrt(sum((sender.position[i] - other.position[i]) ** 2 for i in range(3)))
            if d < 20: # Whisper range
                other._received_whispers.append({
                    "indices": indices,
                    "distance": d,
                    "is_svo": is_svo
                })

    def _record_episodic_metrics(self):
        alive = [a for a in self.agents if a.alive]
        if len(alive) < 2:
            return

        summaries = {}
        for a in alive:
            if a._episodic_summary is not None:
                summaries[a.id] = a.get_episodic_summary()

        if len(summaries) < 2:
            return

        ids = list(summaries.keys())
        total_sim = 0.0
        pairs = 0
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                s1 = summaries[ids[i]].flatten()[:10]
                s2 = summaries[ids[j]].flatten()[:10]
                # Avoid similarity check on all-zero vectors
                if torch.norm(s1) < 1e-6 or torch.norm(s2) < 1e-6:
                    continue
                sim_val = float(F.cosine_similarity(s1.unsqueeze(0), s2.unsqueeze(0)).mean().item())
                total_sim += sim_val
                pairs += 1

        avg_sim = total_sim / max(1, pairs)
        self._episodic_metrics["summary_cosine_similarities"].append(avg_sim)
        self._episodic_metrics["tick"].append(self.tick_count)

        behavioral_changes = 0
        total_pairs = 0
        for a in alive:
            if a._received_summaries:
                for rec in a._received_summaries:
                    prev = rec["my_action_before"]
                    curr = a.last_action
                    if prev != -1 and prev != curr:
                        behavioral_changes += 1
                    total_pairs += 1
        self._episodic_metrics["behavioral_changes"].append(
            behavioral_changes / max(1, total_pairs)
        )

    def _write_episodic_results(self):
        with open("episodic_results.jsonl", "a", encoding="utf-8") as f:
            record = {
                "tick": self.tick_count,
                "avg_summary_similarity": self._episodic_metrics["summary_cosine_similarities"][-1] if self._episodic_metrics["summary_cosine_similarities"] else 0.0,
                "behavioral_change_rate": self._episodic_metrics["behavioral_changes"][-1] if self._episodic_metrics["behavioral_changes"] else 0.0,
            }
            f.write(json.dumps(record) + "\n")

    def _record_detailed_metrics(self):
        alive = [a for a in self.agents if a.alive]
        if not alive:
            return

        action_counts = [0] * 7
        for a in alive:
            if 0 <= a.last_action < 7:
                action_counts[a.last_action] += 1
        total = sum(action_counts)
        action_pct = [c / max(1, total) for c in action_counts]

        positions = {a.id: a.position for a in alive}
        n = len(alive)
        proximity_matrix = [[0.0] * n for _ in range(n)]
        ids = list(positions.keys())
        total_dist = 0.0
        pair_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                d = math.sqrt(sum((positions[ids[i]][k] - positions[ids[j]][k]) ** 2 for k in range(3)))
                proximity_matrix[i][j] = d
                proximity_matrix[j][i] = d
                total_dist += d
                pair_count += 1
        avg_distance = total_dist / max(1, pair_count)

        all_tokens = set()
        total_tokens = 0
        for a in alive:
            all_tokens.update(a._seen_tokens)
            total_tokens += len(a._seen_tokens)
        token_diversity = len(all_tokens) / max(1, total_tokens) if total_tokens > 0 else 0.0

        summaries = {}
        for a in alive:
            if a._episodic_summary is not None:
                summaries[a.id] = a.get_episodic_summary()
        ae_similarity = 0.0
        ae_pairs = 0
        ids_s = list(summaries.keys())
        for i in range(len(ids_s)):
            for j in range(i + 1, len(ids_s)):
                s1 = summaries[ids_s[i]].flatten()[:10]
                s2 = summaries[ids_s[j]].flatten()[:10]
                ae_similarity += float(F.cosine_similarity(s1.unsqueeze(0), s2.unsqueeze(0)).item())
                ae_pairs += 1
        ae_similarity = ae_similarity / max(1, ae_pairs)

        bcr = self._episodic_metrics["behavioral_changes"][-1] if self._episodic_metrics["behavioral_changes"] else 0.0

        ae_losses = [a._episodic_train_loss for a in alive if hasattr(a, "_episodic_train_loss")]
        avg_ae_loss = sum(ae_losses) / len(ae_losses) if ae_losses else 0.0

        record = {
            "tick": self.tick_count,
            "num_agents": len(alive),
            "action_pct": action_pct,
            "avg_distance": round(avg_distance, 4),
            "proximity_matrix": proximity_matrix,
            "agent_ids": ids,
            "token_diversity": round(token_diversity, 6),
            "unique_tokens": len(all_tokens),
            "total_tokens": total_tokens,
            "ae_similarity": round(ae_similarity, 6),
            "behavioral_change_rate": round(bcr, 6),
            "avg_ae_loss": round(avg_ae_loss, 6),
            "ram_efficiency": round(self.metrics["ram_efficiency"][-1], 4) if self.metrics["ram_efficiency"] else 0,
            "territory_control": round(self.metrics["territory_control"][-1], 4) if self.metrics["territory_control"] else 0,
            "action_entropy": round(self.metrics["action_entropy"][-1], 4) if self.metrics["action_entropy"] else 0,
            "graph_complexity": round(self.metrics["graph_complexity"][-1], 4) if self.metrics["graph_complexity"] else 0,
            "rewire_rate": round(self.metrics["rewire_rate"][-1], 6) if self.metrics["rewire_rate"] else 0,
            "modularity": round(self.metrics.get("modularity", [0])[-1], 4),
            "strategy_divergence": round(self.metrics.get("strategy_divergence", [0])[-1], 4),
        }
        with open(self._detailed_metrics_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def stop(self):
        """Stop background threads and release resources."""
        if hasattr(self, "_dashboard"):
            self._dashboard.stop()
        if hasattr(self, "_visualizer"):
            self._visualizer.push_state({"tick": -1})
            self._visualizer.stop()

    def run(self, max_ticks: int = None):
        if max_ticks is None:
            max_ticks = self.config.simulation.max_ticks
        try:
            for _ in range(max_ticks):
                self.tick()
                if not any(a.alive for a in self.agents):
                    break
        except ResourceExceededException:
            state = self._checkpointer.build_state(self)
            self._checkpointer.save(self.tick_count, state)
        finally:
            self.stop()
        return self.metrics

    @property
    def status(self) -> dict:
        alive = [a for a in self.agents if a.alive]
        return {
            "tick": self.tick_count,
            "agents_alive": len(alive),
            "safe_zone_radius": self.env.safe_zone.radius,
            "ram_nodes": len(self.env.active_ram_nodes),
            "metrics": {k: v[-1] if v else 0 for k, v in self.metrics.items()},
        }
