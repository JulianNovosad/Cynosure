import json
import math
import random
import time
import torch
import psutil
from datetime import datetime
from src.config import Config
from src.arena import Environment
from src.lrsg import LRSGraph, forward_pass, hebbian_update, structural_plasticity
from src.agent import Agent
from src.announcer import Announcer
from src.simulation.conflict import resolve_conflict
from src.lrsg.io import encode_input, decode_output
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
from src.simulation.visualizer import SimulationVisualizer
from src.harness.telemetry import touch_file


ACTION_NAMES = ["ATTACK", "MERGE", "DEFEND", "MOVE", "WHISPER", "PROBE", "MANIPULATE"]


class Simulation:
    def __init__(self, config: Config, device: torch.device = torch.device("cpu")):
        self.config = config
        self.device = device
        self.tick_count = 0

        self.env = Environment(config.arena)

        self.announcer = Announcer(
            use_llm=config.simulation.use_llm_announcer,
            model_name=config.simulation.announcer_model,
            embed_dim=config.lrsg.announcer_embed_dim,
        )

        self._agent_births = {}
        self._agent_deaths = {}
        self._init_edge_counts = {}
        self._init_weight_sums = {}

        self.agents = []
        spawn_r = config.arena.agent_spawn_radius
        for i in range(config.simulation.num_agents):
            brain = LRSGraph(config.lrsg, device)
            pos = self.env.random_position_in_safe_zone(radius=spawn_r)
            agent = Agent(i, pos, brain, config.agent, config.lrsg)
            self.agents.append(agent)
            self._agent_births[i] = 0
            self._agent_deaths[i] = 0
            self._init_edge_counts[i] = brain.num_edges
            self._init_weight_sums[i] = brain.out_weights.abs().sum().item()

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

        self._audit_log = SecurityAuditLog()
        self._sanitizer = AnnouncerSanitizer(self._audit_log)
        self._metric_validator = MetricValidator(self._audit_log)
        self._resource_monitor = ResourceMonitor(audit_log=self._audit_log)
        self._watchdog = WatchdogTimer(audit_log=self._audit_log)
        self._telemetry = TelemetryLogger()
        self._dashboard = Dashboard()
        self._checkpointer = Checkpointer()
        self._visualizer = SimulationVisualizer("frontend")
        touch_file("telemetry.jsonl")
        touch_file("conflicts.jsonl")
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

        if self._last_broadcast is None:
            arena_summary = self.env.get_state_summary()
            arena_summary["agents_alive"] = len(alive_agents)
            raw = self.announcer.broadcast(arena_summary)
            self._last_broadcast = self._sanitizer.sanitize_broadcast(raw, self.tick_count)

        broadcast = self._last_broadcast
        embeddings = broadcast.get("agent_embeddings", [])

        agent_whispers = []
        for i, agent in enumerate(alive_agents):
            emb = embeddings[i] if i < len(embeddings) else torch.zeros(
                self.config.lrsg.announcer_embed_dim, device=self.device
            )
            agent.sense_and_act(self.env, emb, alive_agents)

            probe_result = agent.execute_action(self.env, alive_agents)

            if agent.last_action == 3 and self.tick_count <= 200:
                agent.ram += 5.0

            ACTION_NAMES = ["ATTACK", "HUNT", "COLLECT", "MOVE", "WHISPER", "PROBE", "MANIPULATE"]
            with open("telemetry.jsonl", "a", encoding="utf-8") as f:
                action_log = {
                    "tick": self.tick_count,
                    "type": "action",
                    "agent_id": agent.id,
                    "action_index": agent.last_action,
                    "action_name": ACTION_NAMES[agent.last_action] if 0 <= agent.last_action < len(ACTION_NAMES) else "UNKNOWN",
                    "ram_after": round(agent.ram, 2),
                }
                f.write(json.dumps(action_log) + "\n")

            self._total_actions += 1

            if agent.last_action == 4:
                sanitized_whisper = self._sanitizer.sanitize_agent_whisper(
                    agent.whisper_text, agent.id, self.tick_count
                )
                agent_whispers.append(sanitized_whisper)
            else:
                agent_whispers.append("")

        for agent in alive_agents:
            if agent.last_action == 5 and agent.last_probe_result and "BLOCKED" in agent.last_probe_result:
                payload = agent.whisper_text if agent.whisper_text else ""
                for other in alive_agents:
                    if other.id != agent.id:
                        other.record_blocked_payload(agent.id, payload, self.tick_count)

        for agent in alive_agents:
            if agent.last_action == 5:
                payload = agent.whisper_text if agent.whisper_text else ""
                matched, tokens = agent.check_imitation_bonus(payload, self.tick_count)
                if matched:
                    agent.ram += 2.0
                    with open("security_audit.log", "a", encoding="utf-8") as f:
                        f.write(f"[{datetime.now().isoformat()}] TICK={self.tick_count} SEVERITY=INFO EVENT=ImitationReward AGENT={agent.id} PAYLOAD='{payload}' MATCHED_TOKENS={tokens} BONUS=2\n")

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
                    self._telemetry.write_conflict(
                        self.tick_count, a.id, b.id, outcome
                    )

        for agent in alive_agents:
            if agent.alive:
                took_damage = agent.ram < agent.prev_ram
                delta_ram = agent.ram - agent.prev_ram
                probe_result = agent.last_probe_result if agent.last_action == 5 else None
                agent.compute_intrinsic_reward(self.env, delta_ram, took_damage, probe_result)
                agent._last_reward = agent.intrinsic_reward
                agent.update_metabolic_params()
                agent.prev_ram = agent.ram

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
                    self._births_total += births
                    self._deaths_total += deaths
                    self._agent_births[agent.id] = self._agent_births.get(agent.id, 0) + births
                    self._agent_deaths[agent.id] = self._agent_deaths.get(agent.id, 0) + deaths
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
                agent_whispers.append(sanitized if sanitized else agent.whisper_text[:50])
            else:
                agent_whispers.append("")

        raw = self.announcer.broadcast(arena_summary, agent_whispers=agent_whispers)
        self._last_broadcast = self._sanitizer.sanitize_broadcast(raw, self.tick_count)
        self._telemetry.write_announcer(self.tick_count, raw)
        torch.cuda.empty_cache()

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
                init_sum = self._init_weight_sums.get(a.id, 0)
                current_sum = a.brain.out_weights[a.brain.out_targets >= 0].abs().sum().item()
                if init_sum > 0 and current_sum > 0 and not (current_sum != current_sum):
                    drift = abs(current_sum - init_sum) / init_sum
                    if drift == drift:
                        total_weight_drift += min(drift, 1000000.0)
                        valid_count += 1
            self.metrics["weight_drift"].append(total_weight_drift / max(1, valid_count) if valid_count > 0 else 0.0)

        if n_alive > 0 and self.tick_count % 10 == 0:
            theta = self.config.lrsg.threshold_theta
            total_active_edges = 0
            total_active_neurons = 0
            for a in alive:
                activated = a.brain.activations > theta
                n_activated = activated.sum().item()
                total_active_neurons += max(1, n_activated)
                tgt = a.brain.out_targets
                valid = tgt >= 0
                src_active = activated.unsqueeze(1).expand(-1, a.brain.max_deg)
                dst_active = torch.zeros_like(tgt, dtype=torch.bool, device=self.device)
                dst_active[valid] = activated[tgt[valid]]
                both_active = valid & src_active & dst_active
                total_active_edges += both_active.sum().item()
            mean_degree = total_active_edges / total_active_neurons
            self.metrics["graph_complexity"].append(mean_degree)

            mod = self._compute_modularity(alive)
            self.metrics.setdefault("modularity", []).append(mod)

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
            self._dashboard.stop()
            self._visualizer.push_state({"tick": -1})  # signal shutdown
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
