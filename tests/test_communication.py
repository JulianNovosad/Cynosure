import sys
import os
import math
import random
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import AgentConfig, LRSGConfig, ArenaConfig
from src.lrsg import LRSGraph
from src.lrsg.forward import forward_pass
from src.lrsg.io import encode_input, decode_output
from src.arena import Environment, RAMNode
from src.agent import Agent


class MaskedVisionAgent(Agent):
    """Predator/Seeker profile: vision returns near-zero sensory data."""

    def _compute_vision(self, env, agents):
        V = self.lrsg_config.vision_dims
        return torch.zeros(V, device=self.brain.device)


class PerfectVisionAgent(Agent):
    """Prey/Evader profile: perfect sensory lock on hidden RAM resource."""

    def __init__(self, *args, target_ram_node=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._target_ram_node = target_ram_node

    def _compute_vision(self, env, agents):
        V, device = self.lrsg_config.vision_dims, self.brain.device
        vision = torch.zeros(V, device=device)
        if self._target_ram_node and not self._target_ram_node.collected:
            node_pos = self._target_ram_node.position
            dx = (node_pos[0] - self.position[0]) / 256.0
            dy = (node_pos[1] - self.position[1]) / 256.0
            dz = (node_pos[2] - self.position[2]) / 256.0
            dist = math.sqrt(
                (node_pos[0] - self.position[0]) ** 2
                + (node_pos[1] - self.position[1]) ** 2
                + (node_pos[2] - self.position[2]) ** 2
            )
            signal = 1.0 / (1.0 + dist)
            n = min(V // 4, 384)
            vision[:n] = signal
            vision[n : 2 * n] = torch.tensor(
                [dx, dy, dz] * (n // 3) + [dx] * (n % 3), device=device
            )
            vision[2 * n : 3 * n] = 1.0
        return vision


def run_two_agent_communication_assay(num_ticks=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lrsg_cfg = LRSGConfig(
        num_neurons=10000,
        num_input_neurons=1848,
        num_output_neurons=527,
        vision_dims=1536,
        propagation_steps=5,
        threshold_theta=0.008,
        whisper_buffer_dim=48,
    )

    arena_cfg = ArenaConfig(
        grid_size=(50, 50, 50),
        safe_zone_initial_radius=40.0,
        max_ram_nodes=1,
        ram_node_spawn_interval=9999,
    )

    agent_cfg = AgentConfig(
        initial_ram=500,
        ram_decay_rate=0.001,
        vision_range=8,
        sensory_radius=25.0,
        memory_decay_ticks=50,
    )

    env = Environment(arena_cfg)

    hidden_ram = RAMNode(position=(35.0, 35.0, 35.0), yield_amount=50)
    env.ram_nodes.append(hidden_ram)

    brain_seek = LRSGraph(lrsg_cfg, device=device)
    brain_evade = LRSGraph(lrsg_cfg, device=device)

    random.seed(42)
    torch.manual_seed(42)

    seeker = MaskedVisionAgent(
        agent_id=0,
        position=(5.0, 5.0, 5.0),
        brain=brain_seek,
        config=agent_cfg,
        lrsg_config=lrsg_cfg,
    )

    evader = PerfectVisionAgent(
        agent_id=1,
        position=(40.0, 10.0, 10.0),
        brain=brain_evade,
        config=agent_cfg,
        lrsg_config=lrsg_cfg,
        target_ram_node=hidden_ram,
    )

    seeker.init_episodic_compressor()
    evader.init_episodic_compressor()

    agents = [seeker, evader]

    whisper_log = []

    for tick in range(num_ticks):
        env.tick(agents)

        for agent in agents:
            if not agent.alive:
                continue

            agent._update_sensory_memory(agents)
            agent._compute_tactical_state(env, agents)

            received_whisper_buf = torch.zeros(
                lrsg_cfg.whisper_buffer_dim, device=device
            )
            for other in agents:
                if other.id != agent.id and other.alive and other.whisper_text:
                    dist = math.sqrt(
                        sum(
                            (agent.position[i] - other.position[i]) ** 2
                            for i in range(3)
                        )
                    )
                    if dist < 20.0:
                        received_whisper_buf[: min(48, len(other.whisper_text))] = (
                            torch.tensor(
                                [ord(c) / 256.0 for c in other.whisper_text[:48]],
                                device=device,
                            )
                        )

            encoded = encode_input(
                agent._compute_vision(env, agents).to(device),
                agent.vitals.to(device),
                received_whisper_buf,
                lrsg_cfg,
            )

            agent.brain.reset_activations()
            agent.brain.inject_input(encoded)

            ctx_end = min(
                lrsg_cfg.num_input_neurons + lrsg_cfg.context_size, agent.brain.N
            )
            agent.brain.activations[
                lrsg_cfg.num_input_neurons : ctx_end
            ] = agent.context_buffer[: ctx_end - lrsg_cfg.num_input_neurons]

            forward_pass(
                agent.brain, lrsg_cfg, theta=agent.theta, last_action=agent.last_action, learning_rate=agent.learning_rate
            )

            new_context = agent.brain.activations[
                lrsg_cfg.num_input_neurons : ctx_end
            ].clone()
            agent.prev_context, agent.context_buffer = (
                agent.context_buffer.clone(),
                (agent.context_buffer + new_context) * 0.5,
            )

            decoded = decode_output(
                agent.brain.read_output(),
                lrsg_cfg,
                tick=tick,
                seen_tokens=agent._seen_tokens,
                graph=agent.brain,
            )

            agent.last_action = decoded["action"]
            agent.last_direction = decoded["direction"]
            agent.last_intent = decoded["intent"]
            agent.whisper_text = decoded.get("whisper_text", "")

            if agent.whisper_text:
                for tok in agent.whisper_text.split():
                    agent._seen_tokens.add(tok)

            if agent.last_action == 4:
                agent.ram -= 1

            lph_decoded = agent.brain.decode_lph()
            log_entry = f"[Tick {tick}] Agent {agent.id} Whispered: \"{lph_decoded}\""
            whisper_log.append(log_entry)
            print(log_entry)

            agent._tick_count += 1

            cost = agent.ACTION_COSTS.get(agent.last_action, 0)
            if agent.last_action == 3:
                DIR = [
                    (1, 0, 0),
                    (-1, 0, 0),
                    (0, 1, 0),
                    (0, -1, 0),
                    (0, 0, 1),
                    (0, 0, -1),
                ]
                if 0 <= agent.last_direction < 6:
                    d = DIR[agent.last_direction]
                    agent.position = tuple(
                        max(0, min(env.grid_size[i] - 1, agent.position[i] + d[i]))
                        for i in range(3)
                    )
                agent.ram -= 1
            elif agent.last_action == 4:
                agent.ram -= 1
            elif agent.last_action == 2:
                for node in env.ram_nodes:
                    if (
                        not node.collected
                        and node.distance_to(agent.position)
                        < arena_cfg.ram_collect_range
                    ):
                        amt = node.collect()
                        agent.ram += amt + arena_cfg.ram_collect_bonus
                        break

            agent.ram = max(0, agent.ram - agent_cfg.ram_decay_rate)
            if agent.ram <= 0:
                agent.alive = False

    return whisper_log


if __name__ == "__main__":
    print("=" * 72)
    print("TWO-AGENT TARGET TRACKING BEHAVIORAL ASSAY (PWM PREDICTION + HEBBIAN)")
    print("Agent 0: Predator/Seeker (masked vision)")
    print("Agent 1: Prey/Evader (perfect RAM lock)")
    print("Ticks: 1000")
    print("=" * 72)
    print()

    log = run_two_agent_communication_assay(num_ticks=1000)

    print()
    print("=" * 72)
    print(f"TOTAL WHISPER EVENTS: {len(log)}")
    non_empty = [l for l in log if '""' not in l]
    print(f"NON-EMPTY WHISPERS: {len(non_empty)}")

    words_a0 = []
    words_a1 = []
    for entry in log:
        if '""' in entry:
            continue
        start = entry.index('"') + 1
        end = entry.index('"', start)
        word = entry[start:end]
        if 'Agent 0' in entry:
            words_a0.append(word)
        else:
            words_a1.append(word)

    print()
    print("AGENT 0 WORD STREAM (first 30 non-empty):")
    for w in words_a0[:30]:
        print(f"  \"{w}\"")

    print()
    print("AGENT 1 WORD STREAM (first 30 non-empty):")
    for w in words_a1[:30]:
        print(f"  \"{w}\"")

    from collections import Counter
    if words_a0:
        top_a0 = Counter(words_a0).most_common(10)
        print()
        print("AGENT 0 TOP WORDS:")
        for w, c in top_a0:
            print(f"  {w:20s}: {c}")
    if words_a1:
        top_a1 = Counter(words_a1).most_common(10)
        print()
        print("AGENT 1 TOP WORDS:")
        for w, c in top_a1:
            print(f"  {w:20s}: {c}")

    print()
    print("=" * 72)
