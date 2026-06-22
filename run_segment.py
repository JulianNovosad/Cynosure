import sys
import os
import math
import json
import time
import torch
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from src.config import AgentConfig, LRSGConfig, ArenaConfig
from src.lrsg import LRSGraph
from src.lrsg.forward import forward_pass
from src.lrsg.io import encode_input, decode_output
from src.arena import Environment, RAMNode
from src.agent import Agent


class MaskedVisionAgent(Agent):
    def _compute_vision(self, env, agents):
        V = self.lrsg_config.vision_dims
        return torch.zeros(V, device=self.brain.device)


class PerfectVisionAgent(Agent):
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


def save_agent_state(agent, path):
    """Save agent runtime state for cross-segment persistence."""
    state = {
        "position": agent.position,
        "ram": agent.ram,
        "alive": agent.alive,
        "last_action": agent.last_action,
        "last_direction": agent.last_direction,
        "last_intent": agent.last_intent,
        "whisper_text": agent.whisper_text,
        "_tick_count": agent._tick_count,
        "context_buffer": agent.context_buffer.detach().clone(),
        "prev_context": agent.prev_context.detach().clone(),
        "_seen_tokens": list(agent._seen_tokens),
        "action_history": agent.action_history,
    }
    torch.save(state, path)


def load_agent_state(agent, path):
    """Load agent runtime state from checkpoint."""
    state = torch.load(path, map_location=agent.brain.device, weights_only=True)
    agent.position = tuple(state["position"])
    agent.ram = float(state["ram"])
    agent.alive = bool(state["alive"])
    agent.last_action = int(state["last_action"])
    agent.last_direction = int(state["last_direction"])
    agent.last_intent = int(state["last_intent"])
    agent.whisper_text = str(state["whisper_text"])
    agent._tick_count = int(state["_tick_count"])
    agent.context_buffer.copy_(state["context_buffer"])
    agent.prev_context.copy_(state["prev_context"])
    agent._seen_tokens = set(state["_seen_tokens"])
    agent.action_history = list(state["action_history"])


def main():
    segment = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    start_tick = segment * 20_000
    num_ticks = 20_000

    print(f"Running segment {segment} (ticks {start_tick:,}-{start_tick+num_ticks:,})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    lrsg_cfg = LRSGConfig(
        num_neurons=5000,
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
        initial_ram=5000,
        ram_decay_rate=0.0001,
        vision_range=8,
        sensory_radius=25.0,
        memory_decay_ticks=50,
    )

    checkpoint_dir = "calibration_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    env = Environment(arena_cfg)
    hidden_ram = RAMNode(position=(35.0, 35.0, 35.0), yield_amount=50)
    env.ram_nodes.append(hidden_ram)

    brain_seek = LRSGraph(lrsg_cfg, device=device)
    brain_evade = LRSGraph(lrsg_cfg, device=device)

    agent_seek = MaskedVisionAgent(
        agent_id=0, position=(10, 10, 10), brain=brain_seek,
        config=agent_cfg, lrsg_config=lrsg_cfg,
    )
    agent_evade = PerfectVisionAgent(
        agent_id=1, position=(40, 40, 40), brain=brain_evade,
        config=agent_cfg, lrsg_config=lrsg_cfg,
        target_ram_node=hidden_ram,
    )
    agents = [agent_seek, agent_evade]

    # Load previous segment weights if they exist
    if segment > 0:
        prev_seek = os.path.join(checkpoint_dir, f"brain_seek_{(segment-1)*20000:06d}.pt")
        prev_evade = os.path.join(checkpoint_dir, f"brain_evade_{(segment-1)*20000:06d}.pt")
        agent_seek_state = os.path.join(checkpoint_dir, f"agent_seek_{(segment-1)*20000:06d}.pt")
        agent_evade_state = os.path.join(checkpoint_dir, f"agent_evade_{(segment-1)*20000:06d}.pt")

        if os.path.exists(prev_seek):
            brain_seek.load_state(prev_seek)
            print(f"  Loaded brain_seek from segment {segment-1}")
        if os.path.exists(prev_evade):
            brain_evade.load_state(prev_evade)
            print(f"  Loaded brain_evade from segment {segment-1}")
        if os.path.exists(agent_seek_state):
            load_agent_state(agent_seek, agent_seek_state)
            print(f"  Loaded agent_seek state from segment {segment-1}")
        if os.path.exists(agent_evade_state):
            load_agent_state(agent_evade, agent_evade_state)
            print(f"  Loaded agent_evade state from segment {segment-1}")

    torch.cuda.empty_cache()

    all_words_a0 = []
    all_words_a1 = []
    telemetry = []
    tick = 0
    t_start = time.time()

    while tick < num_ticks:
        if not agent_seek.alive or not agent_evade.alive:
            agent_seek.alive = True
            agent_seek.ram = agent_cfg.initial_ram
            agent_evade.alive = True
            agent_evade.ram = agent_cfg.initial_ram

        with torch.no_grad():
            for agent in agents:
                other = agent_evade if agent is agent_seek else agent_seek
                received_whisper_buf = torch.zeros(lrsg_cfg.whisper_buffer_dim, device=device)
                if other.whisper_text:
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
                    tick=start_tick + tick,
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

                agent._tick_count += 1

                if agent.last_action == 3:
                    DIR = [
                        (1, 0, 0), (-1, 0, 0), (0, 1, 0),
                        (0, -1, 0), (0, 0, 1), (0, 0, -1),
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

        if tick % 1000 == 0:
            torch.cuda.empty_cache()

            lph_decoded_0 = agent_seek.brain.decode_lph()
            lph_decoded_1 = agent_evade.brain.decode_lph()

            if lph_decoded_0 and lph_decoded_0 != '""':
                all_words_a0.append(lph_decoded_0)
            if lph_decoded_1 and lph_decoded_1 != '""':
                all_words_a1.append(lph_decoded_1)

            char_pred_loss_0 = getattr(agent_seek.brain, 'char_prediction_loss', torch.tensor(0.0)).item()
            char_pred_loss_1 = getattr(agent_evade.brain, 'char_prediction_loss', torch.tensor(0.0)).item()
            lang_reward_0 = getattr(agent_seek.brain, '_language_reward_accumulator', 0.0)
            lang_reward_1 = getattr(agent_evade.brain, '_language_reward_accumulator', 0.0)

            telemetry.append({
                "tick": start_tick + tick,
                "ram_a0": agent_seek.ram,
                "ram_a1": agent_evade.ram,
                "char_pred_loss_a0": char_pred_loss_0,
                "char_pred_loss_a1": char_pred_loss_1,
                "lang_reward_a0": lang_reward_0,
                "lang_reward_a1": lang_reward_1,
                "total_words_a0": len(all_words_a0),
                "total_words_a1": len(all_words_a1),
            })

        tick += 1

    elapsed = time.time() - t_start
    ms_per_tick = (elapsed / tick) * 1000

    # Save weights and state for next segment
    brain_seek.save_state(os.path.join(checkpoint_dir, f"brain_seek_{start_tick:06d}.pt"))
    brain_evade.save_state(os.path.join(checkpoint_dir, f"brain_evade_{start_tick:06d}.pt"))
    save_agent_state(agent_seek, os.path.join(checkpoint_dir, f"agent_seek_{start_tick:06d}.pt"))
    save_agent_state(agent_evade, os.path.join(checkpoint_dir, f"agent_evade_{start_tick:06d}.pt"))

    char_pred_loss_0 = getattr(agent_seek.brain, 'char_prediction_loss', torch.tensor(0.0)).item()
    char_pred_loss_1 = getattr(agent_evade.brain, 'char_prediction_loss', torch.tensor(0.0)).item()
    lang_reward_0 = getattr(agent_seek.brain, '_language_reward_accumulator', 0.0)
    lang_reward_1 = getattr(agent_evade.brain, '_language_reward_accumulator', 0.0)

    print(f"\nSegment {segment} complete: {tick:,} ticks in {elapsed:.0f}s ({ms_per_tick:.1f}ms/tick)")
    print(f"  Words: A0={len(all_words_a0)} | A1={len(all_words_a1)}")
    print(f"  Char pred loss: A0={char_pred_loss_0:.4f} | A1={char_pred_loss_1:.4f}")
    print(f"  Lang reward: A0={lang_reward_0:.2f} | A1={lang_reward_1:.2f}")

    if all_words_a0:
        top = Counter(all_words_a0).most_common(5)
        print(f"  Top A0: {', '.join(f'\"{w}\"({c})' for w, c in top)}")
    if all_words_a1:
        top = Counter(all_words_a1).most_common(5)
        print(f"  Top A1: {', '.join(f'\"{w}\"({c})' for w, c in top)}")

    # Save segment results
    segment_file = os.path.join(checkpoint_dir, f"segment_{start_tick:06d}.json")
    result = {
        "start_tick": start_tick,
        "end_tick": start_tick + tick,
        "words_a0": len(all_words_a0),
        "words_a1": len(all_words_a1),
        "top_a0": Counter(all_words_a0).most_common(10),
        "top_a1": Counter(all_words_a1).most_common(10),
        "final_char_pred_loss_a0": char_pred_loss_0,
        "final_char_pred_loss_a1": char_pred_loss_1,
        "final_lang_reward_a0": lang_reward_0,
        "final_lang_reward_a1": lang_reward_1,
    }
    with open(segment_file, "w") as f:
        json.dump(result, f, indent=2)

    # Save telemetry
    telemetry_file = os.path.join(checkpoint_dir, f"telemetry_{start_tick:06d}.jsonl")
    with open(telemetry_file, "w") as f:
        for entry in telemetry:
            f.write(json.dumps(entry) + "\n")

    print(f"\nSaved: {segment_file}")
    print(f"Saved: {telemetry_file}")
    print(f"Saved: brain_seek_{start_tick:06d}.pt, brain_evade_{start_tick:06d}.pt")
    print(f"Saved: agent_seek_{start_tick:06d}.pt, agent_evade_{start_tick:06d}.pt")


if __name__ == "__main__":
    main()
