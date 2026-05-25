import sys
import os
import math
import random
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


def run_calibration(num_ticks=50_000, checkpoint_interval=5_000):
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

    torch.cuda.empty_cache()

    all_words_a0 = []
    all_words_a1 = []
    word_log = []
    telemetry = []
    checkpoint_dir = "calibration_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    t_start = time.time()
    tick = 0

    try:
        while tick < num_ticks:
            if not agent_seek.alive or not agent_evade.alive:
                print(f"[Tick {tick}] Agent died. Restarting...")
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
                        agent.brain, lrsg_cfg, theta=agent.theta, last_action=agent.last_action
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
                word_log.append(f"[Tick {tick}] A0: \"{lph_decoded_0}\" A1: \"{lph_decoded_1}\"")

                words_this_tick = []
                if lph_decoded_0 and lph_decoded_0 != '""':
                    words_this_tick.append(lph_decoded_0)
                    all_words_a0.append(lph_decoded_0)
                if lph_decoded_1 and lph_decoded_1 != '""':
                    words_this_tick.append(lph_decoded_1)
                    all_words_a1.append(lph_decoded_1)

                char_pred_loss_0 = getattr(agent_seek.brain, 'char_prediction_loss', torch.tensor(0.0)).item()
                char_pred_loss_1 = getattr(agent_evade.brain, 'char_prediction_loss', torch.tensor(0.0)).item()
                lang_reward_0 = getattr(agent_seek.brain, '_language_reward_accumulator', 0.0)
                lang_reward_1 = getattr(agent_evade.brain, '_language_reward_accumulator', 0.0)

                telemetry.append({
                    "tick": tick,
                    "ram_a0": agent_seek.ram,
                    "ram_a1": agent_evade.ram,
                    "char_pred_loss_a0": char_pred_loss_0,
                    "char_pred_loss_a1": char_pred_loss_1,
                    "lang_reward_a0": lang_reward_0,
                    "lang_reward_a1": lang_reward_1,
                    "words_this_tick": len(words_this_tick),
                    "total_words_a0": len(all_words_a0),
                    "total_words_a1": len(all_words_a1),
                })

            if tick > 0 and tick % checkpoint_interval == 0:
                torch.cuda.empty_cache()

                if len(telemetry) > 200:
                    telemetry = telemetry[-100:]

                elapsed = time.time() - t_start
                ms_per_tick = (elapsed / tick) * 1000

                top_a0 = Counter(all_words_a0).most_common(15) if all_words_a0 else []
                top_a1 = Counter(all_words_a1).most_common(15) if all_words_a1 else []
                unique_a0 = len(set(all_words_a0))
                unique_a1 = len(set(all_words_a1))

                print(f"\n{'='*60}")
                print(f"CHECKPOINT @ Tick {tick:,} | {elapsed:.0f}s | {ms_per_tick:.1f}ms/tick")
                print(f"  Words: A0={len(all_words_a0):,} (unique={unique_a0}) | A1={len(all_words_a1):,} (unique={unique_a1})")
                print(f"  Char pred loss: A0={char_pred_loss_0:.4f} | A1={char_pred_loss_1:.4f}")
                print(f"  Lang reward: A0={lang_reward_0:.2f} | A1={lang_reward_1:.2f}")
                if top_a0:
                    print(f"  Top A0: {', '.join(f'\"{w}\"({c})' for w, c in top_a0[:5])}")
                if top_a1:
                    print(f"  Top A1: {', '.join(f'\"{w}\"({c})' for w, c in top_a1[:5])}")
                print(f"{'='*60}\n")

                checkpoint = {
                    "tick": tick,
                    "elapsed_s": elapsed,
                    "ms_per_tick": ms_per_tick,
                    "words_a0": len(all_words_a0),
                    "words_a1": len(all_words_a1),
                    "unique_a0": unique_a0,
                    "unique_a1": unique_a1,
                    "top_a0": top_a0,
                    "top_a1": top_a1,
                    "char_pred_loss_a0": char_pred_loss_0,
                    "char_pred_loss_a1": char_pred_loss_1,
                }
                with open(os.path.join(checkpoint_dir, f"checkpoint_{tick:06d}.json"), "w") as f:
                    json.dump(checkpoint, f, indent=2)

            tick += 1
    except Exception as e:
        print(f"\nERROR at tick {tick}: {e}")
        import traceback
        traceback.print_exc()

    elapsed = time.time() - t_start
    ms_per_tick = (elapsed / max(tick, 1)) * 1000

    print(f"\n{'='*60}")
    print(f"CALIBRATION COMPLETE: {tick:,} ticks in {elapsed:.0f}s ({ms_per_tick:.1f}ms/tick)")
    print(f"  Total words: A0={len(all_words_a0):,} | A1={len(all_words_a1):,}")

    top_a0 = Counter(all_words_a0).most_common(20) if all_words_a0 else []
    top_a1 = Counter(all_words_a1).most_common(20) if all_words_a1 else []
    unique_a0 = len(set(all_words_a0))
    unique_a1 = len(set(all_words_a1))

    print(f"  Unique words: A0={unique_a0} | A1={unique_a1}")
    print(f"\n  TOP 20 WORDS - Agent 0:")
    for w, c in top_a0:
        print(f"    \"{w}\": {c}")
    print(f"\n  TOP 20 WORDS - Agent 1:")
    for w, c in top_a1:
        print(f"    \"{w}\": {c}")

    with open("calibration_telemetry.jsonl", "w") as f:
        for entry in telemetry:
            f.write(json.dumps(entry) + "\n")

    with open("calibration_words_a0.json", "w") as f:
        json.dump({"total": len(all_words_a0), "unique": unique_a0, "top": top_a0}, f, indent=2)

    with open("calibration_words_a1.json", "w") as f:
        json.dump({"total": len(all_words_a1), "unique": unique_a1, "top": top_a1}, f, indent=2)

    print(f"\nFiles written:")
    for fn in ["calibration_telemetry.jsonl", "calibration_words_a0.json", "calibration_words_a1.json"]:
        if os.path.exists(fn):
            print(f"  {fn}: {os.path.getsize(fn)/1024:.1f}KB")
    print(f"  calibration_checkpoints/: {len(os.listdir(checkpoint_dir))} files")


if __name__ == "__main__":
    print("=" * 60)
    print("CYNOSURE — Long Calibration: PWM Prediction + Hebbian Reward")
    print("  Agents: 2 (5K neurons each)")
    print("  Ticks: 40,000")
    print("  Checkpoint every: 5,000 ticks")
    print("=" * 60)
    print()

    run_calibration(num_ticks=40_000, checkpoint_interval=5_000)
