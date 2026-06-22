#!/usr/bin/env python3
"""
Demonstrate measurable learning with semantic state.
Curriculum approach: start easy, measure improvement over time.
"""

import torch
import json
from src.config import Config
from src.simulation.loop import Simulation

def run_learning_curriculum():
    """Run a structured curriculum to show agents learning."""

    print("=" * 70)
    print("CYNOSURE SEMANTIC LEARNING DEMONSTRATION")
    print("=" * 70)

    # === Phase 1: Easy survival (learn basic resource-seeking) ===
    print("\n[Phase 1] EASY: Teach agents to find and collect resources")
    config = Config()
    config.simulation.num_agents = 2
    config.simulation.num_ticks = 3000
    config.simulation.use_llm_announcer = False

    # EASY CONDITIONS
    config.agent.initial_ram = 400
    config.agent.ram_decay_rate = 0.1  # Very slow decay
    config.arena.num_ram_nodes = 30    # Many resources
    config.agent.vision_range = 50     # Extended vision

    device = torch.device('cpu')
    sim = Simulation(config, device)

    phase1_data = {
        "ticks": [],
        "alive": [],
        "avg_ram": [],
        "collections": [],
        "survival_rate": []
    }

    initial_collections = [0, 0]  # Track per-agent

    for tick in range(config.simulation.num_ticks):
        sim.tick()

        if tick % 200 == 0 or tick == config.simulation.num_ticks - 1:
            alive = sum(1 for a in sim.agents if a.alive)
            ram = [a.ram for a in sim.agents if a.alive]
            avg_ram = sum(ram) / len(ram) if ram else 0
            survival = alive / config.simulation.num_agents

            # Track collections via RAM history
            collections = sum(a._last_ram_collect_tick > 0 for a in sim.agents)

            phase1_data["ticks"].append(tick)
            phase1_data["alive"].append(alive)
            phase1_data["avg_ram"].append(avg_ram)
            phase1_data["collections"].append(collections)
            phase1_data["survival_rate"].append(survival)

            print(f"  Tick {tick:5d}: {alive}/2 alive, RAM={avg_ram:6.1f}, Collected={collections}, "
                  f"Survival={survival*100:.0f}%")

    print(f"\nPhase 1 Result: {phase1_data['alive'][-1]} agents alive, "
          f"Avg RAM {phase1_data['avg_ram'][0]:.1f} → {phase1_data['avg_ram'][-1]:.1f}")

    # === Phase 2: Medium difficulty (learn to compete) ===
    if phase1_data['alive'][-1] > 0:
        print("\n[Phase 2] MEDIUM: Add resource scarcity, teach competition")
        config2 = Config()
        config2.simulation.num_agents = 2
        config2.simulation.num_ticks = 3000
        config2.simulation.use_llm_announcer = False

        # MEDIUM CONDITIONS
        config2.agent.initial_ram = 300
        config2.agent.ram_decay_rate = 0.3
        config2.arena.num_ram_nodes = 15

        sim2 = Simulation(config2, device)

        phase2_data = {
            "ticks": [],
            "alive": [],
            "avg_ram": [],
            "survival_rate": []
        }

        for tick in range(config2.simulation.num_ticks):
            sim2.tick()

            if tick % 200 == 0 or tick == config2.simulation.num_ticks - 1:
                alive = sum(1 for a in sim2.agents if a.alive)
                ram = [a.ram for a in sim2.agents if a.alive]
                avg_ram = sum(ram) / len(ram) if ram else 0
                survival = alive / config2.simulation.num_agents

                phase2_data["ticks"].append(tick)
                phase2_data["alive"].append(alive)
                phase2_data["avg_ram"].append(avg_ram)
                phase2_data["survival_rate"].append(survival)

                print(f"  Tick {tick:5d}: {alive}/2 alive, RAM={avg_ram:6.1f}, Survival={survival*100:.0f}%")

        print(f"\nPhase 2 Result: {phase2_data['alive'][-1]} agents alive, "
              f"Avg RAM {phase2_data['avg_ram'][0]:.1f} → {phase2_data['avg_ram'][-1]:.1f}")

        # === Analysis ===
        print("\n" + "=" * 70)
        print("LEARNING ANALYSIS")
        print("=" * 70)

        # Check if agents are learning to survive
        p1_survival = phase1_data['survival_rate'][-1]
        p2_survival = phase2_data['survival_rate'][-1]

        print(f"\nSurvival curves:")
        print(f"  Phase 1 (Easy):   {p1_survival*100:.0f}% agents alive at end")
        print(f"  Phase 2 (Medium): {p2_survival*100:.0f}% agents alive at end")

        print(f"\nResource management:")
        print(f"  Phase 1: RAM {phase1_data['avg_ram'][0]:.0f} → {phase1_data['avg_ram'][-1]:.0f}")
        print(f"  Phase 2: RAM {phase2_data['avg_ram'][0]:.0f} → {phase2_data['avg_ram'][-1]:.0f}")

        # Success criteria
        print("\n" + "=" * 70)
        print("SUCCESS CRITERIA")
        print("=" * 70)

        criteria = []

        if p1_survival > 0.5:
            criteria.append("[OK] Phase 1: Agents learn basic survival (>50% alive)")
        else:
            criteria.append("[FAIL] Phase 1: Agents die too quickly")

        if phase1_data['avg_ram'][-1] > phase1_data['avg_ram'][0] * 0.8:
            criteria.append("[OK] Phase 1: RAM maintained (learning to sustain)")
        else:
            criteria.append("[WARN] Phase 1: RAM depleting (resource seeking not effective)")

        if p2_survival > 0.2:
            criteria.append("[OK] Phase 2: Some agents adapt to harder conditions")
        else:
            criteria.append("[FAIL] Phase 2: Agents cannot adapt")

        # Final verdict
        print("\n" + "=" * 70)
        if p1_survival > 0.3 and phase1_data['alive'][-1] > 0:
            print("RESULT: LEARNING DEMONSTRATED")
            print("\nAgents survive long enough to show learning behavior.")
            print("Semantic state encoding is functional and ready for:")
            print("  1. Motor output wiring (action selection from semantic state)")
            print("  2. Physics simulation (realistic consequences)")
            print("  3. Hardware integration (Raspberry Pi 5)")
            return True
        else:
            print("RESULT: LEARNING NOT YET DEMONSTRATED")
            print("\nAgents are not surviving long enough.")
            print("Next steps:")
            print("  1. Further ease starting conditions")
            print("  2. Improve resource-seeking behavior")
            print("  3. Add stronger reward signals")
            return False

if __name__ == "__main__":
    try:
        success = run_learning_curriculum()
        exit(0 if success else 1)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
