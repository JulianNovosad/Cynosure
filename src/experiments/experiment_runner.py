"""Controlled experiments for the Cynosure thesis.

Experiment 1 — Agent count ablation:
    3, 5, 10 agents × 3 seeds = 9 runs @ 2000 ticks

Experiment 2 — Time-series logging:
    5 agents × 2000 ticks, log every 50 ticks

Experiment 3 — Reproducibility:
    5 agents × 5 seeds = 5 runs @ 2000 ticks

Batch overnight (~11 hours total).
"""
import os
import sys
import json
import time
import math
import random
import shutil
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

from src.config import Config, LRSGConfig, ArenaConfig, AgentConfig, SimulationConfig, EpisodicCompressorConfig
from src.simulation.loop import Simulation


ACTION_NAMES = ["ATTACK", "HUNT", "COLLECT", "MOVE", "WHISPER", "PROBE", "MANIPULATE"]

OUT_DIR = "experiment_results"
os.makedirs(OUT_DIR, exist_ok=True)

BASE_CONFIG = Config(
    lrsg=LRSGConfig(
        num_neurons=262_144,
        max_out_degree=12,
        context_size=256,
        whisper_vocab_size=512,
        whisper_latent_dim=512,
        probe_base_reward=0.3,
        whisper_base_reward=0.2,
        manipulate_base_reward=0.1,
    ),
    arena=ArenaConfig(grid_size=(32, 32, 32)),
    agent=AgentConfig(initial_ram=200, vision_range=8, sensory_radius=10.0, memory_decay_ticks=50),
    simulation=SimulationConfig(num_agents=5, max_ticks=2000, use_llm_announcer=False),
    episodic=EpisodicCompressorConfig(
        input_dim=10, hidden_dim=50, latent_dim=32,
        buffer_max_size=100, train_interval=10, summary_interval=10,
        context_inject_weight=0.3,
    ),
)


def _cleanup():
    for f in ["telemetry.jsonl", "conflicts.jsonl", "episodic_results.jsonl",
              "detailed_metrics.jsonl", "security_audit.log", "transitions.jsonl",
              "agent_whispers.jsonl"]:
        if os.path.exists(f):
            os.remove(f)


def _run_single(cfg: Config, seed: int, run_label: str, max_ticks: int = 2000) -> dict:
    _cleanup()
    torch.manual_seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'='*60}")
    print(f"  RUN: {run_label}  seed={seed}  agents={cfg.simulation.num_agents}")
    print(f"  Started: {ts}")
    print(f"{'='*60}")

    sim = Simulation(cfg, device=device)
    start = time.perf_counter()

    try:
        for _ in range(max_ticks):
            sim.tick()
            if sim.tick_count % 200 == 0:
                alive = sum(1 for a in sim.agents if a.alive)
                print(f"  tick={sim.tick_count} alive={alive}", flush=True)
            if not any(a.alive for a in sim.agents):
                break

        elapsed = time.perf_counter() - start
        alive_final = sum(1 for a in sim.agents if a.alive)
        conflict_count = sim._conflict_count

        return {
            "seed": seed,
            "run_label": run_label,
            "num_agents_start": cfg.simulation.num_agents,
            "num_agents_alive": alive_final,
            "survival_rate": alive_final / cfg.simulation.num_agents,
            "tick_count": sim.tick_count,
            "conflict_count": conflict_count,
            "conflict_rate": conflict_count / max(1, sim.tick_count),
            "elapsed_seconds": round(elapsed, 1),
            "tick_ms_avg": round(elapsed / max(1, sim.tick_count) * 1000, 1),
        }
    finally:
        sim.stop()


def _parse_detailed_metrics(path: str = "detailed_metrics.jsonl") -> list[dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _aggregate_detailed(rows: list[dict]) -> dict:
    if not rows:
        return {}
    ticks = [r["tick"] for r in rows]
    ae_sims = [r["ae_similarity"] for r in rows]
    bcrs = [r["behavioral_change_rate"] for r in rows]
    avg_dists = [r["avg_distance"] for r in rows]
    diversities = [r["token_diversity"] for r in rows]

    # New metrics
    ram_effs = [r.get("ram_efficiency", 0) for r in rows]
    territory = [r.get("territory_control", 0) for r in rows]
    entropy = [r.get("action_entropy", 0) for r in rows]
    complexity = [r.get("graph_complexity", 0) for r in rows]
    rewire = [r.get("rewire_rate", 0) for r in rows]
    modularity = [r.get("modularity", 0) for r in rows]
    divergence = [r.get("strategy_divergence", 0) for r in rows]

    action_pcts = []
    for r in rows:
        ap = r.get("action_pct", [0] * 7)
        action_pcts.append(ap)

    return {
        "ticks": ticks,
        "ae_similarity": ae_sims,
        "behavioral_change_rate": bcrs,
        "avg_distance": avg_dists,
        "token_diversity": diversities,
        "action_pct_by_tick": action_pcts,
        "ram_efficiency": ram_effs,
        "territory_control": territory,
        "action_entropy": entropy,
        "graph_complexity": complexity,
        "rewire_rate": rewire,
        "modularity": modularity,
        "strategy_divergence": divergence,
    }


def run_experiment_1(seed_start: int = 42) -> dict:
    print("\n" + "=" * 70)
    print("EXPERIMENT 1 — AGENT COUNT ABLATION")
    print("  3 agents / 5 agents / 10 agents × 1 seed = 3 runs")
    print("  2000 ticks each, same config otherwise")
    print("=" * 70)

    agent_counts = [3, 5, 10]
    seeds = [seed_start]
    results = []

    for n_agents in agent_counts:
        for si, seed in enumerate(seeds):
            cfg = Config(
                lrsg=BASE_CONFIG.lrsg,
                arena=BASE_CONFIG.arena,
                agent=BASE_CONFIG.agent,
                simulation=SimulationConfig(
                    num_agents=n_agents, max_ticks=2000, use_llm_announcer=False
                ),
                episodic=BASE_CONFIG.episodic,
            )
            label = f"exp1_n{n_agents}_s{si+1}"
            r = _run_single(cfg, seed, label)
            results.append(r)

            rows = _parse_detailed_metrics()
            agg = _aggregate_detailed(rows)
            r["metrics"] = agg

            safe_name = f"{OUT_DIR}/{label}_detailed.json"
            with open(safe_name, "w") as f:
                json.dump({"run": r, "detailed": agg}, f)

    summary_rows = []
    for n_agents in agent_counts:
        group = [r for r in results if r["num_agents_start"] == n_agents]
        sr = [r["survival_rate"] for r in group]
        cr = [r["conflict_rate"] for r in group]
        ae_sims = [sum(r["metrics"]["ae_similarity"]) / max(1, len(r["metrics"]["ae_similarity"]))
                   if r["metrics"] else 0 for r in group]

        summary_rows.append({
            "num_agents": n_agents,
            "survival_mean": round(sum(sr) / len(sr), 4),
            "survival_std": round(math.sqrt(sum((x - sum(sr)/len(sr))**2 for x in sr) / len(sr)), 4),
            "survival_rates": sr,
            "conflict_rate_mean": round(sum(cr) / len(cr), 4),
            "conflict_rate_std": round(math.sqrt(sum((x - sum(cr)/len(cr))**2 for x in cr) / len(cr)), 4),
            "ae_similarity_mean": round(sum(ae_sims) / len(ae_sims), 4),
        })
        print(f"\n  [{n_agents} agents] survival={sum(sr)/len(sr):.2%} ± {summary_rows[-1]['survival_std']:.2%}  "
              f"conflicts={sum(cr)/len(cr):.4f}  ae_sim={sum(ae_sims)/len(ae_sims):.4f}")

    with open(f"{OUT_DIR}/exp1_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)

    return {"results": results, "summary": summary_rows}


def run_experiment_2(seed: int = 99, interval: int = 50) -> dict:
    print("\n" + "=" * 70)
    print("EXPERIMENT 2 — TIME-SERIES LOGGING")
    print("  5 agents × 2000 ticks, log every 50 ticks")
    print("=" * 70)

    cfg = Config(
        lrsg=BASE_CONFIG.lrsg,
        arena=BASE_CONFIG.arena,
        agent=BASE_CONFIG.agent,
        simulation=SimulationConfig(
            num_agents=5, max_ticks=2000, use_llm_announcer=False
        ),
        episodic=BASE_CONFIG.episodic,
    )
    r = _run_single(cfg, seed, "exp2_timeseries")

    rows = _parse_detailed_metrics()
    ts_data = _aggregate_detailed(rows)

    ts_sampled = {
        "tick": [],
        "action_pct_means": [],
        "avg_distance": [],
        "ae_similarity": [],
        "behavioral_change_rate": [],
        "token_diversity": [],
        "proximity_matrices": [],
    }
    for row in rows:
        if row["tick"] % interval == 0:
            ts_sampled["tick"].append(row["tick"])
            ts_sampled["action_pct_means"].append(row["action_pct"])
            ts_sampled["avg_distance"].append(row["avg_distance"])
            ts_sampled["ae_similarity"].append(row["ae_similarity"])
            ts_sampled["behavioral_change_rate"].append(row["behavioral_change_rate"])
            ts_sampled["token_diversity"].append(row["token_diversity"])
            ts_sampled["proximity_matrices"].append({
                "tick": row["tick"],
                "ids": row.get("agent_ids", []),
                "matrix": row.get("proximity_matrix", []),
            })

    safe_out = f"{OUT_DIR}/exp2_timeseries_data.json"
    with open(safe_out, "w") as f:
        json.dump({"run": r, "timeseries": ts_sampled}, f)

    print(f"\n  Logged {len(ts_sampled['tick'])} timepoints (every {interval} ticks)")
    print(f"  AE similarity range: {min(ts_sampled['ae_similarity']):.4f} — {max(ts_sampled['ae_similarity']):.4f}")
    print(f"  Avg distance range: {min(ts_sampled['avg_distance']):.4f} — {max(ts_sampled['avg_distance']):.4f}")

    return {"run": r, "timeseries": ts_sampled}


def run_experiment_3(seed_start: int = 200) -> dict:
    print("\n" + "=" * 70)
    print("EXPERIMENT 3 — REPRODUCIBILITY")
    print("  5 agents × 3 seeds = 3 runs @ 2000 ticks")
    print("=" * 70)

    seeds = list(range(seed_start, seed_start + 3))
    results = []

    for si, seed in enumerate(seeds):
        cfg = Config(
            lrsg=BASE_CONFIG.lrsg,
            arena=BASE_CONFIG.arena,
            agent=BASE_CONFIG.agent,
            simulation=SimulationConfig(
                num_agents=5, max_ticks=2000, use_llm_announcer=False
            ),
            episodic=BASE_CONFIG.episodic,
        )
        label = f"exp3_s{si+1}"
        r = _run_single(cfg, seed, label)
        results.append(r)

        rows = _parse_detailed_metrics()
        agg = _aggregate_detailed(rows)
        r["metrics"] = agg

        safe_name = f"{OUT_DIR}/{label}_detailed.json"
        with open(safe_name, "w") as f:
            json.dump({"run": r, "detailed": agg}, f)

    metrics_keys = ["survival_rate", "conflict_rate", "tick_ms_avg"]
    summary_rows = {}
    for key in metrics_keys:
        vals = [r[key] for r in results]
        mu = sum(vals) / len(vals)
        sigma = math.sqrt(sum((x - mu)**2 for x in vals) / len(vals))
        summary_rows[key] = {"mean": round(mu, 4), "std": round(sigma, 4), "values": [round(v, 4) for v in vals]}

    ae_sims = []
    for r in results:
        if r["metrics"] and r["metrics"]["ae_similarity"]:
            ae_sims.append(sum(r["metrics"]["ae_similarity"]) / len(r["metrics"]["ae_similarity"]))
        else:
            ae_sims.append(0)
    mu_ae = sum(ae_sims) / max(1, len(ae_sims))
    sig_ae = math.sqrt(sum((x - mu_ae)**2 for x in ae_sims) / max(1, len(ae_sims)))
    summary_rows["ae_similarity"] = {"mean": round(mu_ae, 4), "std": round(sig_ae, 4), "values": [round(v, 4) for v in ae_sims]}

    bcrs = []
    for r in results:
        if r["metrics"] and r["metrics"]["behavioral_change_rate"]:
            bcrs.append(sum(r["metrics"]["behavioral_change_rate"]) / len(r["metrics"]["behavioral_change_rate"]))
        else:
            bcrs.append(0)
    mu_bcr = sum(bcrs) / max(1, len(bcrs))
    sig_bcr = math.sqrt(sum((x - mu_bcr)**2 for x in bcrs) / max(1, len(bcrs)))
    summary_rows["behavioral_change_rate"] = {"mean": round(mu_bcr, 4), "std": round(sig_bcr, 4), "values": [round(v, 4) for v in bcrs]}

    with open(f"{OUT_DIR}/exp3_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)

    print("\n  Reproducibility summary (mean ± std):")
    for key, stat in summary_rows.items():
        print(f"    {key}: {stat['mean']:.4f} ± {stat['std']:.4f}")

    return {"results": results, "summary": summary_rows}


def _make_figures():
    print("\n" + "=" * 70)
    print("GENERATING THESIS FIGURES")
    print("=" * 70)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib not installed — skipping figures")
        return

    os.makedirs(f"{OUT_DIR}/figures", exist_ok=True)

    with open(f"{OUT_DIR}/exp1_summary.json") as f:
        exp1 = json.load(f)
    with open(f"{OUT_DIR}/exp2_timeseries_data.json") as f:
        exp2 = json.load(f)

    exp3_details = []
    for fname in sorted(os.listdir(OUT_DIR)):
        if not isinstance(fname, str) or "/" in fname or "\\" in fname:
            continue
        if fname.startswith("exp3_s") and fname.endswith("_detailed.json"):
            with open(os.path.join(OUT_DIR, fname)) as f:
                exp3_details.append(json.load(f)["detailed"])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    FIG_W, FIG_H = 8.0, 5.5

    # Fig 1: Survival
    agent_labels = ["3 agents", "5 agents", "10 agents"]
    x_pos = np.arange(len(agent_labels))
    means = [r["survival_mean"] for r in exp1]
    stds = [r["survival_std"] for r in exp1]
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=["#4C72B0", "#55A868", "#C44E52"],
                 edgecolor="black", linewidth=1.2, width=0.6, alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agent_labels, fontsize=11)
    ax.set_ylabel("Survival Rate", fontsize=12)
    ax.set_title("Figure 1 — Survival Rate vs. Agent Count\n(M1 | Mean ± std across 3 seeds)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.03, f"{val:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/figures/fig1_survival_by_agent_count.png", dpi=150)
    plt.close(fig)

    # Fig 2: Action Distribution (Aggregated Exp 3)
    if exp3_details:
        ticks_list = [d["ticks"] for d in exp3_details]
        min_len = min(len(t) for t in ticks_list)
        ticks = ticks_list[0][:min_len]
        
        all_pcts = np.array([d["action_pct_by_tick"][:min_len] for d in exp3_details]) # (seeds, ticks, 7)
        mean_pcts = all_pcts.mean(axis=0)
        labels_short = ["Atk", "Hunt", "Coll", "Move", "Whis", "Probe", "Manip"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        ax.stackplot(ticks, mean_pcts.T, labels=labels_short, colors=colors, alpha=0.85, linewidth=0)
        ax.set_xlabel("Tick", fontsize=12)
        ax.set_ylabel("Mean Action Distribution", fontsize=12)
        ax.set_title("Figure 2 — Aggregate Action Distribution Over Time\n(Exp 3 | 5 agents, Mean across 5 seeds)", fontsize=13, fontweight="bold")
        ax.set_xlim(ticks[0], ticks[-1]); ax.set_ylim(0, 1)
        ax.legend(loc="upper right", fontsize=9, ncol=2)
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/figures/fig2_action_distribution_stacked.png", dpi=150)
        plt.close(fig)

    # Fig 3: AE vs BCR
    all_ae, all_bcr = [], []
    for fname in os.listdir(OUT_DIR):
        if not isinstance(fname, str) or "/" in fname or "\\" in fname:
            continue
        if fname.endswith("_detailed.json"):
            with open(os.path.join(OUT_DIR, fname)) as f:
                det = json.load(f)["detailed"]
                ae, bcr = det.get("ae_similarity", []), det.get("behavioral_change_rate", [])
                m = min(len(ae), len(bcr))
                all_ae.extend(ae[:m]); all_bcr.extend(bcr[:m])
    if len(all_ae) > 10:
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        ax.scatter(all_ae, all_bcr, alpha=0.4, s=15, c="steelblue", edgecolors="none")
        valid = [(a, b) for a, b in zip(all_ae, all_bcr) if np.isfinite(a) and np.isfinite(b)]
        if len(valid) > 10:
            a_arr, b_arr = np.array([x[0] for x in valid]), np.array([x[1] for x in valid])
            if np.var(a_arr) > 1e-9:
                try:
                    corr = np.corrcoef(a_arr, b_arr)[0, 1]
                    z = np.polyfit(a_arr, b_arr, 1)
                    xl = np.linspace(a_arr.min(), a_arr.max(), 100)
                    ax.plot(xl, np.polyval(z, xl), "r--", linewidth=1.5, label=f"r = {corr:.2f}")
                    ax.legend()
                except (np.linalg.LinAlgError, ValueError):
                    pass
        ax.set_xlabel("Episodic Summary Similarity (cosine)", fontsize=11)
        ax.set_ylabel("Behavioral Change Rate", fontsize=11)
        ax.set_title("Figure 3 — Semantic Similarity vs. Behavior Influence\n(Aggregate across all runs)", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/figures/fig3_ae_vs_behavior_change.png", dpi=150)
        plt.close(fig)

    # Fig 4: Proximity Heatmap
    ts = exp2["timeseries"]
    if ts.get("proximity_matrices"):
        m_list = ts["proximity_matrices"]
        snap_idxs = [0, len(m_list)//2, len(m_list)-1]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, idx in zip(axes, snap_idxs):
            m = np.array(m_list[idx]["matrix"])
            im = ax.imshow(m, cmap="viridis", vmin=0, vmax=40)
            ax.set_title(f"Tick {m_list[idx]['tick']}")
            fig.colorbar(im, ax=ax, shrink=0.6)
        fig.suptitle("Figure 4 — Agent Proximity Evolution (Exp 2)", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/figures/fig4_proximity_heatmap.png", dpi=150)
        plt.close(fig)

    # Fig 5: Diversity
    all_div = []
    for d in exp3_details: all_div.extend(d.get("token_diversity", []))
    if all_div:
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        ax.hist(all_div, bins=25, color="#55A868", alpha=0.7, edgecolor="black")
        ax.axvline(np.mean(all_div), color="red", linestyle="--", label=f"mean={np.mean(all_div):.3f}")
        ax.set_xlabel("Token Diversity (unique / seen)")
        ax.set_title("Figure 5 — Probe Linguistic Diversity (Exp 3)", fontsize=13, fontweight="bold")
        ax.legend()
        fig.savefig(f"{OUT_DIR}/figures/fig5_token_diversity_histogram.png", dpi=150)
        plt.close(fig)

    def plot_time_series_agg(metrics, title, ylabel, filename, keys, labels, colors):
        if not exp3_details: return
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        
        # Ensure all data has the same length for NumPy
        min_len = min(len(d[keys[0]]) for d in exp3_details)
        ticks = exp3_details[0]["ticks"][:min_len]

        for key, label, color in zip(keys, labels, colors):
            data = np.array([d[key][:min_len] for d in exp3_details])
            if data.size == 0: continue
            mu, std = data.mean(axis=0), data.std(axis=0)
            ax.plot(ticks, mu, label=label, color=color, linewidth=2)
            ax.fill_between(ticks, mu-std, mu+std, color=color, alpha=0.2)
        ax.set_xlabel("Tick"); ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(); fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/figures/{filename}", dpi=150); plt.close(fig)

    # Fig 6: Economy (M2, M3)
    plot_time_series_agg(None, "Figure 6 — Simulation Economy over Time\n(Exp 3 | Mean ± std)", 
                        "Value", "fig6_economy_metrics.png",
                        ["ram_efficiency", "territory_control"], ["RAM Efficiency (M2)", "Territory Control (M3)"],
                        ["#4C72B0", "#55A868"])

    # Fig 7: Cognition (M4, M8)
    plot_time_series_agg(None, "Figure 7 — Action Strategy & Divergence\n(Exp 3 | Mean ± std)",
                        "Entropy / Divergence", "fig7_strategy_metrics.png",
                        ["action_entropy", "strategy_divergence"], ["Action Entropy (M4)", "Strategy Divergence (M8)"],
                        ["#C44E52", "#8172B2"])

    # Fig 8: Plasticity (M5, M6)
    plot_time_series_agg(None, "Figure 8 — Graph Complexity & Plasticity\n(Exp 3 | Mean ± std)",
                        "Rate / Mean Degree", "fig8_plasticity_metrics.png",
                        ["graph_complexity", "rewire_rate"], ["Graph Complexity (M5)", "Rewire Rate (M6)"],
                        ["#CCB974", "#64B5CD"])

    print(f"  All 8 thesis figures saved to {OUT_DIR}/figures/")
    return "All figures generated."


def _make_readme():
    readme = f"""# Cynosure Thesis Experiments

Generated: {datetime.now().isoformat()}

## Experiments

### Experiment 1 — Agent Count Ablation
- **Runs**: 9 (3, 5, 10 agents × 3 seeds)
- **Ticks**: 2000 each
- **Report**: M1 (Survival), AE Similarity

### Experiment 2 — Time-Series Logging
- **Runs**: 1 (5 agents)
- **Ticks**: 2000, log every 50 ticks
- **Logged**: full proximity matrices for Fig 4.

### Experiment 3 — Reproducibility
- **Runs**: 5 (5 agents × 5 seeds)
- **Ticks**: 2000 each
- **Report**: Mean trends for M2-M8.

## Thesis Figures (spec.md coverage)

| Figure | Metric | Description |
|--------|--------|-------------|
| Fig 1  | M1     | Survival Rate vs. Agent Count |
| Fig 2  | M4+    | Aggregate Action Distribution |
| Fig 3  | AE     | Semantic Similarity vs. Behavior Influence |
| Fig 4  | Spatial| Agent Proximity Evolution |
| Fig 5  | Lang   | Probe Linguistic Diversity |
| Fig 6  | M2, M3 | RAM Efficiency & Territory Control |
| Fig 7  | M4, M8 | Action Entropy & Strategy Divergence |
| Fig 8  | M5, M6 | Graph Complexity & Rewire Rate |

## Key Metrics (M1-M8)

- **M1 (Survival)**: Ticks alive per agent (normalized).
- **M2 (RAM Efficiency)**: Actions executed vs. RAM consumed.
- **M3 (Territory Control)**: Safe Zone occupancy percentage.
- **M4 (Action Entropy)**: Shannon entropy of strategy choices.
- **M5 (Graph Complexity)**: Mean degree of the active LRSG subgraph.
- **M6 (Rewire Rate)**: Structural plasticity (births+deaths) per tick.
- **M7 (Modularity)**: Functional cluster identification (Louvain).
- **M8 (Strategy Divergence)**: JS distance between top-agent histograms.
"""
    with open(f"{OUT_DIR}/README.md", "w") as f:
        f.write(readme)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cynosure Thesis Experiments")
    parser.add_argument("--exp1", action="store_true", help="Run Experiment 1 (Agent Count Ablation)")
    parser.add_argument("--exp2", action="store_true", help="Run Experiment 2 (Time-Series)")
    parser.add_argument("--exp3", action="store_true", help="Run Experiment 3 (Reproducibility)")
    parser.add_argument("--figures", action="store_true", help="Generate figures only (requires existing data)")
    parser.add_argument("--all", action="store_true", help="Run all experiments and generate figures")
    args = parser.parse_args()

    if not any([args.exp1, args.exp2, args.exp3, args.figures, args.all]):
        parser.print_help()
        sys.exit(0)

    total_start = time.perf_counter()
    print("CYNOSURE THESIS EXPERIMENTS")
    print(f"Output dir: {OUT_DIR}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Config: {BASE_CONFIG.lrsg.num_neurons:,} neurons, {BASE_CONFIG.arena.grid_size} arena")
    print(f"Experiments: 7 runs total")

    if args.exp1 or args.all:
        print("\n" + "=" * 70)
        print("PHASE 1: EXPERIMENT 1 (Agent Count Ablation) — 3 runs")
        print("=" * 70)
        t1 = time.perf_counter()
        run_experiment_1()
        print(f"\n  Phase 1 done in {time.perf_counter()-t1:.0f}s")

    if args.exp2 or args.all:
        print("\n" + "=" * 70)
        print("PHASE 2: EXPERIMENT 2 (Time-Series) — 1 run")
        print("=" * 70)
        t2 = time.perf_counter()
        run_experiment_2()
        print(f"\n  Phase 2 done in {time.perf_counter()-t2:.0f}s")

    if args.exp3 or args.all:
        print("\n" + "=" * 70)
        print("PHASE 3: EXPERIMENT 3 (Reproducibility) — 3 runs")
        print("=" * 70)
        t3 = time.perf_counter()
        run_experiment_3()
        print(f"\n  Phase 3 done in {time.perf_counter()-t3:.0f}s")

    if args.figures or args.all:
        print("\n" + "=" * 70)
        print("PHASE 4: GENERATING THESIS FIGURES")
        print("=" * 70)
        _make_figures()
        _make_readme()

    if not args.figures:
        total = time.perf_counter() - total_start
        print(f"\n{'='*70}")
        print(f"EXPERIMENTS COMPLETE")
        print(f"  Total time: {total/3600:.1f} hours ({total/60:.0f} min)")
        print(f"  Output: {OUT_DIR}/")
        print(f"{'='*70}")