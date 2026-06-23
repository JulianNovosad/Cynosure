import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "experiment_results"
os.makedirs(f"{OUT_DIR}/figures", exist_ok=True)

def generate_all():
    print("GENERATING THESIS FIGURES...")

    # Load Summaries
    with open(f"{OUT_DIR}/exp1_summary.json") as f:
        exp1 = json.load(f)
    with open(f"{OUT_DIR}/exp2_timeseries_data.json") as f:
        exp2 = json.load(f)

    # Load Exp 3 Details (for aggregate trends)
    exp3_details = []
    for fname in sorted(os.listdir(OUT_DIR)):
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

    # Fig 1: Survival (M1)
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
        ticks = exp3_details[0]["ticks"]
        all_pcts = np.array([d["action_pct_by_tick"] for d in exp3_details]) # (seeds, ticks, 7)
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
        if fname.endswith("_detailed.json"):
            with open(os.path.join(OUT_DIR, fname)) as f:
                det = json.load(f)["detailed"]
                ae, bcr = det.get("ae_similarity", []), det.get("behavioral_change_rate", [])
                m = min(len(ae), len(bcr))
                all_ae.extend(ae[:m]); all_bcr.extend(bcr[:m])
    if len(all_ae) > 10:
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        ax.scatter(all_ae, all_bcr, alpha=0.4, s=15, c="steelblue", edgecolors="none")
        valid = [(a, b) for a, b in zip(all_ae, all_bcr) if abs(a) < 2 and abs(b) < 2]
        if len(valid) > 10:
            a_arr, b_arr = np.array([x[0] for x in valid]), np.array([x[1] for x in valid])
            corr = np.corrcoef(a_arr, b_arr)[0, 1]
            z = np.polyfit(a_arr, b_arr, 1)
            xl = np.linspace(a_arr.min(), a_arr.max(), 100)
            ax.plot(xl, np.polyval(z, xl), "r--", linewidth=1.5, label=f"r = {corr:.2f}")
            ax.legend()
        ax.set_xlabel("Episodic Summary Similarity (cosine)", fontsize=11)
        ax.set_ylabel("Behavioral Change Rate", fontsize=11)
        ax.set_title("Figure 3 — Semantic Similarity vs. Behavior Influence\n(Aggregate across all runs)", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/figures/fig3_ae_vs_behavior_change.png", dpi=150)
        plt.close(fig)

    # Fig 4: Proximity Heatmap (Exp 2)
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

    def plot_time_series_agg(title, ylabel, filename, keys, labels, colors):
        if not exp3_details: return
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        ticks = exp3_details[0]["ticks"]
        for key, label, color in zip(keys, labels, colors):
            data = np.array([d[key] for d in exp3_details if len(d[key]) == len(ticks)])
            if data.size == 0: continue
            mu, std = data.mean(axis=0), data.std(axis=0)
            ax.plot(ticks, mu, label=label, color=color, linewidth=2)
            ax.fill_between(ticks, mu-std, mu+std, color=color, alpha=0.2)
        ax.set_xlabel("Tick"); ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(); fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/figures/{filename}", dpi=150); plt.close(fig)

    # Fig 6: Economy (M2, M3)
    plot_time_series_agg("Figure 6 — Simulation Economy over Time\n(Exp 3 | Mean ± std)", 
                        "Value", "fig6_economy_metrics.png",
                        ["ram_efficiency", "territory_control"], ["RAM Efficiency (M2)", "Territory Control (M3)"],
                        ["#4C72B0", "#55A868"])

    # Fig 7: Cognition (M4, M8)
    plot_time_series_agg("Figure 7 — Action Strategy & Divergence\n(Exp 3 | Mean ± std)",
                        "Entropy / Divergence", "fig7_strategy_metrics.png",
                        ["action_entropy", "strategy_divergence"], ["Action Entropy (M4)", "Strategy Divergence (M8)"],
                        ["#C44E52", "#8172B2"])

    # Fig 8: Plasticity (M5, M6)
    plot_time_series_agg("Figure 8 — Graph Complexity & Plasticity\n(Exp 3 | Mean ± std)",
                        "Rate / Mean Degree", "fig8_plasticity_metrics.png",
                        ["graph_complexity", "rewire_rate"], ["Graph Complexity (M5)", "Rewire Rate (M6)"],
                        ["#CCB974", "#64B5CD"])

    print("ALL 8 FIGURES SAVED TO experiment_results/figures/")

if __name__ == "__main__":
    generate_all()
