#!/usr/bin/env python3
"""analyze_learning_experiment.py

Analyzes results from run_learning_experiment.py.
Reads experiment_results/learning_experiment.json and prints:

1. Collection rate by episode (learning vs baseline)
2. Ticks to collect (mean ± SD per episode, collected only)
3. Learning trend (slope of ticks_to_collect over episodes)
4. Two-sample t-test on episode 10 collection rate
5. Verdict
"""

import json
import math
import os
import sys

import numpy as np
import math


def ttest_ind(a, b):
    """Two-sample independent t-test using normal approximation (valid for large n)."""
    n1, n2 = len(a), len(b)
    m1, m2 = np.mean(a), np.mean(b)
    v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
    se = math.sqrt(v1 / n1 + v2 / n2)
    if se < 1e-12:
        return 0.0, 1.0
    t = (m1 - m2) / se
    from math import erfc
    p = erfc(abs(t) / math.sqrt(2))
    return t, float(p)


def load_data(path="experiment_results/learning_experiment.json"):
    with open(path) as f:
        raw = json.load(f)
    seeds = {}
    for rec in raw:
        s = rec["seed"]
        if s not in seeds:
            seeds[s] = {}
        seeds[s][rec["condition"]] = rec["episodes"]
    return seeds


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "experiment_results", "learning_experiment.json"
        )
    if not os.path.exists(path):
        print(f"Error: {path} not found. Run run_learning_experiment.py first.")
        sys.exit(1)

    seeds = load_data(path)
    n_seeds = len(seeds)
    print(f"Loaded {n_seeds} seeds\n")

    learning_ep = [[] for _ in range(10)]
    baseline_ep = [[] for _ in range(10)]

    for s, conds in seeds.items():
        for ep_idx in range(10):
            if "learning" in conds:
                learning_ep[ep_idx].append(conds["learning"][ep_idx])
            if "baseline" in conds:
                baseline_ep[ep_idx].append(conds["baseline"][ep_idx])

    # 1. Collection rate by episode
    print("=" * 60)
    print("1. COLLECTION RATE BY EPISODE")
    print("=" * 60)
    print(f"{'Ep':>4} {'Learning':>12} {'Baseline':>12} {'Delta':>12}")
    print("-" * 44)
    lr_rates = []
    bl_rates = []
    for ep_idx in range(10):
        lr_n = len(learning_ep[ep_idx])
        bl_n = len(baseline_ep[ep_idx])
        lr_coll = sum(1 for e in learning_ep[ep_idx] if e["collected"])
        bl_coll = sum(1 for e in baseline_ep[ep_idx] if e["collected"])
        lr_rate = lr_coll / lr_n * 100 if lr_n > 0 else 0
        bl_rate = bl_coll / bl_n * 100 if bl_n > 0 else 0
        lr_rates.append(lr_rate)
        bl_rates.append(bl_rate)
        print(f"{ep_idx+1:4d} {lr_rate:>8.1f}%  {bl_rate:>8.1f}%  {lr_rate-bl_rate:>+9.1f}%")

    # 2. Ticks to collect
    print("\n" + "=" * 60)
    print("2. TICKS TO COLLECT (mean ± SD, collected only)")
    print("=" * 60)
    print(f"{'Ep':>4} {'Learning':>30} {'Baseline':>30}")
    print("-" * 68)
    for ep_idx in range(10):
        lr_ticks = [e["ticks_to_collect"] for e in learning_ep[ep_idx] if e["collected"]]
        bl_ticks = [e["ticks_to_collect"] for e in baseline_ep[ep_idx] if e["collected"]]
        lr_str = f"n={len(lr_ticks)}"
        bl_str = f"n={len(bl_ticks)}"
        if lr_ticks:
            lr_str += f"  {np.mean(lr_ticks):.1f} ± {np.std(lr_ticks):.1f}"
        if bl_ticks:
            bl_str += f"  {np.mean(bl_ticks):.1f} ± {np.std(bl_ticks):.1f}"
        print(f"{ep_idx+1:4d} {lr_str:>30} {bl_str:>30}")

    # 3. Learning trend
    print("\n" + "=" * 60)
    print("3. LEARNING TREND (slope of ticks_to_collect over episodes 1-10)")
    print("=" * 60)
    for condition, ep_data, label in [
        ("learning", learning_ep, "Learning"),
        ("baseline", baseline_ep, "Baseline"),
    ]:
        slopes = []
        for s, conds in seeds.items():
            if condition not in conds:
                continue
            ticks = [e["ticks_to_collect"] for e in conds[condition]]
            x = np.arange(10)
            slope, _ = np.polyfit(x, ticks, 1)
            slopes.append(slope)
        mean_slope = np.mean(slopes)
        sd_slope = np.std(slopes)
        n = len(slopes)
        print(f"  {label}: mean slope = {mean_slope:.2f} ± {sd_slope:.2f} (n={n})")

    # 4. Statistical test
    print("\n" + "=" * 60)
    print("4. STATISTICAL TEST (episode 10 collection rate)")
    print("=" * 60)
    lr_ep10 = [1 if conds.get("learning", [{}]*10)[9].get("collected", False) else 0
               for s, conds in seeds.items()]
    bl_ep10 = [1 if conds.get("baseline", [{}]*10)[9].get("collected", False) else 0
               for s, conds in seeds.items()]

    # Filter out seeds missing a condition
    valid = [i for i in range(len(seeds)) if i < len(list(seeds.keys()))]
    lr_ep10 = [1 if seeds[s].get("learning", [{}]*10)[9].get("collected", False) else 0
               for s in sorted(seeds.keys())]
    bl_ep10 = [1 if seeds[s].get("baseline", [{}]*10)[9].get("collected", False) else 0
               for s in sorted(seeds.keys())]

    n_lr = len(lr_ep10)
    n_bl = len(bl_ep10)
    lr_rate_ep10 = sum(lr_ep10) / n_lr * 100 if n_lr > 0 else 0
    bl_rate_ep10 = sum(bl_ep10) / n_bl * 100 if n_bl > 0 else 0
    print(f"  Learning episode 10 collection rate: {sum(lr_ep10)}/{n_lr} = {lr_rate_ep10:.1f}%")
    print(f"  Baseline episode 10 collection rate: {sum(bl_ep10)}/{n_bl} = {bl_rate_ep10:.1f}%")

    t_stat, p_value = ttest_ind(lr_ep10, bl_ep10)
    print(f"  Two-sample t-test: t = {t_stat:.4f}, p = {p_value:.6f}")
    print(f"  Significant at p < 0.05: {'YES' if p_value < 0.05 else 'NO'}")

    # 5. Verdict
    print("\n" + "=" * 60)
    print("5. VERDICT")
    print("=" * 60)
    if p_value < 0.05 and lr_rate_ep10 > bl_rate_ep10:
        verdict = (
            f"LEARNING CONFIRMED: LRSG shows statistically significant improvement "
            f"over random walk baseline (p={p_value:.4f}). Proceed with transfer experiment."
        )
    elif p_value < 0.15 and lr_rate_ep10 > bl_rate_ep10:
        verdict = (
            f"MARGINAL SIGNAL: LRSG shows non-significant trend toward improvement "
            f"(p={p_value:.4f}). Investigate learning rate and reward scale before proceeding."
        )
    else:
        verdict = (
            f"LEARNING FAILURE: LRSG performs at or below random walk baseline "
            f"(p={p_value:.4f}). The Hebbian learning mechanism requires redesign "
            f"before further development."
        )
    print(f"  {verdict}")

    return {
        "lr_rates": lr_rates,
        "bl_rates": bl_rates,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "verdict": verdict,
        "lr_rate_ep10": lr_rate_ep10,
        "bl_rate_ep10": bl_rate_ep10,
        "n_seeds": n_seeds,
    }


if __name__ == "__main__":
    result = main()
