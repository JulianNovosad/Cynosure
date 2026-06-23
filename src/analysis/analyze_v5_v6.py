#!/usr/bin/env python3
"""analyze_v5_v6.py -- Statistical analysis of LRSG learning experiments.

Reads learning_experiment_v5.json (old algorithm) and/or
learning_experiment_v6.json (positive-only fix) and prints:
  - Per-seed collection rates (early vs late half)
  - Paired t-test on (L_improvement - B_improvement) slope
  - Cohen's d effect size
  - Verdict
"""
import json, math, os, sys, statistics

def load_jsonl(path):
    records = []
    with open(path) as f:
        raw = f.read()
    # Try JSON array first
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    # Fall back to JSONL
    for line in raw.splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records

def analyze(path, label):
    if not os.path.exists(path):
        print(f"{label}: FILE NOT FOUND ({path})")
        return
    records = load_jsonl(path)
    if not records:
        print(f"{label}: NO RECORDS YET")
        return

    # Group by seed and condition
    seeds = {}
    for rec in records:
        s = rec["seed"]
        c = rec["condition"]
        if s not in seeds:
            seeds[s] = {}
        if c in seeds[s]:
            # duplicate — skip (take first)
            continue
        seeds[s][c] = [e["collected"] for e in rec["episodes"]]

    complete = {s: d for s, d in seeds.items() if "learning" in d and "baseline" in d}
    n = len(complete)
    if n == 0:
        print(f"{label}: no complete seed pairs yet")
        return

    print(f"\n{'='*60}")
    print(f"{label}  ({path})")
    print(f"{'='*60}")
    print(f"Complete seed pairs: {n}")
    print(f"{'Seed':>4} | {'L early':>7} | {'L late':>7} | {'L diff':>7} | {'B early':>7} | {'B late':>7} | {'B diff':>7}")
    L_diffs, B_diffs = [], []
    for s in sorted(complete):
        L = complete[s]["learning"]
        B = complete[s]["baseline"]
        half = len(L) // 2
        le = sum(L[:half]) / half * 100
        ll = sum(L[half:]) / half * 100
        be = sum(B[:half]) / half * 100
        bl = sum(B[half:]) / half * 100
        ld, bd = ll - le, bl - be
        L_diffs.append(ld)
        B_diffs.append(bd)
        print(f"{s:>4} | {le:>6.1f}% | {ll:>6.1f}% | {ld:>+6.1f}% | {be:>6.1f}% | {bl:>6.1f}% | {bd:>+6.1f}%")

    lm = statistics.mean(L_diffs)
    bm = statistics.mean(B_diffs)
    print(f"\nLearning  mean improvement: {lm:+.1f}% (std={statistics.stdev(L_diffs):.1f}%)" if n > 1 else f"\nLearning  improvement: {lm:+.1f}%")
    print(f"Baseline  mean improvement: {bm:+.1f}% (std={statistics.stdev(B_diffs):.1f}%)" if n > 1 else f"Baseline  improvement: {bm:+.1f}%")

    if n >= 2:
        diffs = [L_diffs[i] - B_diffs[i] for i in range(n)]
        mean_d = statistics.mean(diffs)
        std_d = statistics.stdev(diffs)
        if std_d < 1e-9:
            print("std=0, cannot compute t")
            return
        t = mean_d / (std_d / math.sqrt(n))
        cohens_d = mean_d / std_d
        # approximate p-value (one-tailed, df=n-1) using t-distribution table
        # For quick verdict, use critical values
        print(f"\nPaired t (L_diff - B_diff): t = {t:.3f}, n = {n}, Cohen's d = {cohens_d:.3f}")
        if t > 2.861 and n >= 10:
            p_str = "< 0.01"
        elif t > 1.833 and n >= 10:
            p_str = "< 0.05"
        elif t > 2.776 and n >= 5:
            p_str = "< 0.025 (df=4)"
        elif t > 2.132 and n >= 5:
            p_str = "< 0.05 (df=4)"
        elif t > 1.533 and n >= 5:
            p_str = "< 0.10 (df=4)"
        else:
            p_str = "ns (> 0.10)"
        print(f"One-tailed p ~ {p_str}")

        if t > 1.533:
            print("\n[PASS] LEARNING SIGNAL DETECTED -- LRSG shows improvement over baseline.")
        else:
            print("\n[FAIL] NOT SIGNIFICANT -- insufficient evidence of learning.")

    # Overall collection rates
    print(f"\nOverall: L={sum(1 for s in complete for e in complete[s]['learning'] if e)}/{sum(len(complete[s]['learning']) for s in complete)} " +
          f"B={sum(1 for s in complete for e in complete[s]['baseline'] if e)}/{sum(len(complete[s]['baseline']) for s in complete)}")


if __name__ == "__main__":
    analyze("experiment_results/learning_experiment_v5.json", "v5 (old algorithm, no fix)")
    analyze("experiment_results/learning_experiment_v6.json", "v6 (positive-only fix)")
