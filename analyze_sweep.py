import json
import os
import numpy as np
from scipy import stats
from collections import defaultdict

def calculate_slope(episodes):
    """Calculate the linear slope of ticks_to_collect over episodes."""
    y = [ep["ticks_to_collect"] for ep in episodes]
    x = np.arange(len(y))
    slope, _, _, _, _ = stats.linregress(x, y)
    return slope

def main():
    input_file = "experiment_results/sweep_metrics.jsonl"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    # Group data by configuration
    config_data = defaultdict(lambda: {"learning": [], "baseline": []})
    
    with open(input_file, "r") as f:
        for line in f:
            rec = json.loads(line)
            # Create a string key for the hyperparams
            hp = rec["hyperparams"]
            key = f"lambda={hp['lambda']}, eta={hp['eta']}, cap={hp['trace_cap']}"
            
            if rec["condition"] == "learning":
                config_data[key]["learning"].append(rec)
            else:
                config_data[key]["baseline"].append(rec)
            
            # Store the raw dict for the winner output
            config_data[key]["params"] = hp

    results = []
    
    print(f"{'Configuration':<40} | {'Slope':<8} | {'p-value':<8} | {'Verdict'}")
    print("-" * 75)

    for key, data in config_data.items():
        learning_recs = data["learning"]
        baseline_recs = data["baseline"]
        
        if not learning_recs or not baseline_recs:
            continue
            
        # 1. Learning Slope
        slopes = [calculate_slope(rec["episodes"]) for rec in learning_recs]
        mean_slope = np.mean(slopes)
        
        # 2. Collection Rates at Episode 20
        # (Assuming episodes are 1-indexed and sorted)
        lr_ep20 = [rec["episodes"][-1]["collected"] for rec in learning_recs]
        bl_ep20 = [rec["episodes"][-1]["collected"] for rec in baseline_recs]
        
        lr_rate = np.mean(lr_ep20)
        bl_rate = np.mean(bl_ep20)
        
        # 3. Statistical Test (Episode 20 collection success)
        # Using t-test on the binary success outcomes at ep 20
        t_stat, p_val = stats.ttest_ind(lr_ep20, bl_ep20)
        
        verdict = "p < 0.05" if p_val < 0.05 else "N.S."
        print(f"{key:<40} | {mean_slope:>8.3f} | {p_val:>8.4f} | {verdict}")
        
        results.append({
            "key": key,
            "params": data["params"],
            "slope": mean_slope,
            "p_value": p_val,
            "lr_rate": lr_rate,
            "bl_rate": bl_rate
        })

    # Isolate optimal configuration
    # Priority: lowest p-value, then steepest negative slope
    if not results:
        print("No valid results found.")
        return

    results.sort(key=lambda x: (x["p_value"], x["slope"]))
    winner = results[0]
    
    print("\n" + "="*40)
    print("WINNING CONFIGURATION")
    print("="*40)
    print(f"Params: {winner['key']}")
    print(f"p-value: {winner['p_value']:.6f}")
    print(f"Learning Slope: {winner['slope']:.4f}")
    print(f"Final Success Rate: {winner['lr_rate']*100:.1f}% vs {winner['bl_rate']*100:.1f}%")

    # Save to production config
    os.makedirs("config", exist_ok=True)
    prod_config_path = "config/production_agent.json"
    with open(prod_config_path, "w") as f:
        json.dump(winner["params"], f, indent=4)
    print(f"\nProduction config saved to {prod_config_path}")

if __name__ == "__main__":
    main()
