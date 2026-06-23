import os
import subprocess
import json
import itertools
import time

def main():
    # Experimental Grid
    lambdas = [0.85, 0.90, 0.95]
    etas = [0.002, 0.005, 0.010]
    trace_caps = [25.0, 50.0]
    
    seeds = 10
    episodes = 10
    neurons = 5000
    
    output_log = "experiment_results/sweep_metrics.jsonl"
    os.makedirs("experiment_results", exist_ok=True)
    
    # Clear existing log
    if os.path.exists(output_log):
        os.remove(output_log)
    
    combinations = list(itertools.product(lambdas, etas, trace_caps))
    total_combos = len(combinations)
    
    print(f"Starting Hyperparameter Sweep: {total_combos} configurations, {seeds} seeds each.")
    
    for i, (l, e, cap) in enumerate(combinations):
        combo_id = f"L{l}_E{e}_C{cap}"
        temp_output = f"experiment_results/temp_{combo_id}.json"
        
        cmd = [
            "python", "run_learning_experiment.py",
            "--seeds", str(seeds),
            "--episodes", str(episodes),
            "--neurons", str(neurons),
            "--lambda_val", str(l),
            "--eta", str(e),
            "--trace_cap", str(cap),
            "--output", temp_output
        ]
        
        print(f"[{i+1}/{total_combos}] Running {combo_id}...")
        t0 = time.time()
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Read temp results and append to master log with metadata
            with open(temp_output, "r") as f:
                records = json.load(f)
                
            with open(output_log, "a") as f:
                for rec in records:
                    rec["hyperparams"] = {
                        "lambda": l,
                        "eta": e,
                        "trace_cap": cap
                    }
                    f.write(json.dumps(rec) + "\n")
            
            os.remove(temp_output)
            print(f"   Done in {time.time() - t0:.1f}s")
            
        except subprocess.CalledProcessError as err:
            print(f"   FAILED {combo_id}: {err.stderr.decode()}")
            continue

if __name__ == "__main__":
    main()
