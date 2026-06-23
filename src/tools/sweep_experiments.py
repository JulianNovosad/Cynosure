import os
import subprocess
import json

def run_experiment(lambda_val, eta_val, output_file):
    cmd = [
        "python", "run_learning_experiment.py",
        "--seeds", "10",
        "--episodes", "10",
        "--lambda_val", str(lambda_val),
        "--eta", str(eta_val),
        "--output", output_file
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def analyze_results(output_file):
    cmd = ["python", "analyze_learning_experiment.py", output_file]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout

def main():
    lambdas = [0.85, 0.90, 0.95]
    etas = [0.002, 0.005, 0.01]
    
    results = []
    
    for l in lambdas:
        for e in etas:
            out_file = f"experiment_results/sweep_l{l}_e{e}.json"
            run_experiment(l, e, out_file)
            analysis = analyze_results(out_file)
            
            # Extract p-value and verdict from analysis output
            p_val = 1.0
            verdict = "N/A"
            for line in analysis.split("\n"):
                if "p =" in line:
                    try:
                        p_val = float(line.split("p =")[-1].strip())
                    except: pass
                if "VERDICT" in line:
                    verdict = line.strip()
            
            results.append({
                "lambda": l,
                "eta": e,
                "p_value": p_val,
                "verdict": verdict,
                "file": out_file
            })
            
    print("\n" + "="*40)
    print("SWEEP RESULTS")
    print("="*40)
    for res in sorted(results, key=lambda x: x["p_value"]):
        print(f"L:{res['lambda']} E:{res['eta']} | p={res['p_value']:.4f} | {res['verdict']}")

if __name__ == "__main__":
    main()
