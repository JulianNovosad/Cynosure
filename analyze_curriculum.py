import json
import os
from collections import defaultdict
import sys

# Ensure src is in path
sys.path.append(os.getcwd())

from src.lrsg.io import validate_curriculum

def analyze():
    whisper_file = "agent_whispers.jsonl"
    if not os.path.exists(whisper_file):
        print("Whisper log not found.")
        return

    stats = {
        "Stage 1 (0-1000)": {"hits": 0, "total": 0},
        "Stage 2 (1000-3000)": {"hits": 0, "total": 0},
        "Stage 3 (3000-5000)": {"hits": 0, "total": 0},
    }
    
    samples = defaultdict(list)
    target_ticks = [1000, 3000, 5000]
    
    with open(whisper_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
            except:
                continue
            tick = data.get("tick", 0)
            text = data.get("whisper", "")
            
            stage_reached = validate_curriculum(text, tick)
            
            if tick < 1000:
                stats["Stage 1 (0-1000)"]["total"] += 1
                if stage_reached >= 1:
                    stats["Stage 1 (0-1000)"]["hits"] += 1
            elif tick < 3000:
                stats["Stage 2 (1000-3000)"]["total"] += 1
                if stage_reached >= 2:
                    stats["Stage 2 (1000-3000)"]["hits"] += 1
            elif tick <= 5000:
                stats["Stage 3 (3000-5000)"]["total"] += 1
                if stage_reached >= 3:
                    stats["Stage 3 (3000-5000)"]["hits"] += 1
                    
            for target in target_ticks:
                if abs(tick - target) < 5: # Capture whispers near target ticks
                    samples[target].append(text)

    print("\n=== LINGUISTIC CURRICULUM REPORT ===")
    for stage, data in stats.items():
        pct = (data["hits"] / data["total"] * 100) if data["total"] > 0 else 0
        print(f"{stage}: {pct:.2f}% hits ({data['hits']}/{data['total']})")
        
    print("\n=== SAMPLE WHISPERS ===")
    for tick in sorted(samples.keys()):
        print(f"\nAround Tick {tick}:")
        unique_samples = list(set(samples[tick]))
        for s in unique_samples[:5]: # Show up to 5 unique
            print(f"  - {s}")

if __name__ == "__main__":
    analyze()
