import json
from pathlib import Path

for f in sorted(Path("calibration_checkpoints").glob("telemetry_*.jsonl")):
    data = [json.loads(l) for l in f.read_text().strip().split("\n")]
    print(f"{f.name}: {len(data)} entries")
    print(f"  A0 reward: {data[0]['lang_reward_a0']:.2f} -> {data[-1]['lang_reward_a0']:.2f}")
    print(f"  A1 reward: {data[0]['lang_reward_a1']:.2f} -> {data[-1]['lang_reward_a1']:.2f}")

    # Count how many entries have non-zero reward
    a0_nonzero = sum(1 for d in data if d["lang_reward_a0"] > 0)
    a1_nonzero = sum(1 for d in data if d["lang_reward_a1"] > 0)
    print(f"  A0 non-zero reward entries: {a0_nonzero}/{len(data)}")
    print(f"  A1 non-zero reward entries: {a1_nonzero}/{len(data)}")
