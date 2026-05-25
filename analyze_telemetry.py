import json
from pathlib import Path

all_data = []
for f in sorted(Path("calibration_checkpoints").glob("telemetry_*.jsonl")):
    for line in f.read_text().strip().split("\n"):
        all_data.append(json.loads(line))

print(f"Total telemetry entries: {len(all_data)}")
print(f"Tick range: {all_data[0]['tick']} - {all_data[-1]['tick']}")
print()

# Sample every 10th entry
for i in range(0, len(all_data), 10):
    d = all_data[i]
    print(f"Tick {d['tick']:,}: A0 loss={d['char_pred_loss_a0']:.3f} A1 loss={d['char_pred_loss_a1']:.3f} A0 reward={d['lang_reward_a0']:.2f} A1 reward={d['lang_reward_a1']:.2f}")
