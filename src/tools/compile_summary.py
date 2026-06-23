import json
from pathlib import Path
from collections import Counter

segments = []
for f in sorted(Path("calibration_checkpoints").glob("segment_*.json")):
    data = json.loads(f.read_text())
    segments.append(data)
    seg_num = data["start_tick"] // 20000
    print(f"Segment {seg_num}: ticks {data['start_tick']:,}-{data['end_tick']:,} | "
          f"A0 loss={data['final_char_pred_loss_a0']:.4f} | A1 loss={data['final_char_pred_loss_a1']:.4f} | "
          f"A0 reward={data['final_lang_reward_a0']:.2f} | A1 reward={data['final_lang_reward_a1']:.2f} | "
          f"A0 words={data['words_a0']} | A1 words={data['words_a1']}")
    if data.get("top_a0"):
        print(f"  Top A0: {data['top_a0'][:3]}")
    if data.get("top_a1"):
        print(f"  Top A1: {data['top_a1'][:3]}")

print()
print("Char pred loss A0:", " -> ".join(f"{s['final_char_pred_loss_a0']:.2f}" for s in segments))
print("Char pred loss A1:", " -> ".join(f"{s['final_char_pred_loss_a1']:.2f}" for s in segments))
print("Lang reward A0:", " -> ".join(f"{s['final_lang_reward_a0']:.2f}" for s in segments))
print("Lang reward A1:", " -> ".join(f"{s['final_lang_reward_a1']:.2f}" for s in segments))
print("Words A0:", " -> ".join(str(s['words_a0']) for s in segments))
print("Words A1:", " -> ".join(str(s['words_a1']) for s in segments))

total_words_a0 = sum(s["words_a0"] for s in segments)
total_words_a1 = sum(s["words_a1"] for s in segments)
print(f"\nTotal words: A0={total_words_a0} | A1={total_words_a1}")
