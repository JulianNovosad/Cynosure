import subprocess
import sys
import time
import json
from pathlib import Path

NUM_SEGMENTS = 5
CHECKPOINT_DIR = "calibration_checkpoints"

print("=" * 60)
print("CYNOSURE — Continuous 100K Calibration (Weight Persistence)")
print(f"  Segments: {NUM_SEGMENTS} x 20,000 ticks = 100,000 total")
print("=" * 60)
print()

t_start = time.time()
all_segments = []

for seg in range(NUM_SEGMENTS):
    print(f"\n{'='*60}")
    print(f"SEGMENT {seg}/{NUM_SEGMENTS-1}: Ticks {seg*20_000:,} - {(seg+1)*20_000:,}")
    print(f"{'='*60}")

    result = subprocess.run(
        [sys.executable, "run_segment.py", str(seg)],
        capture_output=True,
        text=True,
        timeout=600,
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:500])

    seg_file = Path(CHECKPOINT_DIR) / f"segment_{seg*20000:06d}.json"
    if seg_file.exists():
        data = json.loads(seg_file.read_text())
        all_segments.append(data)
        print(f"  A0 loss: {data['final_char_pred_loss_a0']:.4f} -> ", end="")
        print(f"A1 loss: {data['final_char_pred_loss_a1']:.4f}")
        print(f"  A0 reward: {data['final_lang_reward_a0']:.2f} | A1 reward: {data['final_lang_reward_a1']:.2f}")
        print(f"  A0 words: {data['words_a0']} | A1 words: {data['words_a1']}")
    else:
        print(f"  WARNING: No segment file found for segment {seg}")

    elapsed = time.time() - t_start
    print(f"\n  Total elapsed: {elapsed:.0f}s")

print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")
print(f"Total time: {time.time() - t_start:.0f}s")
print()

if all_segments:
    print("Char pred loss A0:", " -> ".join(f"{s['final_char_pred_loss_a0']:.2f}" for s in all_segments))
    print("Char pred loss A1:", " -> ".join(f"{s['final_char_pred_loss_a1']:.2f}" for s in all_segments))
    print("Lang reward A0:", " -> ".join(f"{s['final_lang_reward_a0']:.2f}" for s in all_segments))
    print("Lang reward A1:", " -> ".join(f"{s['final_lang_reward_a1']:.2f}" for s in all_segments))
    print("Words A0:", " -> ".join(str(s['words_a0']) for s in all_segments))
    print("Words A1:", " -> ".join(str(s['words_a1']) for s in all_segments))

    total_words_a0 = sum(s["words_a0"] for s in all_segments)
    total_words_a1 = sum(s["words_a1"] for s in all_segments)
    print(f"\nTotal words: A0={total_words_a0} | A1={total_words_a1}")

    print("\nTOP WORDS ACROSS ALL SEGMENTS:")
    from collections import Counter
    all_words_a0 = []
    all_words_a1 = []
    for s in all_segments:
        for w, c in s.get("top_a0", []):
            all_words_a0.extend([w] * c)
        for w, c in s.get("top_a1", []):
            all_words_a1.extend([w] * c)

    if all_words_a0:
        print("  A0:", Counter(all_words_a0).most_common(10))
    if all_words_a1:
        print("  A1:", Counter(all_words_a1).most_common(10))
