import json
from collections import Counter

counts = Counter()
with open("telemetry.jsonl", "r") as f:
    for line in f:
        try:
            data = json.loads(line)
            if data.get("type") == "action":
                counts[data.get("action_name")] += 1
        except:
            continue

print(json.dumps(counts, indent=2))
