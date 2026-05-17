import json

# Read telemetry for action distribution
actions = {}

with open('telemetry.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        if 'action_counts' in data:
            for act, cnt in data['action_counts'].items():
                actions[act] = actions.get(act, 0) + cnt

print('=== ACTION DISTRIBUTION (from telemetry) ===')
total = sum(actions.values()) if actions else 0
for act, cnt in sorted(actions.items(), key=lambda x: -x[1]):
    pct = (cnt/total*100) if total > 0 else 0
    print(f'  Action {act}: {cnt} ({pct:.1f}%)')

print()
print('=== SURVIVAL TIMELINE ===')
survival_by_tick = {}
with open('telemetry.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        survival_by_tick[data.get('tick')] = data.get('survival', 0)

ticks = sorted(survival_by_tick.keys())
for tick in ticks[:20]:
    print(f'  Tick {tick}: {survival_by_tick[tick]} alive')

print(f'... to tick {max(ticks)}: {survival_by_tick[max(ticks)]} alive')