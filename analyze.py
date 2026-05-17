import json

entries = []
with open('telemetry.jsonl', 'r') as f:
    for line in f:
        entries.append(json.loads(line))

print('=== LAST 5 TELEMETRY ENTRIES ===')
for e in entries[-5:]:
    print(f"Tick {e.get('tick')}: survival={e.get('survival')}, action_entropy={e.get('action_entropy')}")

print()
print('=== SURVIVAL TIMELINE ===')
survival_by_tick = {}
for e in entries:
    survival_by_tick[e.get('tick')] = e.get('survival', 0)

ticks = sorted(survival_by_tick.keys())
print(f'First tick: {min(ticks)}, survival: {survival_by_tick[min(ticks)]}')
print(f'Last tick: {max(ticks)}, survival: {survival_by_tick[max(ticks)]}')

# Find when deaths happened
prev = None
death_ticks = []
for tick in sorted(survival_by_tick.keys()):
    cur = survival_by_tick[tick]
    if prev is not None and cur < prev:
        death_ticks.append((tick, prev, cur))
    prev = cur

print(f'Death events: {death_ticks}')