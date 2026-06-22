import json


def _safe_load_jsonl(path: str, expected_type: str = None):
    records = []
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if not isinstance(rec, dict):
                    continue
                if expected_type and rec.get("type") != expected_type:
                    continue
                records.append(rec)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
    return records


telemetry = _safe_load_jsonl('telemetry.jsonl')

print("=== CYNOSURE SIMULATION REPORT ===")
print("Ticks: 2000")
print("Safe Zone: 28.0 -> 10.0 radius (1.0/100 ticks)")
print()

for aid in range(5):
    agent_data = [d for d in telemetry if d.get('agent_id') == aid and d.get('type') == 'action']
    if agent_data:
        last = agent_data[-1]
        print(f"Agent {aid}: final_tick={last['tick']}, ram={last['ram_after']:.1f}, last_action={last['action_name']}")

print("\n=== ACTION DISTRIBUTION ===")
actions = {}
for d in telemetry:
    if d.get('type') == 'action':
        act = d.get('action_name', 'UNKNOWN')
        actions[act] = actions.get(act, 0) + 1
total = sum(actions.values())
for act, cnt in sorted(actions.items(), key=lambda x: -x[1]):
    print(f"  {act}: {cnt} ({cnt/total*100:.1f}%)")

print("\n=== RAM STATS ===")
for aid in range(5):
    ram_series = [d['ram_after'] for d in telemetry if d.get('agent_id') == aid and d.get('type') == 'action']
    if ram_series:
        print(f"Agent {aid}: min={min(ram_series):.1f}, max={max(ram_series):.1f}, final={ram_series[-1]:.1f}")

conflicts = _safe_load_jsonl('conflicts.jsonl')
if conflicts:
    print(f"\n=== CONFLICTS: {len(conflicts)} total ===")

trans = _safe_load_jsonl('transitions.jsonl')
if trans:
    print(f"\n=== ZONE TRANSITIONS ===")
    print(f"Inside->Outside: {len([t for t in trans if 'outside' in t.get('direction','')])}")
    print(f"Outside->Inside: {len([t for t in trans if 'inside' in t.get('direction','')])}")