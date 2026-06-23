import json

with open('telemetry.jsonl') as f:
    lines = [json.loads(l) for l in f]

metrics = [l for l in lines if l.get('type') == 'metrics']
if metrics:
    print('Metric keys:', list(metrics[0]['data'].keys()))
    print()

first, mid, last = metrics[0], metrics[len(metrics)//2], metrics[-1]
for label, m in [('First', first), ('Mid', mid), ('Last', last)]:
    d = m['data']
    print(f'{label} tick={m["tick"]:>4}: alive={d["survival"]} ram_eff={d["ram_efficiency"]:.3f} entropy={d["action_entropy"]:.3f} territory={d["territory_control"]:.2e}')

arena_lines = [l for l in lines if l.get('type') == 'arena']
print(f'\nTotal arena snapshots: {len(arena_lines)}')
ram_nodes = [a['total_ram_nodes'] for a in arena_lines]
print(f'RAM nodes over time: min={min(ram_nodes)} max={max(ram_nodes)}')
if arena_lines:
    for a in [arena_lines[0], arena_lines[-1]]:
        print(f'  tick={a["tick"]}: radius={a["radius"]:.1f} nodes={a["total_ram_nodes"]}')
