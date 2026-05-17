import json

metrics = []
with open('telemetry.jsonl', 'r') as f:
    for line in f:
        try:
            rec = json.loads(line)
            if rec.get('type') == 'metrics':
                metrics.append(rec)
        except:
            pass

if not metrics:
    print('No metrics found.')
else:
    print(f'Found {len(metrics)} metric records.')
    print(f'First tick: {metrics[0]["tick"]}')
    print(f'Last tick: {metrics[-1]["tick"]}')
    
    fields = ['graph_complexity', 'rewire_rate', 'modularity', 'strategy_divergence']
    for field in fields:
        vals = [m['data'].get(field, 0) for m in metrics]
        if any(v != 0 for v in vals):
            print(f'{field}: min={min(vals):.4f}, max={max(vals):.4f}, last={vals[-1]:.4f}')
        else:
            print(f'{field}: All zeros or missing')
