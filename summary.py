import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('telemetry.jsonl') as f:
    lines = [json.loads(l) for l in f]
metrics = [l for l in lines if l.get('type') == 'metrics']

print(f'=== TELEMETRY: {len(metrics)} records ===')
first, mid, last = metrics[0], metrics[len(metrics)//2], metrics[-1]
for label, m in [('Start', first), ('Mid', mid), ('End', last)]:
    d = m['data']
    print(f'{label} tick={m["tick"]:>4}: alive={d["survival"]} territory={d["territory_control"]:.2f} ram_eff={d["ram_efficiency"]:.4f} entropy={d["action_entropy"]:.3f} strat_div={d.get("strategy_divergence",0):.3f}')

try:
    with open('transitions.jsonl') as f:
        tx = [json.loads(l) for l in f]
    print(f'\n=== TRANSITIONS: {len(tx)} events ===')
    for t in tx[:5]:
        ctx_preview = t["context_buffer"][:4]
        print(f'  Tick {t["tick"]} AI{t["agent_id"]}: dir={t["direction"]} reward={t["intrinsic_reward"]} theta={t["theta"]} ctx={[round(v,3) for v in ctx_preview]}')
except FileNotFoundError:
    print('\nNo transitions file')

security_escapes = 0
try:
    with open('security_audit.log') as f:
        for line in f:
            if 'EscapeAttempt' in line:
                security_escapes += 1
except FileNotFoundError:
    pass
print(f'\n=== SECURITY ===')
print(f'  Escape attempt logs: {security_escapes}')

# Final state
d = metrics[-1]["data"]
print(f'\n=== FINAL STATE (tick={metrics[-1]["tick"]}) ===')
for k, v in d.items():
    print(f'  {k}: {v}')
