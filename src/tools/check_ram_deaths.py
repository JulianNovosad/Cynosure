import json

max_ram = 0.0
first_death_tick = None

with open("telemetry.jsonl", "r") as f:
    for line in f:
        try:
            data = json.loads(line)
            if data.get("type") == "action":
                ram = data.get("ram_after", 0)
                if ram > max_ram:
                    max_ram = ram
            
            # Check for deaths in metrics
            if data.get("tick") and "survival" in data:
                 # telemetry.jsonl metrics might not show which agent died
                 pass
        except:
            continue

# Find first death from agent whispers or security audit
deaths = []
with open("security_audit.log", "r") as f:
    for line in f:
        if "EVENT=PredationReward" in line:
            import re
            match = re.search(r"TICK=(\d+)", line)
            if match:
                deaths.append(int(match.group(1)))

print(f"Max RAM: {max_ram}")
print(f"Deaths found in audit: {deaths}")
if deaths:
    print(f"First death tick: {min(deaths)}")
else:
    print("No deaths found in audit log.")
