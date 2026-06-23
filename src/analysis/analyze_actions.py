import json
from collections import Counter, defaultdict

def analyze_actions(path="telemetry.jsonl"):
    # Action map from src/simulation/loop.py
    # ACTION_NAMES = ["ATTACK", "MERGE", "DEFEND", "MOVE", "WHISPER", "PROBE", "MANIPULATE"]
    
    agent_actions = defaultdict(Counter)
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("type") == "action":
                    aid = rec["agent_id"]
                    aname = rec["action_name"]
                    agent_actions[aid][aname] += 1
            except:
                continue
                
    # Header
    actions = ["ATTACK", "MERGE", "DEFEND", "MOVE", "WHISPER", "PROBE", "MANIPULATE"]
    header = "| Agent | " + " | ".join(actions) + " |"
    sep = "|---|" + "|---|".join(["" for _ in actions]) + "---|"
    print(header)
    print(sep)
    
    for aid in sorted(agent_actions.keys()):
        counts = agent_actions[aid]
        total = sum(counts.values())
        if total == 0: continue
        
        row = f"| {aid} | "
        cells = []
        for a in actions:
            pct = (counts[a] / total) * 100
            cells.append(f"{pct:.1f}%")
        row += " | ".join(cells) + " |"
        print(row)

if __name__ == "__main__":
    analyze_actions()
