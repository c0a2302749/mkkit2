import csv, json, os, hashlib

LOGDIR = r"\\wsl.localhost\Ubuntu\home\manaka\mkkit2\logs"
RUNS = ["0704_132157", "0704_133600", "0704_133714"]

hashes = {}
for r in RUNS:
    path = os.path.join(LOGDIR, r, "opinion_trajectory.csv")
    with open(path, "rb") as f:
        hashes[r] = hashlib.md5(f.read()).hexdigest()
print("Opinion CSV hashes:", hashes)

persona_map = {}
with open(os.path.join(LOGDIR, RUNS[0], "actions.json"), encoding="utf-8") as f:
    data = json.load(f)
for entry in data:
    if entry["turn"] == 1:
        persona_map[entry["agent_id"]] = entry["persona"]

for r in RUNS:
    with open(os.path.join(LOGDIR, r, "actions.json"), encoding="utf-8") as f:
        data = json.load(f)
    print(f"\n--- {r} ---")
    for entry in data:
        if entry["action"] == "VOTE" and entry["votes"] is not None:
            aid = entry["agent_id"]
            p = persona_map[aid]
            yes = sum(1 for v in entry["votes"] if v["vote"] == "yes")
            print(f"  agent {aid} ({p}): {yes}/8 YES")
