import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import json, csv, os
from collections import defaultdict

fp = fm.FontProperties(fname=r"C:\Windows\Fonts\msgothic.ttc")
plt.rcParams["font.family"] = fp.get_name()
plt.rcParams["axes.unicode_minus"] = False

LOGDIR = r"\\wsl.localhost\Ubuntu\home\manaka\mkkit2\logs"
RUNS = ["0704_132157", "0704_133600", "0704_133714"]
RUN_LABELS = ["Run 1", "Run 2", "Run 3"]
COLORS = ["#4472C4", "#ED7D31", "#70AD47"]

# Load persona map
with open(os.path.join(LOGDIR, RUNS[0], "actions.json"), encoding="utf-8") as f:
    actions_all = json.load(f)
persona_map = {}
for e in actions_all:
    if e["turn"] == 1:
        persona_map[e["agent_id"]] = e["persona"]

persona_order = ["risk_overestimator", "skeptic", "rational", "information_seeker", "conformist", "agitator"]
persona_color_map = {
    "risk_overestimator": "#C00000",
    "skeptic": "#ED7D31",
    "rational": "#4472C4",
    "information_seeker": "#70AD47",
    "conformist": "#7030A0",
    "agitator": "#FFC000",
}

theoretical = {
    "risk_overestimator": "低 (危険回避重視)",
    "skeptic": "中-低 (懐疑的・慎重)",
    "rational": "中 (エビデンス重視)",
    "information_seeker": "中-高 (情報収集)",
    "conformist": "高 (同調圧力追随)",
    "agitator": "高 (活動的・対立的)",
}

# ===== ①: ペルソナ別意見範囲と理論予測の対応表 (シンプル版) =====
opinion_rows = []
with open(os.path.join(LOGDIR, RUNS[0], "opinion_trajectory.csv"), encoding="utf-8") as f:
    for row in csv.DictReader(f):
        opinion_rows.append(row)

opinion_by_agent = defaultdict(dict)
for row in opinion_rows:
    t = int(row["turn"])
    aid = int(row["agent_id"])
    opinion_by_agent[aid][t] = float(row["opinion"])

persona_agents = defaultdict(list)
for aid, p in persona_map.items():
    persona_agents[p].append(aid)

table_data = []
for p in persona_order:
    agents = persona_agents[p]
    t1_vals = [opinion_by_agent[a][1] for a in agents if 1 in opinion_by_agent[a]]
    t2_vals = [opinion_by_agent[a][2] for a in agents if 2 in opinion_by_agent[a]]
    t3_vals = [opinion_by_agent[a][3] for a in agents if 3 in opinion_by_agent[a]]
    t1_avg = sum(t1_vals)/len(t1_vals) if t1_vals else 0
    t2_avg = sum(t2_vals)/len(t2_vals) if t2_vals else 0
    t3_avg = sum(t3_vals)/len(t3_vals) if t3_vals else 0
    min_v = min(min(t1_vals or [1]), min(t2_vals or [1]), min(t3_vals or [1]))
    max_v = max(max(t1_vals or [0]), max(t2_vals or [0]), max(t3_vals or [0]))
    table_data.append([p, f"{t1_avg:.4f}", f"{t2_avg:.4f}", f"{t3_avg:.4f}",
                       f"{min_v:.4f}～{max_v:.4f}", theoretical[p]])

fig1, ax1 = plt.subplots(figsize=(12, 5))
ax1.axis("off")
col_labels = ["ペルソナ", "Turn 1", "Turn 2", "Turn 3", "意見範囲", "理論予測"]
tbl = ax1.table(cellText=table_data, colLabels=col_labels,
                cellLoc="center", loc="center",
                colWidths=[0.18, 0.10, 0.10, 0.10, 0.16, 0.28])
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", weight="bold", fontproperties=fp)
    else:
        cell.set_text_props(fontproperties=fp)
        if col == 5:
            cell.set_facecolor("#E2EFDA")
        elif col == 4:
            cell.set_facecolor("#D6E4F0")
ax1.set_title("① ペルソナ別意見範囲と理論予測の対応表", fontproperties=fp, fontsize=13, weight="bold", pad=20)
plt.tight_layout()
fig1.savefig(os.path.join(LOGDIR, "01_opinion_range_table.png"), dpi=200, bbox_inches="tight")
plt.close(fig1)

# ===== ②: 投票行動のβ依存性 (黒太線なし) =====
vote_by_run = {}
for r in RUNS:
    with open(os.path.join(LOGDIR, r, "actions.json"), encoding="utf-8") as f:
        data = json.load(f)
    per_agent = {}
    for entry in data:
        if entry["action"] == "VOTE" and entry["votes"] is not None:
            aid = entry["agent_id"]
            yes = sum(1 for v in entry["votes"] if v["vote"] == "yes")
            per_agent[aid] = yes / 8 * 100
    vote_by_run[r] = per_agent

display_order = ["risk_overestimator", "skeptic", "rational", "information_seeker", "conformist", "agitator"]
persona_short = {
    "risk_overestimator": "risk_over\nestimator",
    "skeptic": "skeptic",
    "rational": "rational",
    "information_seeker": "info\nseeker",
    "conformist": "conformist",
    "agitator": "agitator",
}

fig2, ax2 = plt.subplots(figsize=(12, 6))
x_pos = range(len(display_order))
bar_width = 0.25

for i, r in enumerate(RUNS):
    vals = []
    for p in display_order:
        agents = persona_agents[p]
        arr = [vote_by_run[r].get(a, 0) for a in agents]
        vals.append(sum(arr)/len(arr) if arr else 0)
    ax2.bar([j + bar_width*(i-1) for j in x_pos], vals,
            bar_width, label=RUN_LABELS[i], color=COLORS[i], alpha=0.7, zorder=2)
    # Individual scatter points
    for j, p in enumerate(display_order):
        agents = persona_agents[p]
        for a in agents:
            v = vote_by_run[r].get(a, 0)
            ax2.scatter(j + bar_width*(i-1), v,
                        color=COLORS[i], s=40, alpha=0.5, zorder=3)

# Annotations for key personas
key_ann = {"risk_overestimator": "100%", "skeptic": "~50%", "agitator": "~33%"}
for p, label in key_ann.items():
    j = display_order.index(p)
    agents = persona_agents[p]
    all_v = []
    for r in RUNS:
        all_v.extend([vote_by_run[r].get(a, 0) for a in agents])
    avg_v = sum(all_v)/len(all_v) if all_v else 0
    ax2.annotate(label, xy=(j, avg_v), xytext=(0, 15),
                 textcoords="offset points", ha="center", fontsize=11,
                 weight="bold", color=persona_color_map[p], fontproperties=fp,
                 arrowprops=dict(arrowstyle="->", color=persona_color_map[p], lw=1.5))

ax2.set_xticks(x_pos)
ax2.set_xticklabels([persona_short[p] for p in display_order], fontproperties=fp, fontsize=9)
ax2.set_ylabel("賛成率 (%)", fontproperties=fp, fontsize=11)
ax2.set_title("② 投票行動のβ依存性", fontproperties=fp, fontsize=13, weight="bold", pad=15)
ax2.legend(prop=fp, fontsize=8, loc="lower left")
ax2.grid(True, axis="y", alpha=0.3)
ax2.set_ylim(-5, 115)
plt.tight_layout()
fig2.savefig(os.path.join(LOGDIR, "02_beta_dependency.png"), dpi=200, bbox_inches="tight")
plt.close(fig2)

print("Done: 01_opinion_range_table.png, 02_beta_dependency.png")
