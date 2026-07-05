import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import json, os, csv
import numpy as np
from collections import defaultdict

fp = fm.FontProperties(fname=r"C:\Windows\Fonts\msgothic.ttc")
plt.rcParams["font.family"] = fp.get_name()
plt.rcParams["axes.unicode_minus"] = False

LOGDIR = r"\\wsl.localhost\Ubuntu\home\manaka\mkkit2\logs"

EXPERIMENTS = {
    "A (LLM確率性)": {
        "runs": ["0704_163755", "0704_163902", "0704_164135"],
        "label": "seed=42固定, LLMのみ変動",
        "color": "#4472C4",
    },
    "B (Seed変動)": {
        "runs": ["0704_164135", "0704_164304", "0704_164424"],
        "label": "community固定, seed=42/43/44",
        "color": "#ED7D31",
    },
    "C (Topology比較)": {
        "runs": ["0704_165546", "0704_170013", "0704_170316", "0704_170555"],
        "label": "seed=42固定, topology変更",
        "color": "#70AD47",
    },
}

def load_trajectory(run):
    rows = []
    with open(os.path.join(LOGDIR, run, "opinion_trajectory.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "turn": int(row["turn"]),
                "agent_id": int(row["agent_id"]),
                "persona": row["persona"],
                "opinion": float(row["opinion"]),
                "risk": float(row["risk_perception"]),
                "social_opinion": float(row["social_opinion"]),
                "social_risk": float(row["social_risk"]),
            })
    return rows

def compute_metrics(rows, max_turn=3):
    turns = sorted(set(r["turn"] for r in rows if r["turn"] <= max_turn))
    n_agents = len(set(r["agent_id"] for r in rows))
    metrics = {}
    for t in turns:
        trows = [r for r in rows if r["turn"] == t]
        opinions = np.array([r["opinion"] for r in trows])
        s_opinions = np.array([r["social_opinion"] for r in trows])
        divergence = np.abs(opinions - s_opinions)
        extreme = np.mean((opinions <= 0.2) | (opinions >= 0.8))
        corr = np.corrcoef(opinions, s_opinions)[0, 1] if len(trows) > 2 else 0
        bimo = np.mean((opinions <= 0.3) | (opinions >= 0.7))
        metrics[t] = {
            "mean_divergence": float(np.mean(divergence)),
            "std_divergence": float(np.std(divergence)),
            "mean_opinion": float(np.mean(opinions)),
            "std_opinion": float(np.std(opinions)),
            "extreme_ratio": float(extreme),
            "bimodal_ratio": float(bimo),
            "opinion_social_corr": float(corr),
            "n_converged": int(np.sum(divergence < divergence[0])),
            "n_diverged": int(np.sum(divergence > divergence[0])),
        }
    return metrics

def get_topology(run):
    s = json.load(open(os.path.join(LOGDIR, run, "summary.json"), encoding="utf-8"))
    return s["config"]["topology"]

def get_seed(run):
    s = json.load(open(os.path.join(LOGDIR, run, "summary.json"), encoding="utf-8"))
    return s["config"]["seed"]

# ─── Collect all results ───
all_results = {}
for exp_name, exp in EXPERIMENTS.items():
    exp_data = []
    for r in exp["runs"]:
        topo = get_topology(r)
        seed = get_seed(r)
        rows = load_trajectory(r)
        met = compute_metrics(rows)
        exp_data.append({"run": r, "topology": topo, "seed": seed, "metrics": met})
    all_results[exp_name] = exp_data

# ===== 1. Echo Chamber Comparison Table =====
print("=" * 100)
print("{:^100}".format("エコーチェンバー・極性化指標 比較"))
print("=" * 100)

header = f"{'実験':<16} {'Run':<14} {'Topo':<12} {'Seed':<6} {'Turn':<5} {'Diverg':<8} {'O-S_O r':<8} {'Std(O)':<8} {'Extreme':<8} {'Bimodal':<8}"
print(header)
print("-" * 100)

for exp_name, exp_data in all_results.items():
    for ed in exp_data:
        for t in sorted(ed["metrics"].keys()):
            m = ed["metrics"][t]
            print(f"{exp_name:<16} {ed['run']:<14} {ed['topology']:<12} {ed['seed']:<6} T{t:<3} {m['mean_divergence']:.4f}     {m['opinion_social_corr']:.4f}  {m['std_opinion']:.4f}   {m['extreme_ratio']:.4f}   {m['bimodal_ratio']:.4f}")
        print("-" * 100)

# Summary per experiment (Turn 3 only)
print("\n")
print("=" * 100)
print("{:^100}".format("実験別 平均 (Turn 3)"))
print("=" * 100)
print(f"{'実験':<20} {'平均Diverg':<12} {'平均r(O,S_O)':<14} {'平均Std(O)':<12} {'平均Extreme':<12} {'Diverg率変動':<12}")
print("-" * 100)

for exp_name, exp_data in all_results.items():
    t3 = [ed["metrics"][3] for ed in exp_data if 3 in ed["metrics"]]
    avg_div = np.mean([m["mean_divergence"] for m in t3])
    avg_corr = np.mean([m["opinion_social_corr"] for m in t3])
    avg_std = np.mean([m["std_opinion"] for m in t3])
    avg_ext = np.mean([m["extreme_ratio"] for m in t3])
    # Divergence change T1→T3
    t1 = [ed["metrics"][1] for ed in exp_data if 1 in ed["metrics"]]
    t3 = [ed["metrics"][3] for ed in exp_data if 3 in ed["metrics"]]
    div_changes = []
    for ed1, ed3 in zip(t1, t3):
        div_changes.append(float(ed3["mean_divergence"] - ed1["mean_divergence"]))
    avg_change = np.mean(div_changes)
    print(f"{exp_name:<20} {avg_div:<12.4f} {avg_corr:<14.4f} {avg_std:<12.4f} {avg_ext:<12.4f} {avg_change:<+12.4f}")

# ===== 2. Polarization check: histogram bins =====
print("\n")
print("=" * 100)
print("{:^100}".format("最終意見分布 (Turn 3) - 極性化チェック"))
print("=" * 100)

bin_labels = ["0.0-0.2(反対)", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0(賛成)"]
for exp_name, exp_data in all_results.items():
    print(f"\n--- {exp_name} ---")
    for ed in exp_data:
        rows = load_trajectory(ed["run"])
        t3_ops = [r["opinion"] for r in rows if r["turn"] == 3]
        bins = np.histogram(t3_ops, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0])[0]
        parts = [f"{ed['run']} (topo={ed['topology']}, seed={ed['seed']}):"]
        parts.extend(f"{b}/{len(t3_ops)}" for b in bins)
        print("  " + " | ".join(parts))

# ===== 3. Per-persona divergence analysis =====
print("\n")
print("=" * 100)
print("{:^100}".format("ペルソナ別 Turn3 意見乖離 |O-S_O|"))
print("=" * 100)

# Baseline run for comparison across experiments
baseline_run = "0704_164135"
persona_map = {}
d0 = json.load(open(os.path.join(LOGDIR, baseline_run, "actions.json"), encoding="utf-8"))
for e in d0:
    if e["turn"] == 1:
        persona_map[e["agent_id"]] = e["persona"]

for exp_name, exp_data in all_results.items():
    print(f"\n--- {exp_name} ---")
    header = "  Persona".ljust(22) + "".join(ed["run"][-4:] + "  " for ed in exp_data)
    print(header)
    for aid in sorted(persona_map.keys()):
        vals = []
        for ed in exp_data:
            rows = load_trajectory(ed["run"])
            t3 = [r for r in rows if r["turn"] == 3 and r["agent_id"] == aid]
            if t3:
                vals.append(abs(t3[0]["opinion"] - t3[0]["social_opinion"]))
        line = f"  ag{aid} {persona_map[aid]:<18}" + "".join(f"{v:.4f}    " for v in vals)
        print(line)

# ===== 4. Generate plots =====
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 4a. Mean opinion divergence trajectories
ax = axes[0, 0]
for exp_name, exp_data in all_results.items():
    c = EXPERIMENTS[exp_name]["color"]
    turns = sorted(exp_data[0]["metrics"].keys())
    for ed in exp_data:
        vals = [ed["metrics"][t]["mean_divergence"] for t in turns]
        ax.plot(turns, vals, "o-", color=c, alpha=0.3, linewidth=1)
    avg = [np.mean([ed["metrics"][t]["mean_divergence"] for ed in exp_data]) for t in turns]
    ax.plot(turns, avg, "o-", color=c, linewidth=3, label=exp_name)
ax.set_xlabel("Turn", fontproperties=fp)
ax.set_ylabel("平均 |O - S_O|", fontproperties=fp)
ax.set_title("(a) 意見乖離の時系列", fontproperties=fp, weight="bold")
ax.legend(prop=fp, fontsize=8)
ax.grid(True, alpha=0.3)

# 4b. Extreme ratio trajectories
ax = axes[0, 1]
for exp_name, exp_data in all_results.items():
    c = EXPERIMENTS[exp_name]["color"]
    turns = sorted(exp_data[0]["metrics"].keys())
    for ed in exp_data:
        vals = [ed["metrics"][t]["extreme_ratio"] for t in turns]
        ax.plot(turns, vals, "o-", color=c, alpha=0.3, linewidth=1)
    avg = [np.mean([ed["metrics"][t]["extreme_ratio"] for ed in exp_data]) for t in turns]
    ax.plot(turns, avg, "o-", color=c, linewidth=3, label=exp_name)
ax.set_xlabel("Turn", fontproperties=fp)
ax.set_ylabel("Extreme Ratio (<0.2 or >0.8)", fontproperties=fp)
ax.set_title("(b) 極性化比率", fontproperties=fp, weight="bold")
ax.legend(prop=fp, fontsize=8)
ax.grid(True, alpha=0.3)

# 4c. Opinion std trajectories
ax = axes[0, 2]
for exp_name, exp_data in all_results.items():
    c = EXPERIMENTS[exp_name]["color"]
    turns = sorted(exp_data[0]["metrics"].keys())
    for ed in exp_data:
        vals = [ed["metrics"][t]["std_opinion"] for t in turns]
        ax.plot(turns, vals, "o-", color=c, alpha=0.3, linewidth=1)
    avg = [np.mean([ed["metrics"][t]["std_opinion"] for ed in exp_data]) for t in turns]
    ax.plot(turns, avg, "o-", color=c, linewidth=3, label=exp_name)
ax.set_xlabel("Turn", fontproperties=fp)
ax.set_ylabel("意見の標準偏差", fontproperties=fp)
ax.set_title("(c) 意見分散", fontproperties=fp, weight="bold")
ax.legend(prop=fp, fontsize=8)
ax.grid(True, alpha=0.3)

# 4d. Opinion-Social correlation (echo chamber)
ax = axes[1, 0]
for exp_name, exp_data in all_results.items():
    c = EXPERIMENTS[exp_name]["color"]
    turns = sorted(exp_data[0]["metrics"].keys())
    for ed in exp_data:
        vals = [ed["metrics"][t]["opinion_social_corr"] for t in turns]
        ax.plot(turns, vals, "o-", color=c, alpha=0.3, linewidth=1)
    avg = [np.mean([ed["metrics"][t]["opinion_social_corr"] for ed in exp_data]) for t in turns]
    ax.plot(turns, avg, "o-", color=c, linewidth=3, label=exp_name)
ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Turn", fontproperties=fp)
ax.set_ylabel("相関係数 r(O, S_O)", fontproperties=fp)
ax.set_title("(d) エコーチェンバー相関", fontproperties=fp, weight="bold")
ax.legend(prop=fp, fontsize=8)
ax.grid(True, alpha=0.3)

# 4e. Final divergence bar chart (Turn 3)
ax = axes[1, 1]
exp_names = list(all_results.keys())
x_pos = np.arange(len(exp_names))
divergences = []
div_stds = []
for exp_name, exp_data in all_results.items():
    t3_divs = [ed["metrics"][3]["mean_divergence"] for ed in exp_data if 3 in ed["metrics"]]
    divergences.append(np.mean(t3_divs))
    div_stds.append(np.std(t3_divs))
ax.bar(x_pos, divergences, yerr=div_stds, color=[EXPERIMENTS[e]["color"] for e in exp_names],
       capsize=5, alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(exp_names, fontproperties=fp, fontsize=9)
ax.set_ylabel("平均 |O - S_O| (Turn 3)", fontproperties=fp)
ax.set_title("(e) 実験別 最終意見乖離", fontproperties=fp, weight="bold")
ax.grid(True, axis="y", alpha=0.3)

# 4f. Extreme ratio bar chart (Turn 3)
ax = axes[1, 2]
ext_ratios = []
ext_stds = []
for exp_name, exp_data in all_results.items():
    t3_exts = [ed["metrics"][3]["extreme_ratio"] for ed in exp_data if 3 in ed["metrics"]]
    ext_ratios.append(np.mean(t3_exts))
    ext_stds.append(np.std(t3_exts))
ax.bar(x_pos, ext_ratios, yerr=ext_stds, color=[EXPERIMENTS[e]["color"] for e in exp_names],
       capsize=5, alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(exp_names, fontproperties=fp, fontsize=9)
ax.set_ylabel("極端比率 (Turn 3)", fontproperties=fp)
ax.set_title("(f) 実験別 極性化比率", fontproperties=fp, weight="bold")
ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(LOGDIR, "echo_polarization.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("\nSaved: logs/echo_polarization.png")
