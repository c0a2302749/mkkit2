import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_data(log_dir: str) -> tuple[dict, dict, list[dict]]:
    traj_csv = os.path.join(log_dir, "opinion_trajectory.csv")
    summary_json = os.path.join(log_dir, "summary.json")
    actions_json = os.path.join(log_dir, "actions.json")

    if not os.path.exists(traj_csv):
        raise FileNotFoundError(f"opinion_trajectory.csv not found in {log_dir}")
    if not os.path.exists(summary_json):
        raise FileNotFoundError(f"summary.json not found in {log_dir}")

    with open(traj_csv, newline="") as f:
        reader = csv.DictReader(f)
        traj_rows = list(reader)

    with open(summary_json) as f:
        summary = json.load(f)

    actions = []
    if os.path.exists(actions_json):
        with open(actions_json) as f:
            actions = json.load(f)

    return {"trajectory": traj_rows}, summary, actions


def _sorted_turns(traj: list[dict]) -> list[int]:
    turns = sorted(set(int(r["turn"]) for r in traj))
    return turns


def _agent_ids(traj: list[dict]) -> list[int]:
    aids = sorted(set(int(r["agent_id"]) for r in traj))
    return aids


def plot_opinion_evolution(traj: list[dict], output_path: str):
    turns = _sorted_turns(traj)
    agents = _agent_ids(traj)
    n_agents = len(agents)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = turns
    palette = plt.cm.tab10.colors if n_agents <= 10 else plt.cm.tab20.colors

    for i, aid in enumerate(agents):
        ys = []
        for t in turns:
            row = [r for r in traj if int(r["turn"]) == t and int(r["agent_id"]) == aid]
            ys.append(float(row[0]["opinion"]) if row else None)
        color = palette[i % len(palette)]
        persona = next(
            (r["persona"] for r in traj if int(r["agent_id"]) == aid), ""
        )
        ax.plot(x, ys, marker="o", label=f"A{aid}({persona})", color=color, linewidth=1.5)

    ax.set_xlabel("Turn")
    ax.set_ylabel("Opinion")
    ax.set_title("Opinion Evolution by Agent")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  -> {output_path}")


def plot_opinion_distribution(traj: list[dict], output_path: str):
    turns = _sorted_turns(traj)
    data = []
    for t in turns:
        opinions = [
            float(r["opinion"]) for r in traj if int(r["turn"]) == t
        ]
        data.append(opinions)

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, positions=turns, widths=0.6, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.6)

    # Overlay individual points
    for i, t in enumerate(turns):
        ys = data[i]
        jitter = np.random.uniform(-0.15, 0.15, len(ys))
        ax.scatter([t + j for j in jitter], ys, alpha=0.5, s=20, color="crimson")

    ax.set_xlabel("Turn")
    ax.set_ylabel("Opinion")
    ax.set_title("Opinion Distribution per Turn")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(turns)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  -> {output_path}")


def plot_polarization(traj: list[dict], output_path: str):
    turns = _sorted_turns(traj)
    extreme_ratios = []
    stds = []
    for t in turns:
        opinions = np.array([
            float(r["opinion"]) for r in traj if int(r["turn"]) == t
        ])
        n = len(opinions)
        extreme = np.sum((opinions <= 0.2) | (opinions >= 0.8)) / n if n > 0 else 0
        extreme_ratios.append(extreme)
        stds.append(float(np.std(opinions)) if n > 0 else 0)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(turns, extreme_ratios, marker="s", color="crimson", linewidth=2, label="Extreme ratio (≤0.2 or ≥0.8)")
    ax1.set_xlabel("Turn")
    ax1.set_ylabel("Extreme ratio", color="crimson")
    ax1.tick_params(axis="y", labelcolor="crimson")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(turns, stds, marker="o", color="steelblue", linewidth=2, linestyle="--", label="Opinion std")
    ax2.set_ylabel("Std deviation", color="steelblue")
    ax2.tick_params(axis="y", labelcolor="steelblue")
    ax2.set_ylim(0, max(stds + [0.5]) * 1.1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    ax1.set_title("Polarization over Time")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  -> {output_path}")


def print_config_table(summary: dict):
    config = summary.get("config", {})
    print()
    print("=" * 50)
    print("Experiment Configuration")
    print("=" * 50)
    key_width = max(len(k) for k in config) + 2
    for k, v in config.items():
        print(f"  {k:<{key_width}}{v}")
    print()


def print_decision_table(summary: dict):
    decisions = summary.get("decisions")
    if not decisions:
        print("  No decision data.")
        return

    print("=" * 60)
    print("Decision Summary")
    print("=" * 60)
    print(f"  Total proposals:    {decisions.get('total_proposals', 0)}")
    print(f"  Resolved:           {decisions.get('resolved', 0)}")
    print(f"  Passed / Failed:    {decisions.get('passed', 0)} / {decisions.get('failed', 0)}")
    print(f"  Min resolution turn: {decisions.get('min_turn_resolved', '-')}")

    details = decisions.get("proposal_details", [])
    if details:
        print()
        print(f"  {'ID':<4} {'Proposer':<9} {'Status':<8} {'For':<4} {'Against':<8} {'Turn':<5} Content")
        print(f"  {'-'*4} {'-'*9} {'-'*8} {'-'*4} {'-'*8} {'-'*5} {'-'*30}")
        for p in details:
            short = p.get("content", "")[:35]
            print(f"  {p['id']:<4} {'A'+str(p['proposer']):<9} {p['status']:<8} {p['votes_for']:<4} {p['votes_against']:<8} {str(p.get('turn_resolved','-')):<5} {short}")
    print()


def print_action_summary(actions: list[dict]):
    from collections import Counter
    total = len(actions)
    c = Counter(a["action"] for a in actions)
    print("=" * 50)
    print("Action Summary")
    print("=" * 50)
    print(f"  Total actions: {total}")
    for action_name in ["PROPOSE", "COMMENT", "SUPPORT", "OPPOSE", "VOTE", "DO_NOTHING"]:
        cnt = c.get(action_name, 0)
        pct = cnt / total * 100 if total else 0
        print(f"  {action_name:<12} {cnt:>4} ({pct:>5.1f}%)")

    # Per-turn breakdown
    turns = sorted(set(a["turn"] for a in actions))
    print()
    print(f"  Per-turn breakdown:")
    print(f"  {'Turn':<6}", end="")
    for an in ["P", "C", "S", "O", "V", "N"]:
        print(f"{an:>4}", end="")
    print()
    for t in turns:
        ta = [a for a in actions if a["turn"] == t]
        tc = Counter(a["action"] for a in ta)
        print(f"  {t:<6}", end="")
        for an in ["PROPOSE", "COMMENT", "SUPPORT", "OPPOSE", "VOTE", "DO_NOTHING"]:
            print(f"{tc.get(an, 0):>4}", end="")
        print()
    print()


def generate_all(log_dir: str, output_dir: str | None = None):
    if output_dir is None:
        output_dir = log_dir

    print(f"Loading data from: {log_dir}")
    traj_data, summary, actions = load_data(log_dir)
    traj = traj_data["trajectory"]

    total_agents = len(set(int(r["agent_id"]) for r in traj))
    total_turns = len(set(int(r["turn"]) for r in traj))
    print(f"  Agents: {total_agents}, Turns: {total_turns}")

    print("\nGenerating opinion evolution plot...")
    plot_opinion_evolution(traj, os.path.join(output_dir, "opinion_evolution.png"))

    print("\nGenerating opinion distribution plot...")
    plot_opinion_distribution(traj, os.path.join(output_dir, "opinion_distribution.png"))

    print("\nGenerating polarization plot...")
    plot_polarization(traj, os.path.join(output_dir, "polarization.png"))

    print_config_table(summary)

    print_decision_table(summary)

    if actions:
        print_action_summary(actions)

    print(f"\nReports saved to: {output_dir}/")
    return output_dir


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.analysis.report <log_dir> [output_dir]")
        sys.exit(1)
    log_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    generate_all(log_dir, out_dir)


if __name__ == "__main__":
    main()
