import csv
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def get_community_id(agent_id: int, n_agents: int, n_communities: int = 2) -> int:
    size = n_agents // n_communities
    return agent_id // size


def plot_community_opinions(traj_csv: str, output_path: str, n_communities: int = 2):
    rows = []
    with open(traj_csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    n_agents = max(int(r["agent_id"]) for r in rows) + 1
    turns = sorted(set(int(r["turn"]) for r in rows))

    community_avgs: dict[int, list[float]] = {c: [] for c in range(n_communities)}
    community_stds: dict[int, list[float]] = {c: [] for c in range(n_communities)}

    for t in turns:
        turn_rows = [r for r in rows if int(r["turn"]) == t]
        for c in range(n_communities):
            members = [r for r in turn_rows if get_community_id(int(r["agent_id"]), n_agents, n_communities) == c]
            opinions = [float(r["opinion"]) for r in members]
            community_avgs[c].append(np.mean(opinions))
            community_stds[c].append(np.std(opinions))

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    for c in range(n_communities):
        color = colors[c % len(colors)]
        ax.plot(turns, community_avgs[c], marker="o", linewidth=2.5,
                color=color, label=f"Community {c} avg")
        ax.fill_between(turns,
                         [community_avgs[c][i] - community_stds[c][i] for i in range(len(turns))],
                         [community_avgs[c][i] + community_stds[c][i] for i in range(len(turns))],
                         color=color, alpha=0.1)

    all_avgs = [np.mean([float(r["opinion"]) for r in rows if int(r["turn"]) == t]) for t in turns]
    ax.plot(turns, all_avgs, marker="s", linewidth=2, color="gray",
            linestyle="--", label="Overall avg")

    ax.set_xlabel("Turn")
    ax.set_ylabel("Average Opinion")
    ax.set_title("Phase 2: Intra- vs Inter-Community Opinion Propagation")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  -> {output_path}")


def plot_per_agent(traj_csv: str, output_path: str, n_communities: int = 2):
    rows = []
    with open(traj_csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    n_agents = max(int(r["agent_id"]) for r in rows) + 1
    turns = sorted(set(int(r["turn"]) for r in rows))
    colors = ["#2196F3", "#FF5722"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for aid in range(n_agents):
        cid = get_community_id(aid, n_agents, n_communities)
        agent_rows = [r for r in rows if int(r["agent_id"]) == aid]
        ys = [float(r["opinion"]) for r in agent_rows]
        persona = agent_rows[0]["persona"]
        color = colors[cid % len(colors)]
        alpha = 0.6 if cid == 0 else 0.6
        ax.plot(turns, ys, marker=".", linewidth=1.2, color=color, alpha=0.7,
                label=f"A{aid}({persona})" if aid < n_communities * 2 else "")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=colors[0], linewidth=2, label="Community 0"),
        Line2D([0], [0], color=colors[1], linewidth=2, label="Community 1"),
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=9)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Opinion")
    ax.set_title("Phase 2: Agent Opinions by Community")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  -> {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 analysis/phase2_plot.py <log_dir>")
        sys.exit(1)
    log_dir = sys.argv[1]
    traj_csv = os.path.join(log_dir, "opinion_trajectory.csv")
    plot_community_opinions(traj_csv, os.path.join(log_dir, "phase2_community_avg.png"))
    plot_per_agent(traj_csv, os.path.join(log_dir, "phase2_per_agent.png"))
