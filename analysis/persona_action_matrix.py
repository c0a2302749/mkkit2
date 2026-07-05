import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PERSONAS_ORDERED = [
    "rational",
    "conformist",
    "information_seeker",
    "risk_overestimator",
    "skeptic",
    "agitator",
]
ACTIONS_TURN2 = ["SUPPORT", "OPPOSE", "COMMENT"]

PERSONA_PARAMS = {
    "rational": {"alpha": 0.2, "beta": 0.4, "gamma": 0.6},
    "conformist": {"alpha": 0.7, "beta": 0.4, "gamma": 0.1},
    "information_seeker": {"alpha": 0.4, "beta": 0.35, "gamma": 0.3},
    "risk_overestimator": {"alpha": 0.4, "beta": 0.7, "gamma": 0.4},
    "skeptic": {"alpha": 0.15, "beta": 0.15, "gamma": 0.6},
    "agitator": {"alpha": 0.2, "beta": 0.1, "gamma": 0.6},
}


def collect_data(log_root="logs"):
    counts = {p: {a: 0 for a in ACTIONS_TURN2} for p in PERSONAS_ORDERED}
    run_details = []

    log_dirs = sorted(glob.glob(os.path.join(log_root, "*")))
    for d in log_dirs:
        actions_path = os.path.join(d, "actions.json")
        if not os.path.exists(actions_path):
            continue
        with open(actions_path) as f:
            actions = json.load(f)
        turn2 = [a for a in actions if a.get("turn") == 2]
        run_name = os.path.basename(d)
        details = []
        for a in turn2:
            p = a.get("persona", "")
            act = a.get("action", "")
            pid = a.get("proposal_id")
            details.append((p, act, pid))
            if p in counts and act in counts[p]:
                counts[p][act] += 1
        run_details.append((run_name, details))

    return counts, run_details


def print_table(counts):
    print("=" * 80)
    print("Persona x Action Type Matrix (Turn 2, 3 runs aggregated)")
    print("=" * 80)
    header = f"{'Persona':<22} {'SUPPORT':>8} {'OPPOSE':>8} {'COMMENT':>8} {'Total':>8}   {'α':>5} {'β':>5} {'γ':>5}  {'dominant'}"
    print(header)
    print("-" * 80)
    for p in PERSONAS_ORDERED:
        s = counts[p]["SUPPORT"]
        o = counts[p]["OPPOSE"]
        c = counts[p]["COMMENT"]
        total = s + o + c
        params = PERSONA_PARAMS[p]
        # determine dominant action
        acts = {"SUPPORT": s, "OPPOSE": o, "COMMENT": c}
        dom = max(acts, key=acts.get)
        pct = acts[dom] / total * 100 if total > 0 else 0
        dom_str = f"{dom} ({pct:.0f}%)" if total > 0 else "-"
        print(
            f"{p:<22} {s:>8} {o:>8} {c:>8} {total:>8}   "
            f"{params['alpha']:>5.2f} {params['beta']:>5.2f} {params['gamma']:>5.2f}  {dom_str}"
        )
    print()


def print_run_details(run_details):
    print("Per-run Turn 2 actions:")
    for run_name, details in run_details:
        print(f"  [{run_name}]")
        for p, act, pid in details:
            pid_str = f"P{pid}" if pid else "-"
            print(f"    {p:<22} {act:<8} ({pid_str})")
        print()


def plot_heatmap(counts, output_path):
    matrix = np.array([[counts[p][a] for a in ACTIONS_TURN2] for p in PERSONAS_ORDERED])
    row_totals = matrix.sum(axis=1, keepdims=True)
    row_totals = np.where(row_totals == 0, 1, row_totals)
    norm_matrix = matrix / row_totals

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(norm_matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(ACTIONS_TURN2)))
    ax.set_xticklabels(ACTIONS_TURN2)
    ax.set_yticks(range(len(PERSONAS_ORDERED)))
    ax.set_yticklabels(PERSONAS_ORDERED)

    for i in range(len(PERSONAS_ORDERED)):
        for j in range(len(ACTIONS_TURN2)):
            val = matrix[i, j]
            color = "white" if norm_matrix[i, j] > 0.5 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=11)

    ax.set_xlabel("Action Type", fontsize=12)
    ax.set_ylabel("Persona", fontsize=12)
    ax.set_title("Persona x Action Type (Turn 2, aggregated over 3 runs)", fontsize=13)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion within persona", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    log_root = "logs"
    counts, run_details = collect_data(log_root)
    print_table(counts)
    print_run_details(run_details)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "fig_action_matrix.png")
    plot_heatmap(counts, out_path)


if __name__ == "__main__":
    main()
