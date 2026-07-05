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

PERSONA_PARAMS = {
    "rational": {"alpha": 0.2, "beta": 0.4, "gamma": 0.6},
    "conformist": {"alpha": 0.7, "beta": 0.4, "gamma": 0.1},
    "information_seeker": {"alpha": 0.4, "beta": 0.35, "gamma": 0.3},
    "risk_overestimator": {"alpha": 0.4, "beta": 0.7, "gamma": 0.4},
    "skeptic": {"alpha": 0.15, "beta": 0.15, "gamma": 0.6},
    "agitator": {"alpha": 0.2, "beta": 0.1, "gamma": 0.6},
}

COLORS = {
    "rational": "#1f77b4",
    "conformist": "#ff7f0e",
    "information_seeker": "#2ca02c",
    "risk_overestimator": "#d62728",
    "skeptic": "#9467bd",
    "agitator": "#8c564b",
}


def collect_data(log_root="logs"):
    results = []

    log_dirs = sorted(glob.glob(os.path.join(log_root, "*")))
    for d in log_dirs:
        actions_path = os.path.join(d, "actions.json")
        traj_path = os.path.join(d, "opinion_trajectory.csv")
        if not os.path.exists(actions_path) or not os.path.exists(traj_path):
            continue

        with open(actions_path) as f:
            actions = json.load(f)
        with open(traj_path, newline="") as f:
            reader = csv.DictReader(f)
            traj_rows = list(reader)

        last_turn = max(int(r["turn"]) for r in traj_rows)
        final = {}
        for r in traj_rows:
            if int(r["turn"]) == last_turn:
                aid = int(r["agent_id"])
                final[aid] = {
                    "opinion": float(r["opinion"]),
                    "risk_perception": float(r["risk_perception"]),
                    "persona": r["persona"],
                }

        vote_actions = [a for a in actions if a.get("action") == "VOTE"]
        for a in vote_actions:
            aid = a.get("agent_id")
            persona = a.get("persona", "")
            votes = a.get("votes")
            if not votes:
                continue
            yes_count = sum(1 for v in votes if v.get("vote") == "yes")
            total = len(votes)
            yes_ratio = yes_count / total if total > 0 else 0.0

            if aid in final:
                params = PERSONA_PARAMS.get(persona, {})
                results.append({
                    "persona": persona,
                    "opinion": final[aid]["opinion"],
                    "risk_perception": final[aid]["risk_perception"],
                    "yes_ratio": yes_ratio,
                    "beta": params.get("beta", 0.0),
                    "alpha": params.get("alpha", 0.0),
                    "run": os.path.basename(d),
                })

    return results


def _pearsonr(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)
    if n < 3:
        return 0.0, 1.0
    r_num = n * np.sum(x * y) - np.sum(x) * np.sum(y)
    denom = np.sqrt((n * np.sum(x**2) - np.sum(x) ** 2) * (n * np.sum(y**2) - np.sum(y) ** 2))
    if denom == 0:
        return 0.0, 1.0
    r = r_num / denom
    r = np.clip(r, -1.0, 1.0)
    t_stat = r * np.sqrt((n - 2) / (1 - r**2)) if abs(r) < 0.9999 else float("inf")
    import math
    from scipy.stats import t as t_dist
    try:
        from scipy.stats import t as t_dist
        p = 2 * (1 - t_dist.cdf(abs(t_stat), df=n - 2))
    except ImportError:
        p = float("nan")
    return float(r), float(p)


def plot_analysis(results, output_path):
    unique_personas = sorted(set(r["persona"] for r in results))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: opinion vs yes_ratio
    ax1 = axes[0]
    for p in unique_personas:
        pts = [r for r in results if r["persona"] == p]
        xs = [pt["opinion"] for pt in pts]
        ys = [pt["yes_ratio"] for pt in pts]
        ax1.scatter(xs, ys, label=p, color=COLORS.get(p, "gray"), s=50, alpha=0.8, edgecolors="k", linewidth=0.5)

    all_x = [r["opinion"] for r in results]
    all_y = [r["yes_ratio"] for r in results]
    if len(all_x) >= 3:
        coeffs = np.polyfit(all_x, all_y, 1)
        x_line = np.linspace(min(all_x), max(all_x), 100)
        y_line = np.polyval(coeffs, x_line)
        ax1.plot(x_line, y_line, "k--", alpha=0.4, linewidth=1.5)
        r_val, p_val = _pearsonr(all_x, all_y)
        ax1.text(
            0.05, 0.05, f"Pearson r={r_val:.3f}",
            transform=ax1.transAxes, fontsize=9, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    # Annotate key points
    for r in results:
        if r["persona"] == "risk_overestimator" and r["yes_ratio"] >= 0.9:
            ax1.annotate(
                "risk_overestimator\n(β=0.7)", (r["opinion"], r["yes_ratio"]),
                xytext=(5, 10), textcoords="offset points", fontsize=7,
                color=COLORS["risk_overestimator"], alpha=0.7,
            )

    ax1.set_xlabel("Final Opinion", fontsize=12)
    ax1.set_ylabel("Yes Ratio (votes)", fontsize=12)
    ax1.set_title("Vote-Opinion Alignment", fontsize=13)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="lower right", fontsize=8, ncol=1)
    ax1.grid(alpha=0.3)

    # Panel 2: beta vs yes_ratio
    ax2 = axes[1]
    for p in unique_personas:
        pts = [r for r in results if r["persona"] == p]
        xs = [pt["beta"] for pt in pts]
        ys = [pt["yes_ratio"] for pt in pts]
        ax2.scatter(xs, ys, label=p, color=COLORS.get(p, "gray"), s=50, alpha=0.8, edgecolors="k", linewidth=0.5)

    all_beta = [r["beta"] for r in results]
    if len(all_beta) >= 3:
        coeffs = np.polyfit(all_beta, all_y, 1)
        x_line = np.linspace(min(all_beta), max(all_beta), 100)
        y_line = np.polyval(coeffs, x_line)
        ax2.plot(x_line, y_line, "k--", alpha=0.4, linewidth=1.5)
        r_val, p_val = _pearsonr(all_beta, all_y)
        ax2.text(
            0.05, 0.05, f"Pearson r={r_val:.3f}",
            transform=ax2.transAxes, fontsize=9, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    ax2.set_xlabel("Beta (risk sensitivity)", fontsize=12)
    ax2.set_ylabel("Yes Ratio (votes)", fontsize=12)
    ax2.set_title("Risk Sensitivity vs Vote Ratio", fontsize=13)
    ax2.set_xlim(-0.05, 0.85)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="lower right", fontsize=8, ncol=1)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def print_stats(results):
    print("=" * 70)
    print("Vote-Opinion Alignment: Per-Persona Summary")
    print("=" * 70)
    print(f"{'Persona':<22} {'N':>4} {'avg_opinion':>12} {'avg_yes_ratio':>14} {'β':>5}")
    print("-" * 70)
    for p in PERSONAS_ORDERED:
        pts = [r for r in results if r["persona"] == p]
        if not pts:
            continue
        avg_o = np.mean([r["opinion"] for r in pts])
        avg_y = np.mean([r["yes_ratio"] for r in pts])
        beta = PERSONA_PARAMS[p]["beta"]
        print(f"{p:<22} {len(pts):>4} {avg_o:>12.4f} {avg_y:>14.4f} {beta:>5.2f}")
    print()


def main():
    log_root = "logs"
    results = collect_data(log_root)
    print(f"Collected {len(results)} data points from {log_root}/*")
    print_stats(results)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "fig_vote_opinion_alignment.png")
    plot_analysis(results, out_path)


if __name__ == "__main__":
    main()
