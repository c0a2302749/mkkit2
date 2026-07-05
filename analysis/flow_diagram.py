import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def make_box(ax, x, y, w, h, color, text_lines, edgecolor="k"):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1",
        facecolor=color, edgecolor=edgecolor, linewidth=1.5,
    )
    ax.add_patch(rect)
    cx = x + w / 2
    cy = y + h / 2
    offset = (len(text_lines) - 1) * 0.06
    for i, line in enumerate(text_lines):
        ax.text(cx, cy + offset - i * 0.12, line,
                ha="center", va="center", fontsize=10, fontweight="bold" if i == 0 else "normal")


def make_arrow(ax, x1, y1, x2, y2, label=""):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color="gray", lw=2.5),
    )
    if label:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2 + 0.08
        ax.text(mx, my, label, ha="center", va="bottom", fontsize=9, color="gray", fontstyle="italic")


def main():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    box_w = 2.8
    box_h = 2.0

    # Turn labels
    for i, (label, x) in enumerate([("Turn 1: Ideation", 1.2), ("Turn 2: Discussion", 4.8), ("Turn 3: Voting", 8.4)]):
        ax.text(x + box_w / 2, 4.5, label, ha="center", va="center", fontsize=13, fontweight="bold", color="#333")

    # Box 1: Ideation
    make_box(ax, 1.2, 1.5, box_w, box_h, "#E3F2FD", [
        "PROPOSE only",
        "",
        "8 agents each",
        "propose 1 policy",
        "",
        "8 proposals generated",
    ])

    # Box 2: Discussion
    make_box(ax, 4.8, 1.5, box_w, box_h, "#FFF3E0", [
        "SUPPORT / OPPOSE",
        "/ COMMENT",
        "",
        "SNS timeline:",
        "followees only",
        "",
        "Opinion dynamics",
    ])

    # Box 3: Voting
    make_box(ax, 8.4, 1.5, box_w, box_h, "#E8F5E9", [
        "VOTE (multi-vote)",
        "",
        "All 8 proposals",
        "simultaneously",
        "",
        "Majority decision",
    ])

    # Arrows between boxes
    make_arrow(ax, 4.0, 2.5, 4.8, 2.5)
    make_arrow(ax, 7.6, 2.5, 8.4, 2.5)

    # Bottom annotations
    annotations = [
        (2.6, 0.8, "Policy proposals\ncollected"),
        (6.2, 0.8, "Social influence\nthrough network"),
        (9.8, 0.8, "Pass/fail per\nproposal"),
    ]
    for x, y, txt in annotations:
        ax.text(x, y, txt, ha="center", va="top", fontsize=8, color="#666", linespacing=1.4)

    # Title
    ax.text(6, 4.9, "Simulation Flow: 3 Turns", ha="center", va="bottom",
            fontsize=14, fontweight="bold")

    fig.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "fig_flow_diagram.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
