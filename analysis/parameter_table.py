import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config.personas import PERSONAS

HEADERS = ["Persona", "α (conformity)", "β (risk sensitivity)", "γ (stance persistence)",
           "λ (memory)", "bias", "w_sys", "w_sns", "initial_opinion"]

def print_table():
    rows = []
    for name, params in PERSONAS.items():
        rows.append([
            name,
            params["alpha"],
            params["beta"],
            params["gamma"],
            params["lambda_"],
            params["bias"],
            params["w_sys"],
            params["w_sns"],
            params["initial_opinion"],
        ])

    col_widths = [max(len(str(r[i])) for r in rows + [HEADERS]) for i in range(len(HEADERS))]
    col_widths = [max(w, len(h)) for w, h in zip(col_widths, HEADERS)]

    sep = " | ".join("-" * w for w in col_widths)
    header = " | ".join(h.ljust(w) for h, w in zip(HEADERS, col_widths))
    print(header)
    print(sep)
    for row in rows:
        vals = " | ".join(str(v).ljust(w) for v, w in zip(row, col_widths))
        print(vals)

    print(f"\nα ∈ [0, 1]: higher = more influenced by social average opinion S_O")
    print(f"β ∈ [0, 1]: higher = more cautious (risk perception reduces opinion)")
    print(f"γ ∈ [0, 1]: higher = stronger anchor to initial opinion O₀")
    print(f"λ ∈ [0, 1]: update rate (1−λ)R + λ·sigmoid(…)")
    print(f"Opinion update: O ← clip(O + α·S_O − β·R − γ·(O−O₀) + noise, 0, 1)")

if __name__ == "__main__":
    print_table()
