from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "main_figure_2_bias_scaling.csv"
OUTPUT = ROOT / "main_figure_2_reproduced.pdf"


def main() -> None:
    data = pd.read_csv(DATA)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.loglog(data["L"], data["direct_overall"], "o-", label="direct (overall)")
    ax.loglog(data["L"], data["direct_interior"], "x--", label="direct (interior)")
    ax.loglog(data["L"], data["refined_overall"], "o-", label="refined (overall)")
    ax.loglog(data["L"], data["refined_interior"], "x--", label="refined (interior)")

    guide_L = np.geomspace(70, 90, 80)
    ax.loglog(guide_L, 2.0 / guide_L, "k--", linewidth=1.2)
    ax.text(90, 2.0 / 80, r"$1/L$", ha="left", va="bottom")
    ax.loglog(guide_L, 3.0 / guide_L**2, "k--", linewidth=1.2)
    ax.text(90, 3.0 / 80**2, r"$1/L^2$", ha="left", va="bottom")

    ax.set_xlabel(r"$L$ (logical-level digitization parameter)")
    ax.set_ylabel("Maximum learning error")
    ax.grid(True, which="both", linestyle=":")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
