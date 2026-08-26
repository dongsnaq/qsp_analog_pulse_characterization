from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "main_figure_3_source_data.csv"
OUTPUT = ROOT / "main_figure_3_reproduced.pdf"

PULSES = ["linear", "sinusoidal", "biharmonic"]
LEVELS = [50, 100, 200]
CONFIGURATIONS = [
    ("delta_0.01_M_10000", r"$\delta=0.01,\ M=10^4$"),
    ("delta_0.001_M_1000000", r"$\delta=0.001,\ M=10^6$"),
]


def main() -> None:
    data = pd.read_csv(DATA)
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 8), sharex="col")

    for column, pulse in enumerate(PULSES):
        pulse_data = data[data["pulse"] == pulse]
        reference = pulse_data[
            (pulse_data["noise_configuration"] == CONFIGURATIONS[0][0])
            & (pulse_data["L"] == 100)
        ]

        top = axes[0, column]
        top.plot(reference["t"], reference["ideal"], color="0.85", linewidth=1.5, alpha=0.75, label="ideal")
        top.plot(reference["t"], reference["actual"], color="k", linewidth=2, label="actual")
        top.plot(reference["t"], reference["estimate"], ":", linewidth=2, label="characterized ($L=100$)")
        top.set_title(pulse.capitalize())
        top.set_ylabel("Pulse function")

        for row, (configuration, label) in enumerate(CONFIGURATIONS, start=1):
            axis = axes[row, column]
            for level in LEVELS:
                curve = pulse_data[
                    (pulse_data["noise_configuration"] == configuration)
                    & (pulse_data["L"] == level)
                ]
                axis.plot(curve["t"], curve["error"], linewidth=1, label=f"$L={level}$")
            axis.axhline(0, color="0.7", linestyle="--", linewidth=1)
            axis.set_ylabel(r"Pointwise error $\hat\phi(t)-\phi(t)$" + "\n" + label)
            if row == 2:
                axis.set_xlabel(r"$t$")

        if column == 0:
            top.legend(loc="upper left", fontsize=9)
            axes[1, column].legend(loc="lower center", fontsize=9, ncol=3)
            axes[2, column].legend(loc="lower center", fontsize=9, ncol=3)

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
