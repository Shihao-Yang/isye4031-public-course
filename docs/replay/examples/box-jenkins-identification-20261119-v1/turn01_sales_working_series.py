"""Plot supplied paper-towel sales and their first differences."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    with (root / "paper_towels.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    week = np.array([int(row["week"]) for row in rows])
    y = np.array([float(row["sales_above_100000_in_10000_rolls"]) for row in rows])
    z = np.diff(y)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.8), sharex=True)
    axes[0].plot(week, y, color="#176087", linewidth=1.7)
    axes[0].set(title="Paper-towel sales: 120 observed weeks", ylabel=r"$y_t$ (10,000 rolls above 100,000)")
    axes[1].plot(week[1:], z, color="#b45420", linewidth=1.4)
    axes[1].axhline(0, color="0.35", linewidth=0.8)
    axes[1].set(title=r"Working series: first differences $z_t=y_t-y_{t-1}$", ylabel=r"$z_t$ (10,000 rolls)", xlabel="Week (calendar dates unspecified)")
    for ax in axes:
        ax.grid(alpha=0.22)
        ax.set_xlim(1, 120)
    fig.tight_layout()
    fig.savefig(root / "turn01_sales_working_series.png", dpi=170)
    plt.close(fig)
    results = {
        "source": "paper_towels.csv, supplied four-decimal observations",
        "n_sales": len(y), "n_first_differences": len(z),
        "sales_units": "10000 rolls above a baseline of 100000 rolls",
        "actual_rolls_formula": "100000 + 10000*y_t",
        "working_series_formula": "z_t = y_t - y_(t-1)",
        "first_sales_value": float(y[0]),
        "first_actual_rolls": float(100000 + 10000*y[0]),
        "first_difference": float(z[0]),
        "mean_first_difference": float(z.mean()),
        "working_series": [{"week": int(t), "z": round(float(v), 4)} for t, v in zip(week[1:], z)],
        "fitted_parameters": None,
    }
    (root / "turn01_sales_working_series.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({key: value for key, value in results.items() if key != "working_series"}, indent=2))


if __name__ == "__main__":
    main()
