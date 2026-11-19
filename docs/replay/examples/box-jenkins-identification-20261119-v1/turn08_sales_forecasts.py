"""Two-week sales forecasts from the saved conditional MA(1) fit."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    fitted = json.loads((root / "turn07_ma1_conditional_ls.json").read_text())
    with (root / "paper_towels.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    y = np.array([float(row["sales_above_100000_in_10000_rolls"]) for row in rows])
    theta = fitted["fitted_parameters"]["theta_1"]
    last_error = fitted["records"][-1]["residual"]
    dz = [-theta * last_error, 0.0]
    levels = y[-1] + np.cumsum(dz)
    rolls = 100000 + 10000 * levels
    fig, ax = plt.subplots(figsize=(9, 4.2))
    weeks = np.arange(1, len(y) + 1)
    ax.plot(weeks[-20:], (100000 + 10000*y[-20:])/1000,
            color="#176087", marker="o", markersize=3, label="Observed sales")
    ax.plot([120, 121, 122], np.r_[100000+10000*y[-1], rolls]/1000,
            color="#b45420", linestyle="--", marker="o", label="Point forecasts made at week 120")
    ax.axvline(120, color="0.5", linewidth=0.8, linestyle=":")
    ax.set(xlabel="Week (calendar dates unspecified)", ylabel="Sales (thousand rolls)",
           title="Paper-towel sales: next two weeks from the conditional MA(1) fit",
           xticks=[101, 105, 110, 115, 120, 121, 122])
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "turn08_sales_forecasts.png", dpi=170)
    plt.close(fig)
    result = {
        "fit_source": "turn07_ma1_conditional_ls.json",
        "fitted_parameters": {"theta_1": theta},
        "forecast_origin": 120, "last_observed_y": float(y[-1]),
        "last_observed_rolls": float(100000+10000*y[-1]),
        "last_observed_difference": float(y[-1]-y[-2]),
        "last_recursive_residual": last_error,
        "assumption": "future innovations have conditional mean zero; retain fitted coefficient and e_1=0 residual initialization",
        "forecasts": [{"week": 121+i, "difference_forecast": float(dz[i]),
                       "y_forecast": float(levels[i]), "sales_rolls_forecast": float(rolls[i])}
                      for i in range(2)],
        "future_observations_used": False,
        "prediction_intervals": None,
    }
    (root / "turn08_sales_forecasts.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
