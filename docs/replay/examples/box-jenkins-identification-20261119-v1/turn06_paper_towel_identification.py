"""Sample correlations of the supplied first-differenced sales."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import acf, pacf


def main():
    root = Path(__file__).resolve().parent
    with (root / "paper_towels.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    y = np.array([float(row["sales_above_100000_in_10000_rolls"]) for row in rows])
    z = np.diff(y)
    nlags = 24
    r = acf(z, nlags=nlags, fft=False, adjusted=False)
    partial = pacf(z, nlags=nlags, method="ywm")
    band = 1.96 / np.sqrt(len(z))
    lag = np.arange(1, nlags + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, values, title in zip(axes, [r, partial], ["Sample ACF", "Sample PACF"]):
        ax.axhline(0, color="0.35", linewidth=0.8)
        ax.axhline(band, color="#b45420", linestyle="--", linewidth=1)
        ax.axhline(-band, color="#b45420", linestyle="--", linewidth=1)
        ax.vlines(lag, 0, values[1:], color="#176087", linewidth=1.8)
        ax.scatter(lag, values[1:], color="#176087", s=22, zorder=3)
        ax.set(title=title, xlabel="Lag k (weeks)", xticks=[1, 4, 8, 12, 16, 20, 24], ylim=(-0.4, 0.6))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Correlation / partial correlation")
    fig.suptitle(r"Paper-towel working series: $z_t=y_t-y_{t-1}$, 119 observed differences")
    fig.tight_layout()
    fig.savefig(root / "turn06_paper_towel_identification.png", dpi=170)
    plt.close(fig)
    result = {
        "source": "supplied paper_towels.csv, four-decimal input values",
        "working_series": "z_t = y_t-y_(t-1)",
        "sample_size": len(z), "lags": list(range(nlags + 1)),
        "sample_acf": r.tolist(), "sample_pacf": partial.tolist(),
        "acf_method": "demeaned, unadjusted autocovariances",
        "pacf_method": "Yule-Walker with unadjusted autocovariances (ywm)",
        "reference_band": band,
        "reference_band_caveat": "approximate pointwise white-noise reference, not simultaneous bounds",
        "acf_lags_outside_reference": [int(k) for k in lag if abs(r[k]) > band],
        "pacf_lags_outside_reference": [int(k) for k in lag if abs(partial[k]) > band],
        "fitted_parameters": None,
    }
    (root / "turn06_paper_towel_identification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
