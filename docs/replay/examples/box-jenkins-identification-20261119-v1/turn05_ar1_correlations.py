"""Population and finite-sample correlation curves for a zero-mean AR(1)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import acf, pacf


def main():
    root = Path(__file__).resolve().parent
    phi, seed, n, burn, nlags = 0.7, 20261119, 119, 2000, 20
    rng = np.random.default_rng(seed)
    innovations = rng.normal(size=burn + n)
    full = np.empty(burn + n)
    previous = 0.0
    for index, innovation in enumerate(innovations):
        previous = phi * previous + innovation
        full[index] = previous
    z = full[burn:]
    population_acf = phi ** np.arange(nlags + 1)
    population_pacf = np.zeros(nlags + 1)
    population_pacf[:2] = [1, phi]
    sample_acf = acf(z, nlags=nlags, fft=False, adjusted=False)
    sample_pacf = pacf(z, nlags=nlags, method="ywm")
    band = 1.96 / np.sqrt(n)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    lag = np.arange(1, nlags + 1)
    curves = [population_acf, sample_acf, population_pacf, sample_pacf]
    titles = ["Population ACF", f"Sample ACF (n={n})", "Population PACF", f"Sample PACF (n={n})"]
    for ax, values, title in zip(axes.flat, curves, titles):
        ax.axhline(0, color="0.35", linewidth=0.8)
        ax.vlines(lag, 0, values[1:], color="#176087", linewidth=1.8)
        ax.scatter(lag, values[1:], color="#176087", s=23, zorder=3)
        ax.set(title=title, xticks=[1, 5, 10, 15, 20], ylim=(-0.4, 0.85))
        ax.grid(alpha=0.2)
    for ax in axes[:, 1]:
        ax.axhline(band, color="#b45420", linestyle="--", linewidth=1)
        ax.axhline(-band, color="#b45420", linestyle="--", linewidth=1)
    for ax in axes[1, :]:
        ax.set_xlabel("Lag k (weeks)")
    axes[0, 0].set_ylabel("Correlation")
    axes[1, 0].set_ylabel("Partial correlation")
    fig.suptitle(r"Constructed AR(1): $z_t=0.7z_{t-1}+a_t$; independent $a_t\sim N(0,1)$")
    fig.tight_layout()
    fig.savefig(root / "turn05_ar1_correlations.png", dpi=170)
    plt.close(fig)
    result = {
        "description": "Illustrative simulated zero-mean AR(1), not paper-towel data",
        "fixed_phi_1": phi, "innovation_variance": 1,
        "seed": seed, "burn_in": burn, "sample_size": n,
        "initial_value_before_burn_in": 0,
        "population_acf": population_acf.tolist(),
        "population_pacf": population_pacf.tolist(),
        "sample_acf": sample_acf.tolist(), "sample_pacf": sample_pacf.tolist(),
        "sample_pacf_method": "Yule-Walker with unadjusted autocovariances (ywm)",
        "reference_band": band,
        "reference_band_caveat": "pointwise approximate white-noise reference, not exact AR(1) or simultaneous bounds",
        "simulated_working_series": z.tolist(),
    }
    (root / "turn05_ar1_correlations.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"sample_acf_first_5": sample_acf[1:6].tolist(), "sample_pacf_first_5": sample_pacf[1:6].tolist(), "reference_band": band}, indent=2))


if __name__ == "__main__":
    main()
