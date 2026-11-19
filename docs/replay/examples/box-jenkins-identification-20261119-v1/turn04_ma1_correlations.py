"""Population and finite-sample correlations for an illustrative MA(1)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import acf, pacf, levinson_durbin


def main():
    root = Path(__file__).resolve().parent
    theta, seed, n, burn, nlags = 0.7, 20261119, 119, 2000, 20
    rng = np.random.default_rng(seed)
    innovations = rng.normal(size=burn + n + 1)
    full = innovations[1:] - theta * innovations[:-1]
    z = full[burn:]
    population_acf = np.zeros(nlags + 1)
    population_acf[0] = 1
    population_acf[1] = -theta / (1 + theta**2)
    population_pacf = levinson_durbin(population_acf, nlags=nlags, isacov=True)[2]
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
        ax.set(title=title, xticks=[1, 5, 10, 15, 20], ylim=(-0.65, 0.45))
        ax.grid(alpha=0.2)
    for ax in axes[:, 1]:
        ax.axhline(band, color="#b45420", linestyle="--", linewidth=1)
        ax.axhline(-band, color="#b45420", linestyle="--", linewidth=1)
    for ax in axes[1, :]:
        ax.set_xlabel("Lag k (weeks)")
    axes[0, 0].set_ylabel("Correlation")
    axes[1, 0].set_ylabel("Partial correlation")
    fig.suptitle(r"Constructed MA(1): $z_t=a_t-0.7a_{t-1}$; independent $a_t\sim N(0,1)$")
    fig.tight_layout()
    fig.savefig(root / "turn04_ma1_correlations.png", dpi=170)
    plt.close(fig)
    result = {
        "description": "Illustrative simulated MA(1), not the paper-towel data",
        "fixed_theta_1": theta, "innovation_variance": 1,
        "seed": seed, "burn_in": burn, "sample_size": n,
        "population_acf": population_acf.tolist(),
        "population_pacf": population_pacf.tolist(),
        "sample_acf": sample_acf.tolist(), "sample_pacf": sample_pacf.tolist(),
        "sample_pacf_method": "Yule-Walker, unadjusted autocovariances (ywm)",
        "reference_band": band,
        "reference_band_caveat": "pointwise approximate white-noise reference, not simultaneous or exact MA(1) bands",
        "simulated_working_series": z.tolist(),
    }
    (root / "turn04_ma1_correlations.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"population_rho1": population_acf[1], "population_pacf_lags_1_to_5": population_pacf[1:6].tolist(), "reference_band": band}, indent=2))


if __name__ == "__main__":
    main()
