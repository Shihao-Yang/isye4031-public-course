"""Constructed zero-mean MA(2): population and sample correlations."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import acf, pacf, levinson_durbin


def main():
    root = Path(__file__).resolve().parent
    theta1, theta2 = 0.5, 0.3
    seed, n, burn, nlags = 20261119, 119, 2000, 20
    rng = np.random.default_rng(seed)
    a = rng.normal(size=burn + n + 2)
    full = a[2:] - theta1*a[1:-1] - theta2*a[:-2]
    z = full[burn:]
    variance = 1 + theta1**2 + theta2**2
    rho = np.zeros(nlags + 1)
    rho[:3] = [1, theta1*(theta2-1)/variance, -theta2/variance]
    population_pacf = levinson_durbin(rho, nlags=nlags, isacov=True)[2]
    sample_acf = acf(z, nlags=nlags, fft=False, adjusted=False)
    sample_pacf = pacf(z, nlags=nlags, method="ywm")
    band = 1.96/np.sqrt(n)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    lag = np.arange(1, nlags+1)
    for ax, values, title in zip(axes.flat,
        [rho, sample_acf, population_pacf, sample_pacf],
        ["Population ACF", f"Sample ACF (n={n})", "Population PACF", f"Sample PACF (n={n})"]):
        ax.axhline(0, color="0.35", linewidth=0.8)
        ax.vlines(lag, 0, values[1:], color="#176087", linewidth=1.8)
        ax.scatter(lag, values[1:], color="#176087", s=23, zorder=3)
        ax.set(title=title, xticks=[1, 2, 5, 10, 15, 20], ylim=(-0.55, 0.4))
        ax.grid(alpha=0.2)
    for ax in axes[:, 1]:
        ax.axhline(band, color="#b45420", linestyle="--", linewidth=1)
        ax.axhline(-band, color="#b45420", linestyle="--", linewidth=1)
    for ax in axes[1, :]:
        ax.set_xlabel("Lag k")
    axes[0, 0].set_ylabel("Correlation")
    axes[1, 0].set_ylabel("Partial correlation")
    fig.suptitle(r"Constructed MA(2): $z_t=a_t-0.5a_{t-1}-0.3a_{t-2}$; independent $a_t\sim N(0,1)$")
    fig.tight_layout()
    fig.savefig(root / "turn09_ma2_correlations.png", dpi=170)
    plt.close(fig)
    result = {
        "description": "Constructed zero-mean MA(2), not paper-towel data or a fitted model",
        "fixed_illustrative_parameters": {"theta_1": theta1, "theta_2": theta2, "innovation_variance": 1},
        "seed": seed, "burn_in": burn, "sample_size": n,
        "population_acf": rho.tolist(), "population_pacf": population_pacf.tolist(),
        "sample_acf": sample_acf.tolist(), "sample_pacf": sample_pacf.tolist(),
        "sample_pacf_method": "Yule-Walker with unadjusted autocovariances (ywm)",
        "reference_band": band,
        "reference_band_caveat": "approximate pointwise white-noise reference, not exact MA(2) confidence bands or simultaneous bounds",
        "simulated_working_series": z.tolist(),
    }
    (root / "turn09_ma2_correlations.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"population_acf_1_2": rho[1:3].tolist(),
                      "population_pacf_1_6": population_pacf[1:7].tolist(),
                      "sample_acf_1_6": sample_acf[1:7].tolist(),
                      "sample_acf_lags_outside_reference": [int(k) for k in lag if abs(sample_acf[k]) > band]}, indent=2))


if __name__ == "__main__":
    main()
