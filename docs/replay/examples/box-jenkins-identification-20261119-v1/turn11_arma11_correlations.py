"""Constructed zero-mean ARMA(1,1) population and sample correlations."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import acf, pacf, levinson_durbin


def main():
    root = Path(__file__).resolve().parent
    phi, theta = 0.6, -0.4
    seed, n, burn, nlags = 20261119, 119, 2000, 20
    rng = np.random.default_rng(seed)
    a = rng.normal(size=burn+n+1)
    full = np.empty(burn+n)
    previous = 0.0
    for i in range(burn+n):
        previous = phi*previous+a[i+1]-theta*a[i]
        full[i] = previous
    z = full[burn:]
    rho = np.zeros(nlags+1)
    rho[0] = 1
    rho[1] = (phi-theta)*(1-phi*theta)/(1+theta**2-2*phi*theta)
    for k in range(2, nlags+1):
        rho[k] = phi*rho[k-1]
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
        ax.set(title=title, xticks=[1, 2, 5, 10, 15, 20], ylim=(-0.5, 0.9))
        ax.grid(alpha=0.2)
    for ax in axes[:, 1]:
        ax.axhline(band, color="#b45420", linestyle="--", linewidth=1)
        ax.axhline(-band, color="#b45420", linestyle="--", linewidth=1)
    for ax in axes[1, :]:
        ax.set_xlabel("Lag k")
    axes[0, 0].set_ylabel("Correlation")
    axes[1, 0].set_ylabel("Partial correlation")
    fig.suptitle(r"Constructed ARMA(1,1): $z_t=0.6z_{t-1}+a_t+0.4a_{t-1}$; independent $a_t\sim N(0,1)$")
    fig.tight_layout()
    fig.savefig(root / "turn11_arma11_correlations.png", dpi=170)
    plt.close(fig)
    result = {
        "description": "Constructed zero-mean ARMA(1,1), not paper-towel data or a fitted model",
        "textbook_convention": "z_t=phi_1*z_(t-1)+a_t-theta_1*a_(t-1)",
        "fixed_illustrative_parameters": {"phi_1": phi, "theta_1": theta, "innovation_variance": 1},
        "causal_stationarity": "abs(phi_1)<1",
        "seed": seed, "burn_in": burn, "sample_size": n,
        "population_acf": rho.tolist(), "population_pacf": population_pacf.tolist(),
        "sample_acf": sample_acf.tolist(), "sample_pacf": sample_pacf.tolist(),
        "sample_pacf_method": "Yule-Walker with unadjusted autocovariances (ywm)",
        "reference_band": band,
        "reference_band_caveat": "approximate pointwise white-noise reference, not exact ARMA confidence bands or simultaneous bounds",
        "simulated_working_series": z.tolist(),
    }
    (root / "turn11_arma11_correlations.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"population_acf_1_5": rho[1:6].tolist(),
                      "population_pacf_1_5": population_pacf[1:6].tolist(),
                      "sample_acf_1_5": sample_acf[1:6].tolist(),
                      "sample_pacf_1_5": sample_pacf[1:6].tolist()}, indent=2))


if __name__ == "__main__":
    main()
