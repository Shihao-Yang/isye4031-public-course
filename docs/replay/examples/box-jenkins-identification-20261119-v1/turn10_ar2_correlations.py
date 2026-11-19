"""Constructed stationary zero-mean AR(2) population and sample curves."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import acf, pacf


def main():
    root = Path(__file__).resolve().parent
    phi1, phi2 = 0.6, -0.3
    seed, n, burn, nlags = 20261119, 119, 2000, 20
    rng = np.random.default_rng(seed)
    innovations = rng.normal(size=burn+n)
    full = np.empty(burn+n)
    previous, previous2 = 0.0, 0.0
    for i, innovation in enumerate(innovations):
        value = phi1*previous + phi2*previous2 + innovation
        full[i] = value
        previous2, previous = previous, value
    z = full[burn:]
    rho = np.zeros(nlags+1)
    rho[:2] = [1, phi1/(1-phi2)]
    for k in range(2, nlags+1):
        rho[k] = phi1*rho[k-1]+phi2*rho[k-2]
    population_pacf = np.zeros(nlags+1)
    population_pacf[:3] = [1, rho[1], phi2]
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
        ax.set(title=title, xticks=[1, 2, 5, 10, 15, 20], ylim=(-0.55, 0.65))
        ax.grid(alpha=0.2)
    for ax in axes[:, 1]:
        ax.axhline(band, color="#b45420", linestyle="--", linewidth=1)
        ax.axhline(-band, color="#b45420", linestyle="--", linewidth=1)
    for ax in axes[1, :]:
        ax.set_xlabel("Lag k")
    axes[0, 0].set_ylabel("Correlation")
    axes[1, 0].set_ylabel("Partial correlation")
    fig.suptitle(r"Constructed AR(2): $z_t=0.6z_{t-1}-0.3z_{t-2}+a_t$; independent $a_t\sim N(0,1)$")
    fig.tight_layout()
    fig.savefig(root / "turn10_ar2_correlations.png", dpi=170)
    plt.close(fig)
    stationarity_conditions = {
        "abs_phi2_less_than_one": bool(abs(phi2)<1),
        "phi1_plus_phi2_less_than_one": bool(phi1+phi2<1),
        "phi2_minus_phi1_less_than_one": bool(phi2-phi1<1),
    }
    result = {
        "description": "Constructed zero-mean AR(2), not paper-towel data or a fitted model",
        "fixed_illustrative_parameters": {"phi_1": phi1, "phi_2": phi2, "innovation_variance": 1},
        "causal_stationarity_checks": stationarity_conditions,
        "seed": seed, "burn_in": burn, "sample_size": n,
        "initial_values_before_burn_in": [0, 0],
        "population_acf": rho.tolist(), "population_pacf": population_pacf.tolist(),
        "sample_acf": sample_acf.tolist(), "sample_pacf": sample_pacf.tolist(),
        "sample_pacf_method": "Yule-Walker with unadjusted autocovariances (ywm)",
        "reference_band": band,
        "reference_band_caveat": "approximate pointwise white-noise reference, not exact AR(2) confidence bands or simultaneous bounds",
        "simulated_working_series": z.tolist(),
    }
    (root / "turn10_ar2_correlations.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"population_acf_1_6": rho[1:7].tolist(),
                      "population_pacf_1_3": population_pacf[1:4].tolist(),
                      "sample_pacf_1_6": sample_pacf[1:7].tolist(),
                      "sample_pacf_lags_outside_reference": [int(k) for k in lag if abs(sample_pacf[k]) > band],
                      "causal_stationarity_checks": stationarity_conditions}, indent=2))


if __name__ == "__main__":
    main()
