"""Conditional least squares for the zero-mean MA(1), textbook minus sign."""
import csv
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares


def residuals(theta, z):
    errors = np.empty(len(z))
    previous = 0.0
    for i, value in enumerate(z):
        previous = value + float(theta[0]) * previous
        errors[i] = previous
    return errors


def main():
    root = Path(__file__).resolve().parent
    with (root / "paper_towels.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    y = np.array([float(row["sales_above_100000_in_10000_rolls"]) for row in rows])
    z = np.diff(y)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = least_squares(residuals, x0=[0.0], args=(z,), ftol=1e-13, xtol=1e-13, gtol=1e-13, max_nfev=10000)
    theta = float(fit.x[0])
    errors = residuals([theta], z)
    predictions = z - errors
    records = [
        {"week": i + 2, "z": float(value),
         "previous_residual": 0.0 if i == 0 else float(errors[i-1]),
         "prediction": float(predictions[i]), "residual": float(errors[i]),
         "squared_residual": float(errors[i]**2)}
        for i, value in enumerate(z)
    ]
    grid = np.linspace(-0.95, 0.95, 401)
    objective = np.array([np.sum(residuals([v], z)**2) for v in grid])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(grid, objective, color="#176087")
    ax.scatter([theta], [errors @ errors], color="#b45420", zorder=3,
               label=f"CLS: theta = {theta:.4f}, S = {errors @ errors:.3f}")
    ax.set(xlabel=r"Trial coefficient $\theta_1$ (textbook minus-sign convention)",
           ylabel=r"$S(\theta_1)=\sum_{t=2}^{120}e_t(\theta_1)^2$",
           title="Paper-towel differences: conditional sum of squared residuals")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "turn07_ma1_conditional_ls.png", dpi=170)
    plt.close(fig)
    result = {
        "source": "supplied four-decimal paper_towels.csv",
        "method": "conditional least squares, zero-mean MA(1), no intercept",
        "model": "z_t=a_t-theta_1*a_(t-1)",
        "conditioning": "e_1=0, before first available difference z_2",
        "recursion": "prediction_t=-theta_1*e_(t-1); e_t=z_t+theta_1*e_(t-1)",
        "fitted_parameters": {"theta_1": theta},
        "n": len(z), "sum_squared_residuals": float(errors @ errors),
        "sse_at_theta_zero": float(z @ z),
        "optimizer": {"name": "scipy.optimize.least_squares", "start": 0.0,
                      "coefficient_bounds": "none", "success": bool(fit.success),
                      "message": str(fit.message), "nfev": int(fit.nfev),
                      "optimality": float(fit.optimality)},
        "warnings": [str(w.message) for w in caught],
        "predictions_use": "past recursive residuals at a coefficient estimated using all 119 differences; in-sample, not held-out forecasts",
        "records": records,
        "objective_plot_grid": {"theta": grid.tolist(), "sse": objective.tolist()},
    }
    (root / "turn07_ma1_conditional_ls.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ["records", "objective_plot_grid"]}, indent=2))
    print(json.dumps(records[:5], indent=2))


if __name__ == "__main__":
    main()
