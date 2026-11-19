"""Zero-mean, constructed responses to one isolated innovation."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    phi = theta = 0.7
    h = np.arange(11)
    ar = phi ** h
    ma = np.zeros(h.size)
    ma[:2] = [1, -theta]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, values, title, color in zip(
        axes, [ar, ma],
        [r"AR(1): $z_t=0.7z_{t-1}+a_t$", r"MA(1): $z_t=a_t-0.7a_{t-1}$"],
        ["#176087", "#b45420"],
    ):
        ax.axhline(0, color="0.4", linewidth=0.8)
        ax.vlines(h, 0, values, color=color, linewidth=2)
        ax.scatter(h, values, color=color, zorder=3)
        ax.set(title=title, xlabel="Weeks after the surprise, h", xticks=h)
        ax.grid(alpha=0.2)
    axes[0].set(ylabel="Effect on working series (10,000 rolls)", ylim=(-0.85, 1.15))
    fig.suptitle("Constructed zero-mean examples: a single +1 innovation", fontsize=12)
    fig.tight_layout()
    fig.savefig(root / "turn03_zero_mean_surprise.png", dpi=170)
    plt.close(fig)
    result = {
        "description": "Exact constructed impulse responses, not observed or fitted series",
        "fixed_illustrative_coefficients": {"phi_1": phi, "theta_1": theta},
        "setup": "zero before week t; a_t=1; every other innovation zero",
        "ma_convention": "z_t=a_t-theta_1*a_(t-1)",
        "h": h.tolist(), "ar_effect": ar.tolist(), "ma_effect": ma.tolist(),
    }
    (root / "turn03_zero_mean_surprise.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
