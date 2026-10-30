"""Conditional normal ETS(A,N,N) prediction intervals; deterministic, no seed.
Month 25 = 384 is supplied by the instructor, not part of cod.csv.
Alpha and innovation variance are kept fixed after the new observation.
"""
from pathlib import Path
import csv
from statistics import NormalDist
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
with (root / 'cod_forecast_turn05.csv').open() as f:
    fit = next(csv.DictReader(f))
alpha = float(fit['alpha'])
level24 = float(fit['forecast_tons'])
with (root / 'cod_fit_updates_turn05.csv').open() as f:
    fitted = list(csv.DictReader(f))
sse = sum(float(r['squared_error']) for r in fitted)
n = len(fitted)
# Approximate degrees-of-freedom adjustment for the one fitted alpha.
# L0 is treated as fixed, despite having been estimated from year 1.
s2 = sse / (n-1)
s = np.sqrt(s2)
z = NormalDist().inv_cdf(.975)
actual25 = 384.
error25 = actual25-level24
level25 = level24+alpha*error25
print(f'alpha={alpha:.12f}; SSE={sse:.9f}; s2={s2:.9f}; s={s:.9f}; z={z:.9f}')
print(f'month25 error={error25:.9f}; adjustment={alpha*error25:.9f}; L25={level25:.9f}')
intervals = []
for origin, level in [(24, level24), (25, level25)]:
    for h in range(1, 13):
        se = s*np.sqrt(1+(h-1)*alpha**2)
        lo, hi = level-z*se, level+z*se
        intervals.append([origin, origin+h, h, level, se, lo, hi])
        if h in [1, 2, 6, 12]:
            print(f'origin={origin}, target={origin+h}, h={h}, forecast={level:.6f}, SE={se:.6f}, 95%=[{lo:.6f}, {hi:.6f}]')
with (root / 'cod_intervals_turn06.csv').open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['forecast_origin', 'target_month', 'horizon', 'point_forecast_tons', 'prediction_SE_tons', 'lower_95_tons', 'upper_95_tons'])
    w.writerows(intervals)
with (root / 'cod_update_turn06.csv').open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['alpha', 'L24', 'month25_catch', 'month25_error', 'adjustment', 'L25', 'fixed_innovation_variance'])
    w.writerow([alpha, level24, actual25, error25, alpha*error25, level25, s2])
fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.5), sharey=True)
historical = [float(r['catch_tons']) for r in fitted]
for ax, origin, level in zip(axes, [24, 25], [level24, level25]):
    rs = np.array([r for r in intervals if r[0] == origin])
    ax.plot(range(19, 25), historical[18:], 'o-', color='#176b91', label='Observed catch')
    ax.fill_between(rs[:, 1], rs[:, 5], rs[:, 6], color='#b85416', alpha=.18,
                    label='Approximate 95% prediction interval')
    ax.plot(rs[:, 1], rs[:, 3], '--', color='#b85416', lw=2,
            label=f'Point forecast: {level:.2f} tons')
    ax.axvline(origin+.5, color='gray', lw=1)
    if origin == 24:
        ax.scatter([25], [actual25], marker='*', s=120, color='#7b4397', zorder=4,
                   label='Next catch: 384 (revealed afterward)')
    else:
        ax.plot([24, 25], [historical[-1], actual25], 'o-', color='#176b91')
        ax.scatter([25], [actual25], marker='*', s=120, color='#7b4397', zorder=4,
                   label='Newly observed catch: 384')
        ax.plot([26, 37], [level24]*2, ':', color='gray', lw=1.5,
                label=f'Previous forecast: {level24:.2f}')
    ax.set(title=f'Information through month {origin}', xlabel='Month',
           xlim=(18.5, 37.5), ylim=(270, 445))
    ax.set_xticks([19, 22, 25, 28, 31, 34, 37])
    ax.grid(axis='y', alpha=.2)
    ax.legend(loc='upper right', fontsize=7.8)
axes[0].set_ylabel('Cod catch (tons)')
fig.suptitle('Prediction uncertainty and the update after a catch of 384 tons', fontsize=13)
fig.tight_layout(pad=1.8)
fig.savefig(root / 'cod_intervals_update_turn06.png', dpi=170)
plt.close(fig)
