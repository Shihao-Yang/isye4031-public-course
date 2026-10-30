"""Fit SES alpha by all-24 SSE with fixed time-zero initialization.
Deterministic calculation, no random sampling or seed.
"""
from pathlib import Path
import csv
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
with (root / 'cod.csv').open() as f:
    y = np.array([float(r['y']) for r in csv.DictReader(f)])
initial = float(y[:12].mean())

def smooth(alpha):
    level = initial
    rows = []
    for t, actual in enumerate(y, 1):
        forecast = level
        error = actual - forecast
        level = forecast + alpha * error
        rows.append([t, actual, forecast, error, error**2, level])
    return np.array(rows)

def sse(alpha):
    return float(smooth(alpha)[:, 4].sum())

# A dense scan checks the full interval, then bounded refinement locates
# its best grid neighborhood. Explicitly compare both endpoints.
grid = np.linspace(0, 1, 10001)
losses = np.array([sse(a) for a in grid])
index = int(losses.argmin())
lower, upper = grid[max(0, index-1)], grid[min(len(grid)-1, index+1)]
opt = minimize_scalar(sse, bounds=(lower, upper), method='bounded',
                      options={'xatol': 1e-14})
best = min([0., 1., float(grid[index]), float(opt.x)], key=sse)
rows = smooth(best)
forecast = rows[-1, 5]
print(f'L0={initial:.12f}; sum first year={y[:12].sum():.0f}')
print(f'alpha={best:.12f}; SSE={sse(best):.12f}; MSE={sse(best)/24:.12f}; RMSE={np.sqrt(sse(best)/24):.12f}; forecast={forecast:.12f}')
for a in [0, .1, best, .3, .5, 1]:
    r = smooth(a)
    print(f'alpha={a:.12f}; SSE={sse(a):.9f}; RMSE={np.sqrt(sse(a)/24):.9f}; next={r[-1,5]:.9f}')
for row in rows[:3]:
    print('First updates:', row.tolist())
with (root / 'cod_fit_updates_turn05.csv').open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['month', 'catch_tons', 'fitted_value', 'error', 'squared_error', 'updated_level'])
    w.writerows(rows)
with (root / 'cod_alpha_curve_turn05.csv').open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['alpha', 'SSE'])
    w.writerows(zip(grid, losses))
with (root / 'cod_forecast_turn05.csv').open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['month', 'forecast_tons', 'alpha', 'initial_level'])
    w.writerows([[t, forecast, best, initial] for t in range(25, 31)])

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
axes[0].plot(grid, losses, color='#176b91', lw=2)
axes[0].scatter([best], [sse(best)], color='#b85416', zorder=3)
axes[0].annotate(f'Minimum at α = {best:.4f}', xy=(best, sse(best)),
                 xytext=(.22, .75), textcoords='axes fraction',
                 arrowprops={'arrowstyle': '->'})
axes[0].set(xlabel='Level smoothing parameter α', ylabel='Sum of squared errors (tons²)',
            title='Historical fitting criterion: all 24 months', xlim=(0, 1))
axes[1].axvspan(.5, 12.5, color='gray', alpha=.10, label='First year used in L₀')
axes[1].plot(range(1, 25), y, 'o-', color='#176b91', lw=1.3, markersize=3.5, label='Observed catch')
axes[1].plot(range(1, 25), rows[:, 2], color='#b85416', lw=1.8, label=f'Fitted values, α = {best:.4f}')
axes[1].plot(range(25, 31), [forecast]*6, 's--', color='#7b4397', markersize=3.5,
             label=f'Future point forecast: {forecast:.2f} tons')
axes[1].axvline(24.5, color='gray', lw=1)
axes[1].set(title='Fitted SES and resulting forecast', xlabel='Month index',
            ylabel='Cod catch (tons)', xlim=(.5, 31), ylim=(250, 460))
axes[1].legend(loc='upper right', fontsize=8)
for ax in axes:
    ax.grid(alpha=.2)
fig.tight_layout(pad=1.8, w_pad=2)
fig.savefig(root / 'cod_alpha_fit_turn05.png', dpi=170)
plt.close(fig)
