"""Historical fit versus chronological holdout. Deterministic; no seed used."""
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
initial = y[:12].mean()
def predictions(a, values):
    level = initial
    pred = []
    for value in values:
        pred.append(level)
        level += a*(value-level)
    return np.array(pred), level
def metrics(actual, pred):
    e = actual-pred
    return [np.mean(abs(e)), np.mean(e**2), 100*np.mean(abs(e)/actual)]
def optimize(values, metric_index):
    def loss(a):
        return metrics(values, predictions(a, values)[0])[metric_index]
    grid = np.linspace(0, 1, 10001)
    i = min(range(len(grid)), key=lambda j: loss(grid[j]))
    lo, hi = grid[max(0, i-1)], grid[min(len(grid)-1, i+1)]
    fit = minimize_scalar(loss, bounds=(lo, hi), method='bounded', options={'xatol': 1e-13})
    return min([0., 1., grid[i], fit.x], key=loss)
all_a = [optimize(y, j) for j in range(3)]
train_a = optimize(y[:12], 1)
print('ALL24 optima by MAD, MSD, MAPE:', all_a)
fitting = []
for name, a in [('alpha=0', 0), ('alpha=.10', .1), ('min MAD', all_a[0]), ('min MSD', all_a[1]), ('min MAPE', all_a[2])]:
    p, level = predictions(a, y)
    m = metrics(y, p)
    fitting.append([name, a, *m, level])
    print('FIT', fitting[-1])
print('TRAIN first12 MSD alpha:', train_a)
for a in [train_a, .1, .3, 1.]:
    print('TRAIN metrics', a, metrics(y[:12], predictions(a, y[:12])[0]))
# Fit the procedure on months 1–12, then evaluate rolling one-step forecasts
# on 13–24, allowing each outcome to update the next forecast.
holdout = []
forecast_rows = []
procedures = [('SES: alpha fit on first year', train_a), ('SES: fixed alpha=.10', .1),
              ('SES: fixed alpha=.30', .3), ('SES: fixed alpha=1 (last catch)', 1.)]
preds = {}
for name, a in procedures:
    p, final = predictions(a, y)
    p = p[12:]
    preds[name] = p
    m = metrics(y[12:], p)
    holdout.append([name, a, *m])
    print('HOLDOUT', holdout[-1])
    forecast_rows.extend([[name, t, a, actual, pred, actual-pred] for t, actual, pred in zip(range(13,25), y[12:], p)])
for filename, header, rows in [
    ('cod_fitting_metrics_turn07.csv', ['method','alpha','MAD_tons','MSD_tons_squared','MAPE_percent','month25_forecast'], fitting),
    ('cod_holdout_metrics_turn07.csv', ['method','alpha','MAD_tons','MSD_tons_squared','MAPE_percent'], holdout),
    ('cod_holdout_forecasts_turn07.csv', ['method','month','alpha','actual','forecast','error'], forecast_rows)]:
    with (root / filename).open('w', newline='') as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
print('Month13 fit error example:', metrics(y[12:13], predictions(all_a[1], y)[0][12:13]))
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
axes[0].plot(range(13,25), y[12:], 'o-', color='#176b91', label='Observed catch')
for name, color, label in [(procedures[0][0], '#7b4397', f'Train-selected SES: α = {train_a:.3f}'),
                            (procedures[1][0], '#b85416', 'Fixed α = 0.10')]:
    axes[0].plot(range(13,25), preds[name], 's--', markersize=3, color=color, label=label)
axes[0].set(title='Second-year rolling one-step forecasts', xlabel='Month', ylabel='Cod catch (tons)', ylim=(250,440))
axes[0].legend(fontsize=8)
labels = ['Train-selected α', 'Fixed α = .10', 'Fixed α = .30', 'Last catch (α = 1)']
axes[1].bar(range(4), [r[3] for r in holdout], color=['#7b4397','#b85416','#ad8751','#176b91'])
axes[1].set_xticks(range(4), labels, rotation=18, ha='right', fontsize=8)
axes[1].set(title='Second-year MSD: smaller is better', ylabel='Mean squared error (tons²)')
for ax in axes: ax.grid(axis='y', alpha=.2)
fig.tight_layout(pad=1.8, w_pad=2)
fig.savefig(root / 'cod_accuracy_turn07.png', dpi=170)
plt.close(fig)
