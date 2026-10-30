"""SES from time zero using first-year mean. Deterministic; no seed needed."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
with (root / 'cod.csv').open() as f:
    y = [float(r['y']) for r in csv.DictReader(f)]
alpha = .10
level = sum(y[:12]) / 12
rows = []
for t, actual in enumerate(y, 1):
    previous = level
    error = actual - previous
    adjustment = alpha * error
    level = previous + adjustment
    rows.append([t, actual, previous, error, adjustment, level])
with (root / 'cod_ses_updates_turn03.csv').open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['month', 'actual_tons', 'forecast_or_fitted_value', 'error_actual_minus_forecast', 'adjustment', 'updated_level_next_forecast'])
    w.writerows(rows)
for r in rows:
    print(f'| {r[0]} | {r[1]:.0f} | {r[2]:.2f} | {r[3]:+.2f} | {r[4]:+.2f} | {r[5]:.2f} |')
other = sum(y[:12]) / 12
for actual in y[12:]:
    other = alpha * actual + (1-alpha) * other
print(f'New month-25 forecast={level:.9f}; previous approach={other:.9f}; difference={level-other:.9f}')
fig, ax = plt.subplots(figsize=(11, 5.3))
ax.axvspan(.5, 12.5, color='#d9b86c', alpha=.22,
           label='Months used to choose L₀: early errors are fitted errors')
ax.plot(range(1, 25), y, 'o-', color='#176b91', lw=1.5, label='Observed catch')
ax.plot(range(1, 26), [r[2] for r in rows] + [level], 's-',
        color='#b85416', lw=1.8, markersize=3.5, label='Fitted values (1–12), then one-step forecasts (13–25)')
ax.axvline(12.5, color='gray', lw=1)
ax.set(xlabel='Month index', ylabel='Cod catch (tons)',
       title='SES with α = 0.10 and L₀ = first-year average (360.67 tons)',
       xlim=(.5, 25.8), ylim=(250, 455))
ax.set_xticks([1, 3, 6, 9, 12, 13, 15, 18, 21, 24, 25])
ax.grid(axis='y', alpha=.2)
ax.legend(loc='upper right', fontsize=9)
fig.tight_layout(pad=1.8)
fig.savefig(root / 'cod_ses_turn03.png', dpi=170)
plt.close(fig)
