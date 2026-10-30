"""SES initialized from months 1–12. Deterministic; random seed not used."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
with (root / 'cod.csv').open() as f:
    data = list(csv.DictReader(f))
y = [float(r['y']) for r in data]
alpha = 0.10
level = sum(y[:12]) / 12
print(f'First-year total={sum(y[:12]):.0f}; initial level={level:.9f}')
updates = []
for t in range(13, 25):
    forecast = level
    error = y[t-1] - forecast
    level = forecast + alpha * error
    updates.append([t, y[t-1], forecast, error, alpha * error, level])
    print(f'{t}: catch={y[t-1]:.0f}, forecast={forecast:.6f}, error={error:.6f}, adjustment={alpha*error:.6f}, next forecast={level:.6f}')
with (root / 'cod_ses_updates_turn02.csv').open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['month', 'actual_tons', 'forecast_before_observation', 'error_actual_minus_forecast', 'adjustment', 'forecast_next_month'])
    w.writerows(updates)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, 25), y, 'o-', color='#176b91', lw=1.5, label='Observed catch')
ax.plot(range(13, 26), [u[2] for u in updates] + [level], 's-',
        color='#c15b18', lw=2, markersize=4, label='One-step forecast, α = 0.10')
ax.axvspan(.5, 12.5, color='gray', alpha=.10, label='Months 1–12: initialize level')
ax.annotate(f'Month 25 forecast: {level:.2f}', xy=(25, level),
            xytext=(17, 285), arrowprops={'arrowstyle': '->', 'color': '#c15b18'},
            color='#93420c')
ax.set(xlabel='Month index', ylabel='Cod catch (tons)',
       title='Simple exponential smoothing: forecasts update after each catch',
       xlim=(.5, 26), ylim=(250, 440))
ax.set_xticks([1, 6, 12, 13, 18, 24, 25])
ax.grid(axis='y', alpha=.2)
ax.legend(loc='upper right')
fig.tight_layout(pad=1.8)
fig.savefig(root / 'cod_ses_turn02.png', dpi=170)
plt.close(fig)
