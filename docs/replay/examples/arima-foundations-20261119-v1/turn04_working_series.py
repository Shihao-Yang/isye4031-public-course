import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

base = Path(__file__).resolve().parent
data = np.genfromtxt(base / 'paper_towels.csv', delimiter=',', names=True)
week = data['week'].astype(int)
y = data['sales_above_100000_in_10000_rolls']
d = np.diff(y)
results = {
    'sales_units': 'Sales above 100000 rolls, in units of 10000 rolls',
    'difference_units': 'Change in sales, in units of 10000 rolls',
    'sales_count': int(y.size),
    'difference_count': int(d.size),
    'sales_mean': float(y.mean()),
    'difference_mean': float(d.mean()),
    'sales_sample_sd': float(y.std(ddof=1)),
    'difference_sample_sd': float(d.std(ddof=1)),
    'recommendation': 'Use first differences as a provisional working series; visual stability is not proof of stationarity.',
}
(base / 'turn04_working_series_results.json').write_text(json.dumps(results, indent=2) + '\n')
plt.rcParams.update({'font.size': 16, 'axes.titlesize': 21, 'axes.labelsize': 16})
fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), sharex=True)
axes[0].plot(week, y, color='#176a9a', lw=1.9)
axes[0].set_title('Original weekly sales', pad=16)
axes[0].set_ylabel('Sales above 100,000 rolls\n(units of 10,000 rolls)', labelpad=12)
axes[1].plot(week[1:], d, color='#b45c20', lw=1.6)
axes[1].axhline(0, color='#333333', lw=1)
axes[1].set_title('First differences', pad=16)
axes[1].set_ylabel('Change in weekly sales\n(units of 10,000 rolls)', labelpad=12)
for ax in axes:
    ax.set_xlabel('Week')
    ax.set_xlim(1, 120)
    ax.set_xticks([1, 30, 60, 90, 120])
    ax.grid(alpha=.22)
    ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout(pad=2, w_pad=3)
fig.savefig(base / 'turn04_working_series.png', dpi=160)
plt.close(fig)
print(json.dumps(results, indent=2))
