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
rolls = 100000 + 10000 * y
results = {
    'observations': int(len(week)),
    'first_week': int(week[0]),
    'last_week': int(week[-1]),
    'calendar_dates': 'unspecified',
    'units_conversion': 'actual rolls = 100000 + 10000 * recorded sales',
    'first_week_rolls': float(rolls[0]),
    'last_week_rolls': float(rolls[-1]),
    'minimum_weekly_rolls': float(rolls.min()),
    'maximum_weekly_rolls': float(rolls.max()),
    'prediction_target': 'Sales in week 121 and subsequent weeks',
    'future_observed_values_supplied': False,
}
(base / 'turn01_paper_towel_results.json').write_text(json.dumps(results, indent=2) + '\n')
plt.rcParams.update({'font.size': 16, 'axes.titlesize': 21, 'axes.labelsize': 17})
fig, ax = plt.subplots(figsize=(12, 5.8))
ax.plot(week, rolls / 1000, color='#176a9a', linewidth=2, marker='o', markersize=3)
ax.set(title='Paper-towel sales: 120 observed weeks', xlabel='Week', ylabel='Weekly sales (thousands of rolls)')
ax.set_xlim(1, 122)
ax.set_xticks([1, 20, 40, 60, 80, 100, 120])
ax.grid(alpha=.23)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout(pad=1.8)
fig.savefig(base / 'turn01_paper_towel_data.png', dpi=160)
plt.close(fig)
print(json.dumps(results, indent=2))
