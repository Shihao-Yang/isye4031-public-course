import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

base = Path(__file__).resolve().parent
data = np.genfromtxt(base / 'paper_towels.csv', delimiter=',', names=True)
weeks = data['week'].astype(int)
sales = 100000 + 10000 * data['sales_above_100000_in_10000_rolls']
first = np.diff(sales)
second = np.diff(first)
result = {
    'units': 'rolls',
    'definitions': {'first': 'sales[t] - sales[t-1]', 'second': 'first[t] - first[t-1]'},
    'first_three_sales': sales[:3].tolist(),
    'first_difference_week_2': float(first[0]),
    'first_difference_week_3': float(first[1]),
    'second_difference_week_3': float(second[0]),
    'original_count': int(len(sales)),
    'first_difference_count': int(len(first)),
    'second_difference_count': int(len(second)),
    'first_difference_mean': float(np.mean(first)),
    'second_difference_mean': float(np.mean(second)),
    'first_difference_sample_sd': float(np.std(first, ddof=1)),
    'second_difference_sample_sd': float(np.std(second, ddof=1)),
}
(base / 'turn03_sales_differences_results.json').write_text(json.dumps(result, indent=2) + '\n')
np.savetxt(base / 'turn03_sales_differences.csv', np.column_stack([weeks, sales, np.r_[np.nan, first], np.r_[np.nan, np.nan, second]]), delimiter=',', header='week,sales_rolls,first_difference_rolls,second_difference_rolls', comments='', fmt=['%d','%.4f','%.4f','%.4f'])
plt.rcParams.update({'font.size': 16, 'axes.titlesize': 20, 'axes.labelsize': 17})
fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, sharey=True)
for ax, values, time, title, color in [
    (axes[0], first, weeks[1:], 'First differences: change in weekly sales', '#176a9a'),
    (axes[1], second, weeks[2:], 'Second differences: change in the weekly change', '#b45c20'),
]:
    ax.plot(time, values / 1000, color=color, linewidth=1.6, marker='o', markersize=2.8)
    ax.axhline(0, color='#333333', linewidth=1)
    ax.set_title(title, pad=12)
    ax.set_ylabel('Thousands of rolls')
    ax.grid(alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)
axes[-1].set_xlabel('Week (ending week of each difference)')
axes[-1].set_xticks([1, 20, 40, 60, 80, 100, 120])
axes[-1].set_xlim(1, 121)
fig.tight_layout(pad=2, h_pad=2.3)
fig.savefig(base / 'turn03_sales_differences.png', dpi=160)
plt.close(fig)
print(json.dumps(result, indent=2))
