import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('cod.csv', newline='') as f:
    rows = list(csv.DictReader(f))
t = [int(r['t']) for r in rows]
y = [float(r['y']) for r in rows]
alpha = 0.2
level = [y[0]]  # Constructed choice: initialize at the first observation.
for value in y[1:]:
    level.append(alpha * value + (1-alpha) * level[-1])
results = {
    'alpha_constructed': alpha,
    'initial_level': level[0],
    'level_after_month_23': level[-2],
    'month_24_observation': y[-1],
    'month_24_one_step_fitted_forecast': level[-2],
    'month_24_fitted_error': y[-1] - level[-2],
    'month_24_update': alpha*(y[-1]-level[-2]),
    'level_after_month_24': level[-1],
    'month_25_future_point_forecast': level[-1],
}
with open('recap_cod_ewma_01_results.json', 'w') as f:
    json.dump(results, f, indent=2)
fig, ax = plt.subplots(figsize=(10, 3.7), constrained_layout=True)
ax.plot(t, y, color='#21618c', marker='o', linewidth=1.5, markersize=4)
ax.set(xlabel='Month index (calendar dates unspecified)', ylabel='Cod catch (tons)',
       title='Observed monthly cod catch: 24 months', xlim=(0.5,24.5))
ax.set_xticks(range(1,25))
ax.grid(axis='y', alpha=.22)
ax.spines[['top','right']].set_visible(False)
fig.savefig('recap_cod_ewma_01.png', dpi=170)
print(json.dumps(results, indent=2))
