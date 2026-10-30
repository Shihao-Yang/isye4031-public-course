"""Deterministic SES weights and illustrative step response; no random seed used.
Step scenario is constructed, not observed cod data.
"""
from pathlib import Path
import csv
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
alphas = [.1, .3, .7]
colors = ['#176b91', '#c15b18', '#7b4397']
with (root / 'cod_weights_turn04.csv').open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['alpha', 'lag', 'observation_weight'])
    for a in alphas:
        for j in range(31):
            w.writerow([a, j, a*(1-a)**j])
with (root / 'cod_step_turn04.csv').open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['alpha', 'number_of_new_observations', 'level_next_forecast', 'remaining_gap'])
    for a in alphas:
        for k in range(21):
            level = 400-50*(1-a)**k
            w.writerow([a, k, level, 400-level])
for a in alphas:
    print(f'alpha={a}: weights lags 0..4={[round(a*(1-a)**j,6) for j in range(5)]}')
    print(f'levels k=1,2,5,10={[round(400-50*(1-a)**k,6) for k in [1,2,5,10]]}; 90% response after {math.ceil(math.log(.1)/math.log(1-a))} observations')
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
for a, color in zip(alphas, colors):
    axes[0].plot(range(21), [a*(1-a)**j for j in range(21)], 'o-',
                 color=color, markersize=3, label=f'α = {a:.2f}')
    axes[1].plot(range(21), [400-50*(1-a)**k for k in range(21)], 'o-',
                 color=color, markersize=3, label=f'α = {a:.2f}')
axes[0].set(title='Weights decay geometrically with age', xlabel='Lag j (0 = latest observation)',
            ylabel='Weight α(1 − α)ʲ', xlim=(-.5, 20.5), ylim=(0, .75))
axes[1].axhline(400, color='black', ls='--', lw=1, label='New catch: 400 tons every month')
axes[1].set(title='Illustration: catch changes from 350 to 400 tons',
            xlabel='Number of observations since change', ylabel='Updated level / next forecast (tons)',
            xlim=(-.5, 20.5), ylim=(345, 410))
for ax in axes:
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
fig.tight_layout(pad=1.8, w_pad=2)
fig.savefig(root / 'cod_weights_response_turn04.png', dpi=170)
plt.close(fig)
