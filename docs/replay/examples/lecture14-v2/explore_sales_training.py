"""Training-only descriptive exploration of constructed sales data. No fitted model."""
import os
os.environ['MPLCONFIGDIR'] = '.mplcache'
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('constructed_sales_seed4031.csv')
split = pd.read_csv('fixed_sales_split.csv')
train = data.loc[split['split'].eq('train')].copy()
cols = ['store_id', 'digital_ad_spend', 'floor_area', 'local_demand_index', 'weekly_sales']
train[cols].head().to_csv('sales_training_preview.csv', index=False)
predictors = [c for c in train if c not in ['store_id', 'weekly_sales']]
corr = train[predictors].corr()
corr.to_csv('sales_training_predictor_correlations.csv')
summary = {
    'dataset': 'Original constructed fictional store data, seed 4031',
    'scope': '168 training rows only; descriptive exploration, no fitted model',
    'sales_summary_dollars': train.weekly_sales.describe().to_dict(),
    'sales_pearson_correlations': train[predictors].corrwith(train.weekly_sales).to_dict(),
    'preview': train[cols].head().round(1).to_dict(orient='records'),
}
with open('sales_training_exploration_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

plt.rcParams.update({'font.size': 10})
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
scatter_specs = [
    ('digital_ad_spend', 'Digital advertising ($/week)'),
    ('floor_area', 'Floor area (square feet)'),
    ('local_demand_index', 'Local demand (index points)'),
    ('print_ad_spend', 'Print advertising ($/week)'),
    ('survey_noise_index', 'Survey noise (arbitrary index points)'),
]
for ax, (col, label) in zip(axes.flat, scatter_specs):
    ax.scatter(train[col], train.weekly_sales / 1000, s=24, alpha=.7, edgecolors='none')
    ax.set(xlabel=label, ylabel='Weekly sales ($ thousands)')
    r = train[col].corr(train.weekly_sales)
    ax.set_title(f'Pearson correlation with sales: {r:.2f}')
    ax.grid(alpha=.18)
ax = axes.flat[-1]
short = ['Digital', 'Print', 'Promo', 'Area', 'Hours', 'Shelves', 'Demand', 'Footfall', 'Activity', 'Code', 'Color', 'Noise']
im = ax.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r')
ax.set_xticks(np.arange(len(short)), short, rotation=70, ha='right', fontsize=8)
ax.set_yticks(np.arange(len(short)), short, fontsize=8)
ax.set_title('Correlations among candidate predictors')
fig.colorbar(im, ax=ax, fraction=.045, pad=.025, ticks=[-1, 0, 1])
fig.suptitle('Constructed weekly store sales: training data only (168 stores)', fontsize=16, y=.98)
fig.subplots_adjust(left=.065, right=.98, top=.90, bottom=.14, hspace=.5, wspace=.32)
fig.text(.065, .035, 'Original synthetic data, seed 4031. Each dot is one fictional store. The 72 reserved test rows are unused.\nDescriptive associations only; no fitted model or prediction accuracy estimate is shown.', fontsize=11)
fig.savefig('constructed_sales_training_relationships.png', dpi=160)
plt.close(fig)
print(json.dumps(summary, indent=2))
