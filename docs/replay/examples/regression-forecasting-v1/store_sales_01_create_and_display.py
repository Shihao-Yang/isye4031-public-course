from pathlib import Path
import os
os.environ['MPLCONFIGDIR'] = str(Path.cwd() / '.mplcache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

seed = 403117
rng = np.random.default_rng(seed)
n = 300
size = np.clip(rng.normal(2500, 700, n), 800, 5000)
ad = np.clip(28 + 0.008 * (size - 2500) + rng.normal(0, 8, n), 5, 65)
staff = np.clip(np.rint(10 + 0.003 * (size - 2500) + rng.normal(0, 2, n)), 3, 22).astype(int)
income = np.clip(rng.normal(65, 15, n), 25, 110)
competitors = rng.poisson(3, n)
# Fix the holdout before generating or inspecting outcomes.
split_rng = np.random.default_rng(seed + 1)
test_ids = split_rng.permutation(n)[:75]
role = np.full(n, 'training', dtype=object)
role[test_ids] = 'test'
noise_rng = np.random.default_rng(seed + 2)
sales = (40 + 0.055 * size + 3 * ad - 0.025 * ad**2
         + 2.5 * staff + income - 8 * competitors + noise_rng.normal(0, 35, n))
df = pd.DataFrame({'store_id': np.arange(1, n + 1), 'size_sqft': size,
    'advertising_thousand_usd_month': ad, 'staff_count': staff,
    'area_household_income_thousand_usd_year': income, 'nearby_competitor_count': competitors,
    'sales_thousand_usd_month': sales, 'split': role})
df.to_csv('store_sales_01_data.csv', index=False)
train = df.loc[df['split'] == 'training'].copy()
preview = train.head(6).copy()
preview.to_csv('store_sales_01_training_preview.csv', index=False)
metadata = {'data_origin': 'Original constructed data, not observed business records',
 'seed': seed, 'rows': n, 'training_rows': len(train), 'reserved_test_rows': len(test_ids),
 'split_seed': seed + 1, 'noise_seed': seed + 2,
 'mechanism': 'sales (thousand USD/month) = 40 + .055*size_sqft + 3*advertising - .025*advertising^2 + 2.5*staff + area_income - 8*competitors + independent N(0,35^2) noise. Advertising is in thousand USD/month and income in thousand USD/year.',
 'predictor_generation': 'size=clip(N(2500,700^2),800,5000); advertising=clip(28+.008*(size-2500)+N(0,8^2),5,65); staff=clip(round(10+.003*(size-2500)+N(0,2^2)),3,22); income=clip(N(65,15^2),25,110); competitors=Poisson(3). Independent underlying random draws across stores.',
 'evaluation': 'Test roles fixed before generating outcomes. Only training outcomes displayed or summarized. Test set reserved for later final evaluation.'}
Path('store_sales_01_design.json').write_text(json.dumps(metadata, indent=2))
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})
fig, ax = plt.subplots(figsize=(8.5, 4.8))
ax.scatter(train['advertising_thousand_usd_month'], train['sales_thousand_usd_month'],
           s=32, alpha=.72, color='#2166ac', edgecolors='white', linewidths=.35)
ax.set(xlabel='Monthly advertising ($1,000)', ylabel='Monthly sales ($1,000)',
       title='Constructed store-sales data: 225 training stores')
ax.grid(alpha=.2)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
fig.savefig('store_sales_01_training_scatter.png', dpi=180)
print(preview.to_string(index=False, float_format=lambda x: f'{x:.1f}'))
print(json.dumps({'stores':n, 'training':len(train), 'reserved_test':len(test_ids)}))
