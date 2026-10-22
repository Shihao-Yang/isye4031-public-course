"""Construct original fictional weekly store sales data, seed 4031."""
import os
os.environ['MPLCONFIGDIR'] = '.mplcache'
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 4031
rng = np.random.default_rng(SEED)
n = 240
ids = np.arange(1, n + 1)
# Fix membership independently of predictors and outcomes, before generating them.
train_idx, test_idx = train_test_split(np.arange(n), test_size=.30, random_state=SEED)
marketing, size, demand = rng.normal(size=(3, n))
def jitter(scale=.45):
    return rng.normal(0, scale, n)

df = pd.DataFrame({
    'store_id': ids,
    'digital_ad_spend': 1500 + 300 * (marketing + jitter()),
    'print_ad_spend': 800 + 160 * (marketing + jitter()),
    'promotion_index': 50 + 10 * (marketing + jitter()),
    'floor_area': 2000 + 400 * (size + jitter()),
    'staff_hours': 300 + 60 * (size + jitter()),
    'shelf_capacity_index': 50 + 10 * (size + jitter()),
    'local_demand_index': 50 + 10 * (demand + jitter()),
    'footfall_index': 50 + 10 * (demand + jitter()),
    'neighborhood_activity_index': 50 + 10 * (demand + jitter()),
    'store_code_index': rng.uniform(0, 100, n),
    'display_color_index': rng.uniform(0, 100, n),
    'survey_noise_index': rng.uniform(0, 100, n),
})
noise = rng.normal(0, 3000, n)
df['weekly_sales'] = (30000 + 4 * (df.digital_ad_spend - 1500)
    + 70 * (df.promotion_index - 50) + 7 * (df.floor_area - 2000)
    + 8 * (df.staff_hours - 300) + 180 * (df.local_demand_index - 50) + noise)
df.to_csv('constructed_sales_seed4031.csv', index=False, float_format='%.6f')
split = np.full(n, 'test', dtype=object)
split[train_idx] = 'train'
pd.DataFrame({'store_id': ids, 'split': split}).to_csv('fixed_sales_split.csv', index=False)
Path('sales_generation_metadata.json').write_text(json.dumps({
    'constructed': True, 'seed': SEED, 'n': n, 'training_rows': len(train_idx),
    'reserved_test_rows': len(test_idx),
    'mechanism': 'Three independent standard-normal latent factors induce correlation within marketing, size, and demand groups. Each grouped predictor adds independent N(0, 0.45^2) noise before rescaling. Three independent Uniform(0,100) inputs have zero direct coefficients.',
    'outcome_equation': 'weekly_sales = 30000 + 4*(digital_ad_spend-1500) + 70*(promotion_index-50) + 7*(floor_area-2000) + 8*(staff_hours-300) + 180*(local_demand_index-50) + epsilon',
    'epsilon': 'Independent normal noise, mean 0, standard deviation 3000 dollars',
    'split': 'sklearn train_test_split row indices, test_size=0.30, random_state=4031; fixed before generating predictors or outcomes'
}, indent=2))
