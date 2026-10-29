from pathlib import Path
import os
os.environ['MPLCONFIGDIR'] = str(Path.cwd() / '.mplcache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

raw = pd.read_csv('store_sales_01_data.csv')
d = raw.loc[raw['split'].eq('training')].reset_index(drop=True)
cols = ['size_sqft', 'advertising_thousand_usd_month', 'staff_count',
        'area_household_income_thousand_usd_year', 'nearby_competitor_count']
X = d[cols].to_numpy()
y = d['sales_thousand_usd_month'].to_numpy()
# Center at a fixed $28,000/month so no full-data preprocessing enters CV.
def features(x, curved):
    z = x.copy()
    z[:, 1] -= 28
    return np.column_stack([z, z[:, 1]**2]) if curved else z
splits = list(KFold(n_splits=5, shuffle=True, random_state=403118).split(X))
fold_id = np.zeros(len(d), dtype=int)
for k, (_, va) in enumerate(splits, 1):
    fold_id[va] = k
pd.DataFrame({'store_id': d.store_id, 'validation_fold':fold_id}).to_csv('store_sales_02_shared_five_folds.csv', index=False)
rows, predictions, models = [], {}, {}
for name, curved in [('Linear advertising', False), ('Quadratic advertising', True)]:
    design = features(X, curved)
    oof = np.empty(len(y))
    for k, (tr, va) in enumerate(splits, 1):
        model = make_pipeline(StandardScaler(), LinearRegression())
        model.fit(design[tr], y[tr])
        oof[va] = model.predict(design[va])
        rows.append({'model':name, 'fold':k, 'validation_rmse':float(np.sqrt(np.mean((y[va]-oof[va])**2)))})
    model = make_pipeline(StandardScaler(), LinearRegression()).fit(design, y)
    predictions[name] = oof
    models[name] = model
    rows.append({'model':name, 'fold':'pooled', 'validation_rmse':float(np.sqrt(np.mean((y-oof)**2))),
                 'training_rmse':float(np.sqrt(np.mean((y-model.predict(design))**2)))})
results = pd.DataFrame(rows)
results.to_csv('store_sales_02_curvature_metrics.csv', index=False)
out = pd.DataFrame({'store_id':d.store_id, 'fold':fold_id, 'advertising':X[:,1], 'actual_sales':y})
for name, pred in predictions.items(): out[name] = pred
out.to_csv('store_sales_02_validation_predictions.csv', index=False)
coef_model = models['Quadratic advertising']
coef = coef_model[-1].coef_ / coef_model[0].scale_
intercept = coef_model[-1].intercept_ - coef @ coef_model[0].mean_
summary = {'quadratic_coefficients':dict(zip(cols + ['centered_advertising_squared'], coef.tolist())),
           'intercept':float(intercept), 'advertising_center':28,
           'validation_rmse_reduction_percent':float(100*(1-np.sqrt(np.mean((y-predictions['Quadratic advertising'])**2))/np.sqrt(np.mean((y-predictions['Linear advertising'])**2)))),
           'test_evaluated':False}
Path('store_sales_02_curvature_results.json').write_text(json.dumps(summary, indent=2))
plt.rcParams.update({'font.size':11, 'axes.labelsize':12, 'axes.titlesize':13})
fig, axes = plt.subplots(1,2,figsize=(12,4.9))
grid = np.linspace(X[:,1].min(), X[:,1].max(), 160)
reference = np.tile(np.median(X, axis=0), (len(grid),1))
reference[:,1] = grid
colors = ['#2166ac','#c45118']
for (name, curved), color in zip([('Linear advertising',False),('Quadratic advertising',True)], colors):
    axes[0].plot(grid, models[name].predict(features(reference,curved)), label=name, color=color, lw=2.5)
    bins = pd.qcut(X[:,1], 6, labels=False, duplicates='drop')
    r = pd.DataFrame({'ad':X[:,1], 'residual':y-predictions[name], 'bin':bins}).groupby('bin').mean()
    axes[1].plot(r.ad, r.residual, 'o-', color=color, lw=2, label=name)
axes[0].set(title='Fitted advertising relationship', xlabel='Monthly advertising ($1,000)', ylabel='Predicted monthly sales ($1,000)')
axes[0].text(.04,.96,'Other predictors held at training medians',transform=axes[0].transAxes,va='top',fontsize=10)
axes[1].axhline(0,color='gray',lw=1,ls=':')
axes[1].set(title='Validation residuals: six advertising groups',xlabel='Mean monthly advertising ($1,000)',ylabel='Mean actual minus predicted sales ($1,000)')
for ax in axes:
    ax.grid(alpha=.2)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='lower right', fontsize=10)
fig.tight_layout()
fig.savefig('store_sales_02_advertising_curvature.png', dpi=180)
print(results.to_string(index=False))
print(json.dumps(summary, indent=2))
