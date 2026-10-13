"""LOOCV of selection procedures versus formulas selected before CV."""
from pathlib import Path
import os
import json
from collections import Counter
ROOT = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(ROOT / '.mplcache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

data = pd.read_csv(ROOT / 'sales_territory.csv')
predictors = list(data.columns.drop('Sales'))
assert len(data) == 25 and np.isfinite(data.to_numpy()).all()

def fit(frame, names):
    return sm.OLS(frame.Sales, sm.add_constant(frame[names], has_constant='add')).fit()

def select(frame, method):
    if method == 'Full':
        return predictors.copy()
    names = [] if method == 'Forward' else predictors.copy()
    while True:
        if method == 'Forward':
            remaining = [x for x in predictors if x not in names]
            if not remaining:
                return names
            pvalues = {x: float(fit(frame, names+[x]).pvalues[x]) for x in remaining}
            best = min(pvalues, key=pvalues.get)
            if pvalues[best] >= 0.10:
                return names
            names.append(best)
        else:
            if not names:
                return names
            pvalues = fit(frame, names).pvalues.drop('const')
            worst = pvalues.idxmax()
            if pvalues[worst] <= 0.10:
                return names
            names.remove(worst)

fixed = {m: select(data, m) for m in ['Full', 'Forward', 'Backward']}
rows = []
for i in range(len(data)):
    training = data.drop(index=i)
    heldout = data.iloc[[i]]
    for method in fixed:
        for calculation in ['Reselect on 24', 'Select on all 25']:
            names = select(training, method) if calculation == 'Reselect on 24' else fixed[method]
            model = fit(training, names)
            prediction = float(model.predict(sm.add_constant(heldout[names], has_constant='add')).iloc[0])
            error = float(heldout.Sales.iloc[0])-prediction
            rows.append(dict(territory=i+1, method=method, calculation=calculation,
                             observed=float(heldout.Sales.iloc[0]), predicted=prediction,
                             error=error, squared_error=error**2, absolute_error=abs(error),
                             predictors=' + '.join(sorted(names))))
results = pd.DataFrame(rows)
results.to_csv(ROOT / 'loo_predictions.csv', index=False)
metrics = results.groupby(['method', 'calculation']).agg(
    PRESS=('squared_error', 'sum'), MSE=('squared_error', 'mean'), MAE=('absolute_error', 'mean')).reset_index()
metrics['RMSE'] = np.sqrt(metrics.MSE)
metrics.to_csv(ROOT / 'loo_metrics.csv', index=False)
frequency = (results[results.calculation == 'Reselect on 24']
             .groupby(['method', 'predictors']).size().reset_index(name='fold_count'))
frequency.to_csv(ROOT / 'loo_selected_formulas.csv', index=False)
# The prespecified full model must give identical predictions in both calculations.
full = results[results.method == 'Full'].pivot(index='territory', columns='calculation', values='predicted')
assert np.allclose(full.iloc[:, 0], full.iloc[:, 1])
# Independently verify full-model LOOCV residuals using the hat matrix identity.
model = fit(data, predictors)
assert np.allclose(model.resid / (1-model.get_influence().hat_matrix_diag),
                   results[(results.method == 'Full') & (results.calculation == 'Reselect on 24')].error)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout='constrained')
methods = ['Full', 'Forward', 'Backward']
colors = ['#4575b4', '#d95f02', '#1b9e77']
for j, calc in enumerate(['Reselect on 24', 'Select on all 25']):
    values = [float(metrics[(metrics.method == m) & (metrics.calculation == calc)].RMSE.iloc[0]) for m in methods]
    bars = axes[0].bar(np.arange(3)+(j-.5)*.36, values, width=.36,
                      label='Selection inside each fold' if j == 0 else 'Formula selected on all rows',
                      color='#377eb8' if j == 0 else '#bbbbbb')
    axes[0].bar_label(bars, fmt='%.1f', padding=3)
axes[0].set(xticks=range(3), xticklabels=methods, ylabel='LOOCV RMSE (recorded Sales units)',
            title='Correct procedure check vs. leaked selection')
axes[0].set_ylim(0, metrics.RMSE.max()*1.28)
axes[0].legend(fontsize=8, loc='upper left')
for method, color in zip(methods, colors):
    part = results[(results.method == method) & (results.calculation == 'Reselect on 24')]
    axes[1].plot(part.territory, part.error, marker='o', markersize=4, linewidth=1,
                 label=method, color=color, alpha=.85)
axes[1].axhline(0, color='black', linewidth=.8)
axes[1].set(xlabel='Held-out territory (CSV row order)', ylabel='Observed minus predicted Sales',
            title='Errors with selection inside each fold', xticks=[1, 5, 10, 15, 20, 25])
axes[1].legend(fontsize=8)
fig.savefig(ROOT / 'loo_comparison.png', dpi=180)
print(metrics.to_string(index=False))
print(frequency.to_string(index=False))
print('Checks passed; no computational errors.')
