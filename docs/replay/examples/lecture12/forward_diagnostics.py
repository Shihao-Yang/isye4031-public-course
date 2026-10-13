"""Diagnostics and one-row sensitivity for the fixed forward-selected formula."""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(ROOT / '.mplcache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

data = pd.read_csv(ROOT / 'sales_territory.csv')
names = ['Accts', 'Adver', 'MktPoten', 'MktShare']
def fit(d):
    return sm.OLS(d.Sales, sm.add_constant(d[names], has_constant='add')).fit()
model = fit(data)
infl = model.get_influence()
n, p = len(data), len(model.params)
diagnostics = pd.DataFrame(dict(territory=np.arange(1, n+1), fitted=model.fittedvalues,
    residual=model.resid, internally_studentized=infl.resid_studentized_internal,
    externally_studentized=infl.resid_studentized_external,
    leverage=infl.hat_matrix_diag, cooks_distance=infl.cooks_distance[0]))
diagnostics.to_csv(ROOT / 'forward_diagnostics.csv', index=False)
target = int(np.argmax(infl.cooks_distance[0]))
reduced = fit(data.drop(index=target))
coefficients = pd.DataFrame(dict(all_25=model.params, without_flagged=reduced.params,
    change=reduced.params-model.params, SE_all_25=model.bse, SE_without_flagged=reduced.bse))
coefficients.to_csv(ROOT / 'forward_sensitivity_coefficients.csv', index_label='term')
metrics = pd.DataFrame([dict(fit=label, n=int(m.nobs), R_squared=m.rsquared,
    adjusted_R_squared=m.rsquared_adj, residual_SE=np.sqrt(m.mse_resid), SSE=m.ssr)
    for label, m in [('all_25', model), (f'without_{target+1}', reduced)]])
metrics.to_csv(ROOT / 'forward_sensitivity_metrics.csv', index=False)
all_x = sm.add_constant(data[names], has_constant='add')
predictions = pd.DataFrame(dict(territory=np.arange(1, n+1), observed=data.Sales,
    fitted_all_25=model.predict(all_x), predicted_without_flagged=reduced.predict(all_x)))
predictions.to_csv(ROOT / 'forward_sensitivity_predictions.csv', index=False)

fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
ax = axes[0, 0]
ax.scatter(model.fittedvalues, model.resid, color='#377eb8')
smooth = sm.nonparametric.lowess(model.resid, model.fittedvalues, frac=.7)
ax.plot(smooth[:, 0], smooth[:, 1], color='#d95f02', label='LOWESS')
ax.axhline(0, color='black', linewidth=.8)
ax.annotate(str(target+1), (model.fittedvalues.iloc[target], model.resid.iloc[target]), xytext=(5, 5), textcoords='offset points')
ax.set(xlabel='Fitted Sales', ylabel='Residual (recorded Sales units)', title='Residuals vs. fitted')
ax.legend()
ax = axes[0, 1]
sm.qqplot(infl.resid_studentized_internal, line='45', ax=ax, markerfacecolor='#377eb8', markeredgecolor='#377eb8')
ax.set(title='Normal Q-Q: internally studentized residuals', xlabel='Theoretical normal quantiles', ylabel='Ordered studentized residuals')
ax = axes[1, 0]
ax.stem(diagnostics.territory, diagnostics.leverage, basefmt=' ')
ax.axhline(2*p/n, color='#d95f02', linestyle='--', label=f'2p/n = {2*p/n:.2f}')
for row in diagnostics.itertuples():
    if row.leverage > 2*p/n or row.territory == target+1:
        ax.annotate(str(row.territory), (row.territory, row.leverage), xytext=(4, 4), textcoords='offset points')
ax.set(xlabel='Territory (CSV row order)', ylabel='Leverage', title='Leverage, p = 5 including intercept', ylim=(0, max(.5, diagnostics.leverage.max()*1.2)))
ax.legend()
ax = axes[1, 1]
ax.stem(diagnostics.territory, diagnostics.cooks_distance, basefmt=' ')
ax.axhline(4/n, color='#d95f02', linestyle='--', label=f'4/n = {4/n:.2f}')
for row in diagnostics.itertuples():
    if row.cooks_distance > 4/n or row.territory == target+1:
        ax.annotate(str(row.territory), (row.territory, row.cooks_distance), xytext=(4, 4), textcoords='offset points')
ax.set(xlabel='Territory (CSV row order)', ylabel="Cook's distance", title='Influence on the fitted coefficients')
ax.legend()
fig.suptitle('Fixed forward model: Sales ~ Accts + Adver + MktPoten + MktShare')
fig.savefig(ROOT / 'forward_diagnostics.png', dpi=180)
print(diagnostics.sort_values('cooks_distance', ascending=False).head(6).to_string(index=False))
print('Flagged row:')
print(data.iloc[[target]].to_string(index=False))
print(coefficients.to_string())
print(metrics.to_string(index=False))
print('Flagged-territory predictions:')
print(predictions.iloc[[target]].to_string(index=False))
print('Checks:', np.isfinite(diagnostics.to_numpy()).all(), 'No computational errors.')
