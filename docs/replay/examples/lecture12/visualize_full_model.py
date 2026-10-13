from pathlib import Path
import os
import json
ROOT = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(ROOT / '.mplcache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr

df = pd.read_csv(ROOT / 'sales_territory.csv')
predictors = ['Time', 'MktPoten', 'Adver', 'MktShare', 'Change', 'Accts', 'WkLoad', 'Rating']
full = sm.OLS(df.Sales, sm.add_constant(df[predictors])).fit()
simple = sm.OLS(df.Sales, sm.add_constant(df[['Accts']])).fit()
others = [c for c in predictors if c != 'Accts']
Z = sm.add_constant(df[others])
ry = sm.OLS(df.Sales, Z).fit().resid
ax_model = sm.OLS(df.Accts, Z).fit()
rx = ax_model.resid
reduced = sm.OLS(df.Sales, Z).fit()
partial_r = pearsonr(rx, ry).statistic
metrics = {
 'full_formula': 'Sales ~ ' + ' + '.join(predictors),
 'simple_formula': 'Sales ~ Accts',
 'simple_accounts_slope': simple.params['Accts'],
 'simple_accounts_se': simple.bse['Accts'],
 'simple_accounts_p': simple.pvalues['Accts'],
 'simple_accounts_ci95': simple.conf_int().loc['Accts'].tolist(),
 'simple_r_squared': simple.rsquared,
 'sales_accounts_correlation': df.Sales.corr(df.Accts),
 'adjusted_accounts_slope': full.params['Accts'],
 'adjusted_accounts_se': full.bse['Accts'],
 'adjusted_accounts_p': full.pvalues['Accts'],
 'adjusted_accounts_ci95': full.conf_int().loc['Accts'].tolist(),
 'partial_correlation': partial_r,
 'accounts_regressed_on_other_predictors_r_squared': ax_model.rsquared,
 'accounts_vif': 1 / (1 - ax_model.rsquared),
 'reduced_r_squared': reduced.rsquared,
 'full_r_squared': full.rsquared,
 'incremental_r_squared': full.rsquared - reduced.rsquared,
 'training_rmse': float(np.sqrt(np.mean(full.resid ** 2))),
 'residual_standard_error': float(np.sqrt(full.mse_resid))
}
(ROOT / 'visualization_evidence.json').write_text(json.dumps(metrics, indent=2))
df.corr().to_csv(ROOT / 'predictor_correlations.csv')
pd.DataFrame({'Sales':df.Sales, 'fitted_sales':full.fittedvalues,
 'residual_sales':full.resid, 'accounts_residual':rx,
 'sales_residual_given_other_seven':ry}).to_csv(ROOT / 'visualization_values.csv', index=False)
plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
color = '#186e9c'
axes[0].scatter(full.fittedvalues, df.Sales, c=color, s=35)
lims = [min(df.Sales.min(),full.fittedvalues.min())-150, max(df.Sales.max(),full.fittedvalues.max())+150]
axes[0].plot(lims, lims, '--', color='gray', label='Perfect fit')
axes[0].set(xlabel='Fitted Sales (recorded units)', ylabel='Observed Sales (recorded units)', title=f'Full model: in-sample fit\n$R^2$ = {full.rsquared:.3f}; RMSE = {metrics["training_rmse"]:.1f}', xlim=lims, ylim=lims)
axes[0].legend(frameon=False)
axes[1].scatter(df.Accts, df.Sales, c=color, s=35)
x = np.linspace(df.Accts.min(),df.Accts.max(),100)
axes[1].plot(x, simple.params['const']+simple.params['Accts']*x, color='#c35a26')
axes[1].set(xlabel='Accts (recorded units)',ylabel='Sales (recorded units)',title=f'Accounts alone\nSlope = {simple.params["Accts"]:.2f}; r = {df.Sales.corr(df.Accts):.3f}')
axes[2].axhline(0,color='gray',lw=.7)
axes[2].axvline(0,color='gray',lw=.7)
axes[2].scatter(rx,ry,c=color,s=35)
x = np.linspace(rx.min(),rx.max(),100)
axes[2].plot(x,full.params['Accts']*x,color='#c35a26')
axes[2].set(xlabel='Accts residual after other seven predictors',ylabel='Sales residual after other seven predictors',title=f'Accounts adjusted: added-variable plot\nSlope = {full.params["Accts"]:.2f}; partial r = {partial_r:.3f}')
fig.suptitle('Sales territory regressions | 25 territories',fontsize=14)
fig.tight_layout()
fig.savefig(ROOT / 'full_model_visualization.png',dpi=180)
print(json.dumps(metrics,indent=2))
print(df.corr()['Accts'].sort_values(ascending=False).to_string())
