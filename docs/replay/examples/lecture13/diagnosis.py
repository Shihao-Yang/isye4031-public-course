import os
os.environ['MPLCONFIGDIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.mplcache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / 'sales_territory.csv')
predictors = [c for c in df.columns if c != 'Sales']
X = sm.add_constant(df[predictors], has_constant='add')
fit = sm.OLS(df['Sales'], X).fit()
corr = df.corr(method='pearson')
corr.to_csv(ROOT / 'correlations.csv')
rows = []
for i, name in enumerate(predictors, start=1):
    rows.append(dict(predictor=name, vif=float(variance_inflation_factor(X.to_numpy(), i)),
                     coefficient=float(fit.params[name]), standard_error=float(fit.bse[name]),
                     p_value=float(fit.pvalues[name])))
table = pd.DataFrame(rows).sort_values('vif', ascending=False)
table.to_csv(ROOT / 'vif_coefficients.csv', index=False)
pd.DataFrame({'coefficient':fit.params, 'standard_error':fit.bse, 'p_value':fit.pvalues}).to_csv(ROOT / 'full_model_coefficients.csv')
(ROOT / 'model_summary.txt').write_text(fit.summary().as_text())
results = dict(n=len(df), predictors=predictors, residual_df=int(fit.df_resid),
               design_rank=int(np.linalg.matrix_rank(X)), design_columns=X.shape[1],
               time_accts_correlation=float(corr.loc['Time','Accts']),
               top_three=table.head(3).to_dict(orient='records'),
               all_vifs_and_coefficients=table.to_dict(orient='records'),
               correlations=corr.to_dict(),
               inference_assumptions='Classical OLS standard errors and two-sided t tests assume a correctly specified linear conditional mean, independent homoskedastic errors, and normal errors for exact finite-sample inference; predictors are treated as fixed for inference.')
(ROOT / 'diagnosis_results.json').write_text(json.dumps(results, indent=2))
plt.rcParams.update({'font.size':14, 'axes.labelsize':15, 'axes.titlesize':17, 'xtick.labelsize':13, 'ytick.labelsize':13})
fig, ax = plt.subplots(figsize=(8,6), layout='constrained')
ax.scatter(df['Time'],df['Accts'],s=65,alpha=.85,edgecolors='white',linewidth=.7)
ax.set(title='Sales territories: Time versus Accts', xlabel='Time with company (months)', ylabel='Accts (supplied accounts measure)')
ax.grid(alpha=.2)
fig.savefig(ROOT / 'time_vs_accts.png',dpi=180)
plt.close(fig)
print(json.dumps({k:results[k] for k in ['n','residual_df','design_rank','design_columns','time_accts_correlation','top_three']},indent=2))
print('\nPredictor correlation matrix:\n'+df[predictors].corr().round(3).to_string())
