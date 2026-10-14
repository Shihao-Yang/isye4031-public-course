import os
os.environ['MPLCONFIGDIR'] = '.mplcache'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import json

df = pd.read_csv('sales_territory.csv')
X = df.drop(columns='Sales')
assert not df.isna().any().any()
Z = sm.add_constant(X)
model = sm.OLS(df.Sales, Z).fit()
vifs = {c: float(variance_inflation_factor(Z.values, i)) for i,c in enumerate(Z.columns) if c != 'const'}
corr = X.corr()
pairs = sorted([{'predictors':[a,b], 'correlation':float(corr.loc[a,b])} for i,a in enumerate(X.columns) for b in X.columns[i+1:]], key=lambda d:abs(d['correlation']), reverse=True)
standardized = (X-X.mean())/X.std(ddof=1)
singular = np.linalg.svd(standardized.values, compute_uv=False)
results = {'model':'OLS Sales on intercept and all eight supplied predictors', 'n':len(df), 'residual_df':float(model.df_resid), 'vif':vifs, 'auxiliary_r_squared':{c:1-1/v for c,v in vifs.items()}, 'largest_absolute_correlations':pairs[:8], 'standardized_predictor_condition_number':float(singular[0]/singular[-1]), 'r_squared':float(model.rsquared),'coefficients':{c:{'estimate':float(model.params[c]),'standard_error':float(model.bse[c]),'p_value':float(model.pvalues[c])} for c in Z.columns}}
with open('sales_multicollinearity_results.json','w') as f: json.dump(results,f,indent=2)
corr.to_csv('sales_predictor_correlations.csv')
fig, ax = plt.subplots(figsize=(7.5,6))
im=ax.imshow(corr, vmin=-1,vmax=1,cmap='RdBu_r')
ax.set_xticks(range(len(X.columns)),X.columns,rotation=45,ha='right')
ax.set_yticks(range(len(X.columns)),X.columns)
for i in range(len(X.columns)):
 for j in range(len(X.columns)):
  ax.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center',color='white' if abs(corr.iloc[i,j])>.65 else 'black',fontsize=9)
fig.colorbar(im,ax=ax,label='Pearson correlation')
ax.set_title('Sales data: correlations among eight predictors')
fig.tight_layout()
fig.savefig('sales_predictor_overlap.png',dpi=160)
print(json.dumps(results,indent=2))
