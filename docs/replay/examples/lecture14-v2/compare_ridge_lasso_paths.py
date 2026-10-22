"""Fixed-penalty sensitivity on constructed training data; no validation."""
import os
os.environ['MPLCONFIGDIR'] = '.mplcache'
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, LinearRegression

df = pd.read_csv('constructed_sales_seed4031.csv')
membership = pd.read_csv('fixed_sales_split.csv')
train = df.loc[membership['split'].eq('train')]
names = [c for c in train if c not in ['store_id', 'weekly_sales']]
X = StandardScaler().fit_transform(train[names])
y = train.weekly_sales.to_numpy()/1000
n = len(y)
zero_threshold = float(np.max(np.abs(X.T @ (y-y.mean()))) / n)
ols = LinearRegression().fit(X,y)
grids = {'Lasso':np.geomspace(.001,zero_threshold*1.1,240),
         'Ridge':np.geomspace(.001,1000,280)}
paths = {}
rows = []
for method, grid in grids.items():
    coefficients = []
    for lam in grid:
        # Ridge sklearn minimizes SSE + alpha * ||beta||^2.
        # Thus alpha = n*lambda for SSE/(2n) + lambda*||beta||^2/2.
        model = (Lasso(alpha=lam,max_iter=100000,tol=1e-10)
                 if method == 'Lasso' else Ridge(alpha=n*lam))
        model.fit(X,y)
        coefficients.append(model.coef_.copy())
        rows.append({'model':method,'lambda':float(lam),
                     **dict(zip(names,model.coef_.tolist()))})
    paths[method] = np.array(coefficients)
pd.DataFrame(rows).to_csv('ridge_lasso_training_coefficient_paths.csv',index=False)
checks = {}
for method in grids:
    model = (Lasso(alpha=1,max_iter=100000,tol=1e-10) if method=='Lasso' else Ridge(alpha=n))
    model.fit(X,y)
    checks[method] = {'lambda':1,'nonzero_above_1e_minus_8':int(np.sum(np.abs(model.coef_)>1e-8)),
                      'coefficients':dict(zip(names,model.coef_.tolist()))}
result = {'dataset':'Original synthetic sales, seed 4031','training_rows':n,
          'scope':'Training-only fixed-penalty sensitivity; no tuning, validation or reserved test evaluation',
          'scaling':'Predictors standardized on training rows; sales in thousands of dollars',
          'objectives':{'Lasso':'SSE/(2n) + lambda * sum(abs(beta))',
                        'Ridge':'SSE/(2n) + (lambda/2) * sum(beta**2)'},
          'intercept':'Unpenalized for both methods',
          'lasso_all_zero_threshold':zero_threshold,
          'OLS_coefficients':dict(zip(names,ols.coef_.tolist())),
          'illustrative_fits':checks}
with open('ridge_lasso_path_results.json','w') as f:
    json.dump(result,f,indent=2)

colors = plt.cm.tab20(np.linspace(0,1,len(names)))
fig, axes = plt.subplots(1,2,figsize=(14,6.4),sharey=True)
for ax,method in zip(axes,['Lasso','Ridge']):
    for j,name in enumerate(names):
        ax.plot(grids[method],paths[method][:,j],color=colors[j],lw=1.8,
                label=name.replace('_',' '))
    ax.set(xscale='log',xlabel='Penalty strength lambda (log scale)',title=method)
    ax.axhline(0,color='black',lw=.7)
    ax.grid(alpha=.18)
axes[0].set_ylabel('Coefficient ($1,000 per predictor standard deviation)')
axes[0].axvline(zero_threshold,color='gray',ls=':',lw=1)
axes[0].text(.98,.95,f'All zero at lambda = {zero_threshold:.2f}',
             transform=axes[0].transAxes,ha='right',va='top',fontsize=10)
axes[1].legend(loc='upper left',bbox_to_anchor=(1.02,1.02),fontsize=9,frameon=False)
fig.suptitle('Constructed sales: Lasso and Ridge coefficient paths',fontsize=16,y=.96)
fig.subplots_adjust(left=.075,right=.79,bottom=.24,top=.86,wspace=.15)
fig.text(.075,.055,
         '168 training stores; both methods use the same training-standardized inputs and an unpenalized intercept.\nHorizontal ranges differ. Equal lambda values do not imply equal shrinkage for the two penalties.\nFixed-penalty sensitivity only; no validation results. The 72 reserved test stores remain unused.',fontsize=11)
fig.savefig('ridge_lasso_training_paths.png',dpi=160)
plt.close(fig)
print(json.dumps({'lasso_all_zero_threshold':zero_threshold,
                  'illustrative_counts_at_lambda_1':{k:v['nonzero_above_1e_minus_8'] for k,v in checks.items()}},indent=2))
