"""Training-only Lasso penalty sensitivity on original constructed sales data."""
import os
os.environ['MPLCONFIGDIR'] = '.mplcache'
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LinearRegression

data = pd.read_csv('constructed_sales_seed4031.csv')
split = pd.read_csv('fixed_sales_split.csv')
train = data.loc[split['split'].eq('train')].copy()
names = [c for c in train if c not in ['store_id', 'weekly_sales']]
scaler = StandardScaler()
X = scaler.fit_transform(train[names])
y = train.weekly_sales.to_numpy() / 1000
alpha_max = float(np.max(np.abs(X.T @ (y-y.mean()))) / len(y))
alphas = np.geomspace(.01, alpha_max * 1.05, 180)
coefs = []
for a in alphas:
    fit = Lasso(alpha=a, max_iter=100000, tol=1e-10).fit(X, y)
    coefs.append(fit.coef_.copy())
coefs = np.array(coefs)
pd.DataFrame(coefs, columns=names).assign(alpha=alphas).to_csv('lasso_penalty_coefficient_path.csv', index=False)
examples = []
for a in [0, .1, .5, 1.0]:
    fit = LinearRegression().fit(X,y) if a == 0 else Lasso(alpha=a, max_iter=100000, tol=1e-10).fit(X,y)
    loss = float(np.mean((y-fit.predict(X))**2)/2)
    norm = float(np.abs(fit.coef_).sum())
    examples.append({'alpha': a, 'nonzero_coefficients': int(np.count_nonzero(np.abs(fit.coef_) > 1e-8)),
                     'coefficient_L1_norm': norm, 'training_SSE_over_2n': loss,
                     'penalty': a*norm, 'objective': loss+a*norm,
                     'intercept': float(fit.intercept_), 'coefficients': dict(zip(names, fit.coef_.tolist()))})
result = {'dataset': 'Original constructed sales, seed 4031', 'rows':len(y),
          'scope':'All fits and scaling use training data only. Fixed-penalty sensitivity, no tuning or validation.',
          'response_units':'Thousands of dollars', 'predictors':'Standardized with training means and population standard deviations',
          'objective':'SSE/(2n) + alpha * sum(abs(beta)); intercept unpenalized',
          'alpha_all_zero_threshold':alpha_max, 'fits':examples,
          'scaling_means':dict(zip(names,scaler.mean_.tolist())),
          'scaling_standard_deviations':dict(zip(names,scaler.scale_.tolist()))}
with open('lasso_penalty_demonstration_results.json','w') as f:
    json.dump(result,f,indent=2)

fig, axes = plt.subplots(1,2,figsize=(13,5.8),gridspec_kw={'width_ratios':[1,1.55]})
b = np.linspace(-4,4,401)
for a in [.5,1.0]:
    axes[0].plot(b,a*np.abs(b),label=f'lambda = {a:g}',lw=2)
axes[0].set(xlabel='One coefficient (thousands of dollars per predictor SD)',
            ylabel='Penalty contribution: lambda × |coefficient|',title='Absolute-value penalty has a corner at zero')
axes[0].legend()
axes[0].grid(alpha=.2)
colors = plt.cm.tab20(np.linspace(0,1,len(names)))
for j,name in enumerate(names):
    axes[1].plot(alphas,coefs[:,j],label=name.replace('_',' '),color=colors[j],lw=1.8)
axes[1].set(xscale='log',xlabel='Penalty strength lambda (log scale)',
            ylabel='Coefficient (thousands of dollars per predictor SD)',
            title='Lasso coefficient paths on 168 training stores')
axes[1].axhline(0,color='black',lw=.7)
axes[1].grid(alpha=.2)
axes[1].legend(loc='upper left',bbox_to_anchor=(1.02,1.02),fontsize=8,frameon=False)
fig.suptitle('Lasso penalty demonstration: constructed sales data',fontsize=15)
fig.subplots_adjust(left=.07,right=.79,bottom=.23,top=.84,wspace=.32)
fig.text(.07,.045,'Sales are measured in $1,000; all predictors use training-only standardization. Intercept is unpenalized.\nThese are in-sample coefficient paths, not validation results. The 72 reserved test stores remain unused.',fontsize=10)
fig.savefig('lasso_penalty_and_training_paths.png',dpi=160)
plt.close(fig)
print(json.dumps({'alpha_all_zero_threshold':alpha_max,'examples':[{k:v for k,v in e.items() if k not in ['coefficients','intercept']} for e in examples]},indent=2))
