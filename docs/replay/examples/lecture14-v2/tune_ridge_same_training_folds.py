"""Tune normalized Ridge and compare saved training CV results; no test use."""
import os
os.environ['MPLCONFIGDIR']='.mplcache'
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator,RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold,GridSearchCV

class NormalizedRidge(RegressorMixin,BaseEstimator):
    def __init__(self,lam=1.0):
        self.lam=lam
    def fit(self,X,y):
        self.model_=Ridge(alpha=len(y)*self.lam).fit(X,y)
        self.n_features_in_=self.model_.n_features_in_
        return self
    def predict(self,X):
        return self.model_.predict(X)

df=pd.read_csv('constructed_sales_seed4031.csv')
split=pd.read_csv('fixed_sales_split.csv')
train=df.loc[split['split'].eq('train')].reset_index(drop=True)
names=[c for c in train if c not in ['store_id','weekly_sales']]
X=train[names]; y=train.weekly_sales.to_numpy()/1000
folds=list(KFold(n_splits=5,shuffle=True,random_state=4031).split(X))
grid=np.r_[0,np.geomspace(.001,1000,121)]
pipe=Pipeline([('scale',StandardScaler()),('ridge',NormalizedRidge())])
search=GridSearchCV(pipe,{'ridge__lam':grid},cv=folds,scoring='neg_root_mean_squared_error',refit=True)
search.fit(X,y)
cv=search.cv_results_
rows=[]
for i,p in enumerate(cv['params']):
    errors=np.array([-cv[f'split{k}_test_score'][i] for k in range(5)])*1000
    rows.append({'lambda':p['ridge__lam'],'mean_fold_RMSE_dollars':float(errors.mean()),
                 'SE_fold_RMSE_dollars':float(errors.std(ddof=1)/np.sqrt(5)),
                 **{f'fold_{k+1}_RMSE_dollars':float(e) for k,e in enumerate(errors)}})
ridge_curve=pd.DataFrame(rows)
ridge_curve.to_csv('ridge_training_fivefold_tuning.csv',index=False)
ridge_best=ridge_curve.loc[ridge_curve.mean_fold_RMSE_dollars.idxmin()].to_dict()
lasso_curve=pd.read_csv('lasso_fivefold_validation_curve.csv')
lasso_best=lasso_curve.loc[lasso_curve.mean_fold_RMSE_dollars.idxmin()].to_dict()
enet_curve=pd.read_csv('elastic_net_joint_cv_results.csv')
enet_best=enet_curve.loc[enet_curve.mean_fold_RMSE_dollars.idxmin()].to_dict()
with open('lasso_fivefold_validation_results.json') as f:
    lasso_full=json.load(f)['fits']['minimum_CV']
with open('elastic_net_demonstration_results.json') as f:
    enet_full=json.load(f)
counts={'Ridge':int(np.sum(np.abs(search.best_estimator_.named_steps['ridge'].model_.coef_)>1e-8)),
        'Lasso':lasso_full['nonzero_after_refit_on_168_training_stores'],
        'Elastic Net':enet_full['nonzero_training_refit_coefficients']}
comparison=[]
for method,row in [('Ridge',ridge_best),('Lasso',lasso_best),('Elastic Net',enet_best)]:
    comparison.append({'model':method,'lambda':row['lambda'],
                       'rho':row.get('rho',1 if method=='Lasso' else 0),
                       'mean_fold_RMSE_dollars':row['mean_fold_RMSE_dollars'],
                       'SE_fold_RMSE_dollars':row['SE_fold_RMSE_dollars'],
                       'nonzero_training_refit':counts[method],
                       'fold_RMSE_dollars':[row[f'fold_{k+1}_RMSE_dollars'] for k in range(5)]})
result={'dataset':'Original constructed sales, seed 4031','scope':'Training-only tuning comparison on identical five folds; test unused',
        'ridge_objective':'SSE/(2n) + lambda*sum(beta**2)/2, intercept unpenalized; sklearn alpha equals fitting-fold n times lambda',
        'ridge_grid':'0 plus 121 log-spaced values from .001 to 1000',
        'preprocessing':'StandardScaler inside each CV training fold; sales fitted in thousands of dollars',
        'comparison':comparison,
        'ridge_training_refit_coefficients_thousands_per_SD':dict(zip(names,search.best_estimator_.named_steps['ridge'].model_.coef_.tolist())),
        'ridge_best_grid_endpoint':bool(ridge_best['lambda'] in [grid[0],grid[-1]]),
        'limits':'Each entry is a selected tuning minimum. The CV folds do not independently evaluate model-family selection or tuning. SE bars describe fold variability, not confidence intervals.'}
with open('ridge_lasso_enet_training_comparison.json','w') as f:
    json.dump(result,f,indent=2)

fig,ax=plt.subplots(figsize=(9,6))
for j,row in enumerate(comparison):
    ax.scatter(np.full(5,j),row['fold_RMSE_dollars'],s=28,color='gray',alpha=.7,
                label='Individual held-out fold RMSE' if j==0 else None)
    ax.errorbar(j,row['mean_fold_RMSE_dollars'],yerr=row['SE_fold_RMSE_dollars'],fmt='o',
                color='#1764a0',capsize=7,ms=8,label='Mean ± 1 SE across folds' if j==0 else None)
ax.set_xticks(range(3),[r['model'] for r in comparison])
ax.set(xlim=(-.5,2.5),ylabel='Validation RMSE ($ per store-week)',title='Tuned models: same five training validation folds')
ax.grid(axis='y',alpha=.18)
ax.legend(fontsize=10)
fig.subplots_adjust(left=.13,right=.97,top=.9,bottom=.24)
fig.text(.13,.05,'Original constructed sales: 168 training stores; all scaling and tuning use training folds only.\nPoints are fold errors at each selected penalty. Bars are descriptive, not confidence intervals.\nThese tuning minima are not independent final performance estimates. Reserved test set unused.',fontsize=10)
fig.savefig('tuned_models_training_cv_comparison.png',dpi=160)
plt.close(fig)
print(json.dumps(comparison,indent=2))
