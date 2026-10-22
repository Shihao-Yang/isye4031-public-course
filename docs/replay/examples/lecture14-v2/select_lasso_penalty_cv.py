"""Training-only five-fold CV for Lasso; reserved test outcomes unused."""
import os
os.environ['MPLCONFIGDIR'] = '.mplcache'
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.dummy import DummyRegressor

data = pd.read_csv('constructed_sales_seed4031.csv')
split = pd.read_csv('fixed_sales_split.csv')
train = data.loc[split['split'].eq('train')].reset_index(drop=True)
names = [c for c in train if c not in ['store_id','weekly_sales']]
X = train[names]
y = train.weekly_sales.to_numpy()/1000
folds = list(KFold(n_splits=5,shuffle=True,random_state=4031).split(X))
fold_assignment = np.zeros(len(train),dtype=int)
for fold,(_,valid) in enumerate(folds,1):
    fold_assignment[valid] = fold
pd.DataFrame({'store_id':train.store_id,'validation_fold':fold_assignment}).to_csv('lasso_cv_fixed_training_folds.csv',index=False)
alphas = np.geomspace(.001,10,121)
pipeline = Pipeline([('scale',StandardScaler()),('lasso',Lasso(max_iter=100000,tol=1e-9))])
search = GridSearchCV(pipeline,{'lasso__alpha':alphas},scoring='neg_root_mean_squared_error',cv=folds,refit=True)
search.fit(X,y)
fold_rmse = np.column_stack([-search.cv_results_[f'split{k}_test_score'] for k in range(5)])
mean = fold_rmse.mean(axis=1)
se = fold_rmse.std(axis=1,ddof=1)/np.sqrt(5)
best = int(np.argmin(mean))
one_se = int(np.flatnonzero(mean <= mean[best]+se[best])[-1])
baseline = -cross_val_score(DummyRegressor(strategy='mean'),X,y,cv=folds,scoring='neg_root_mean_squared_error')
curve = pd.DataFrame({'lambda':alphas,'mean_fold_RMSE_dollars':mean*1000,'SE_fold_RMSE_dollars':se*1000})
for k in range(5):
    curve[f'fold_{k+1}_RMSE_dollars'] = fold_rmse[:,k]*1000
curve.to_csv('lasso_fivefold_validation_curve.csv',index=False)
fits = {}
for label,idx in [('minimum_CV',best),('one_SE_alternative',one_se)]:
    pipeline.set_params(lasso__alpha=alphas[idx]).fit(X,y)
    fit = pipeline.named_steps['lasso']
    fits[label] = {'lambda':float(alphas[idx]),'mean_fold_RMSE_dollars':float(mean[idx]*1000),
                   'SE_fold_RMSE_dollars':float(se[idx]*1000),
                   'nonzero_after_refit_on_168_training_stores':int(np.count_nonzero(np.abs(fit.coef_)>1e-8)),
                   'training_refit_coefficients_thousands_per_SD':dict(zip(names,fit.coef_.tolist())),
                   'training_refit_intercept_thousands':float(fit.intercept_)}
summary = {'dataset':'Original constructed sales, seed 4031','training_rows':len(y),'fold_seed':4031,
           'procedure':'Five-fold shuffled CV; StandardScaler fitted inside each training fold, followed by Lasso. Select minimum mean fold RMSE from 121 log-spaced penalties between .001 and 10.',
           'response_units':'Thousands of dollars during fitting; reported RMSE in dollars',
           'fits':fits,'baseline_mean_only_CV_RMSE_dollars':float(baseline.mean()*1000),
           'baseline_fold_RMSE_dollars':(baseline*1000).tolist(),
           'one_SE_rule':'Largest candidate lambda with mean RMSE <= minimum mean RMSE + SE at minimum',
           'uncertainty':'SE is SD of five fold RMSE values / sqrt(5); descriptive, folds have overlapping training sets, not a formal confidence interval.',
           'limitations':'Curve validates fixed candidates using training-only folds. Its selected minimum is optimistically biased as an estimate of the tuned procedure. No nested validation or final test evaluation has been performed.',
           'best_is_grid_endpoint':bool(best in [0,len(alphas)-1]),
           'test_status':'72 reserved stores remain unused'}
with open('lasso_fivefold_validation_results.json','w') as f:
    json.dump(summary,f,indent=2)

fig,ax = plt.subplots(figsize=(10,6.5))
ax.plot(alphas,mean*1000,color='#1764a0',lw=2,label='Mean validation RMSE (5 folds)')
ax.fill_between(alphas,(mean-se)*1000,(mean+se)*1000,color='#1764a0',alpha=.16,label='Mean ± 1 SE across folds')
ax.axhline(baseline.mean()*1000,color='gray',ls=':',label='Predict training-fold mean sales')
for idx,color,label in [(best,'#c14627','Minimum CV'),(one_se,'#367a42','One-SE alternative')]:
    ax.axvline(alphas[idx],color=color,ls='--',lw=1.4,label=f'{label}: lambda = {alphas[idx]:.3f}')
    ax.scatter(alphas[idx],mean[idx]*1000,color=color,s=55,zorder=3)
ax.set(xscale='log',xlabel='Lasso penalty lambda (sales fitted in $1,000)',ylabel='Validation RMSE ($ per store-week)',
       title='Choosing Lasso penalty: training-only five-fold cross-validation')
ax.grid(alpha=.18)
ax.legend(loc='upper left',fontsize=10)
fig.subplots_adjust(left=.11,right=.97,top=.90,bottom=.26)
fig.text(.11,.06,'Original constructed sales data: 168 training stores; scaling is fitted inside every training fold.\nShaded band describes fold variability, not a confidence interval. The minimum is used for selection.\nThis is tuning evidence, not an independent evaluation of the selected procedure. Test set unused.',fontsize=10)
fig.savefig('lasso_fivefold_validation_evidence.png',dpi=160)
plt.close(fig)
print(json.dumps({k:v for k,v in summary.items() if k in ['baseline_mean_only_CV_RMSE_dollars','best_is_grid_endpoint']},indent=2))
print(json.dumps({label:{k:v for k,v in fit.items() if not k.startswith('training_refit')} for label,fit in fits.items()},indent=2))
