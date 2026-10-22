"""Elastic Net paths and joint training-only tuning on constructed sales data."""
import os
os.environ['MPLCONFIGDIR']='.mplcache'
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

df=pd.read_csv('constructed_sales_seed4031.csv')
split=pd.read_csv('fixed_sales_split.csv')
train=df.loc[split['split'].eq('train')].reset_index(drop=True)
names=[c for c in train if c not in ['store_id','weekly_sales']]
X=train[names]
y=train.weekly_sales.to_numpy()/1000
alphas=np.geomspace(.001,10,121)
ratios=[.05,.1,.25,.5,.75,.9,1.]
folds=list(KFold(n_splits=5,shuffle=True,random_state=4031).split(X))
pipe=Pipeline([('scale',StandardScaler()),('enet',ElasticNet(max_iter=100000,tol=1e-9))])
search=GridSearchCV(pipe,{'enet__alpha':alphas,'enet__l1_ratio':ratios},cv=folds,
                    scoring='neg_root_mean_squared_error',refit=True)
search.fit(X,y)
cv=search.cv_results_
rows=[]
for i,p in enumerate(cv['params']):
    errors=np.array([-cv[f'split{k}_test_score'][i] for k in range(5)])*1000
    rows.append({'lambda':p['enet__alpha'],'rho':p['enet__l1_ratio'],
                 'mean_fold_RMSE_dollars':float(errors.mean()),
                 'SE_fold_RMSE_dollars':float(errors.std(ddof=1)/np.sqrt(5)),
                 **{f'fold_{k+1}_RMSE_dollars':float(e) for k,e in enumerate(errors)}})
curve=pd.DataFrame(rows)
curve.to_csv('elastic_net_joint_cv_results.csv',index=False)
best=search.best_estimator_.named_steps['enet']
summary={'dataset':'Original constructed sales, seed 4031','training_rows':len(y),
         'objective':'SSE/(2n) + lambda*rho*sum(abs(beta)) + lambda*(1-rho)*sum(beta**2)/2; unpenalized intercept',
         'response_units':'Thousands of dollars; predictors standardized within fitting folds',
         'tuning':'Same fixed five training folds as Lasso; joint grid search over 121 lambdas [.001,10] and rho [.05,.1,.25,.5,.75,.9,1]. Minimum mean fold RMSE. No test use.',
         'best_lambda':float(search.best_params_['enet__alpha']),
         'best_rho':float(search.best_params_['enet__l1_ratio']),
         'best_mean_fold_RMSE_dollars':float(-search.best_score_*1000),
         'nonzero_training_refit_coefficients':int(np.sum(np.abs(best.coef_)>1e-8)),
         'training_refit_coefficients_thousands_per_SD':dict(zip(names,best.coef_.tolist())),
         'training_refit_intercept_thousands':float(best.intercept_),
         'best_by_rho':curve.loc[curve.groupby('rho').mean_fold_RMSE_dollars.idxmin(),['rho','lambda','mean_fold_RMSE_dollars']].to_dict(orient='records'),
         'limits':'Coefficient paths at rho=.5 are fixed-mix sensitivity. CV curves are tuning evidence, not independent validation of the selected procedure; minimum can be optimistic. No selection-stability improvement or causal inference is established. 72 test stores remain unused.'}
with open('elastic_net_demonstration_results.json','w') as f:
    json.dump(summary,f,indent=2)

# Descriptive paths at an explicitly fixed mix; not the selected procedure.
Z=StandardScaler().fit_transform(X)
paths=[]
for a in alphas:
    model=ElasticNet(alpha=a,l1_ratio=.5,max_iter=100000,tol=1e-9).fit(Z,y)
    paths.append(model.coef_.copy())
paths=np.array(paths)
pd.DataFrame(paths,columns=names).assign(lambda_value=alphas,rho=.5).to_csv('elastic_net_fixed_half_mix_paths.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(15,7))
colors=plt.cm.tab20(np.linspace(0,1,len(names)))
for j,name in enumerate(names):
    axes[0].plot(alphas,paths[:,j],color=colors[j],label=name.replace('_',' '),lw=1.6)
axes[0].set(xscale='log',xlabel='Penalty strength lambda (log scale)',
            ylabel='Coefficient ($1,000 per predictor SD)',title='Elastic Net coefficient paths: fixed rho = 0.5')
axes[0].axhline(0,color='black',lw=.7)
axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=3,fontsize=7.5,frameon=False)
for rho in ratios:
    sub=curve.loc[curve.rho==rho]
    axes[1].plot(sub['lambda'],sub.mean_fold_RMSE_dollars,label=f'rho = {rho:g}'+(' (Lasso)' if rho==1 else ''),lw=1.5)
axes[1].scatter(summary['best_lambda'],summary['best_mean_fold_RMSE_dollars'],s=85,
                facecolors='none',edgecolors='black',linewidths=1.7,zorder=5,label='Joint grid minimum')
axes[1].set(xscale='log',xlabel='Penalty strength lambda (log scale)',
            ylabel='Mean validation RMSE ($ per store-week)',title='Tune both lambda and rho: five-fold CV')
axes[1].legend(fontsize=9,loc='upper left')
for ax in axes:
    ax.grid(alpha=.18)
fig.suptitle('Combining L1 and L2: Elastic Net on constructed sales data',fontsize=16,y=.97)
fig.subplots_adjust(left=.065,right=.98,top=.88,bottom=.33,wspace=.24)
fig.text(.065,.045,'168 training stores; sales fitted in $1,000. Left: in-sample coefficient sensitivity at a fixed mix.\nRight: joint tuning with scaling fitted inside each training fold; curves do not independently evaluate the selected procedure.\nRho is the L1 fraction of the penalty formula, not a fraction of variance explained. The reserved test set remains unused.',fontsize=11)
fig.savefig('elastic_net_paths_and_joint_tuning.png',dpi=160)
plt.close(fig)
print(json.dumps({k:v for k,v in summary.items() if k in ['best_lambda','best_rho','best_mean_fold_RMSE_dollars','nonzero_training_refit_coefficients','best_by_rho']},indent=2))
