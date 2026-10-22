"""Nested training-only diagnostics for the minimum-CV Lasso procedure."""
import os
os.environ['MPLCONFIGDIR'] = '.mplcache'
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

df = pd.read_csv('constructed_sales_seed4031.csv')
split = pd.read_csv('fixed_sales_split.csv')
train = df.loc[split['split'].eq('train')].reset_index(drop=True)
names = [c for c in train if c not in ['store_id','weekly_sales']]
X = train[names]
y = train.weekly_sales.to_numpy()/1000
alphas = np.geomspace(.001,10,121)
outer = KFold(n_splits=5,shuffle=True,random_state=4031)
pred = np.zeros(len(y))
fold_labels = np.zeros(len(y),dtype=int)
fold_records = []
for fold,(fit_idx,valid_idx) in enumerate(outer.split(X),1):
    pipe = Pipeline([('scale',StandardScaler()),('lasso',Lasso(max_iter=100000,tol=1e-9))])
    inner = KFold(n_splits=5,shuffle=True,random_state=4031)
    search = GridSearchCV(pipe,{'lasso__alpha':alphas},cv=inner,
                          scoring='neg_root_mean_squared_error',refit=True)
    search.fit(X.iloc[fit_idx],y[fit_idx])
    pred[valid_idx] = search.predict(X.iloc[valid_idx])
    fold_labels[valid_idx] = fold
    fold_records.append({'outer_fold':fold,'training_rows':len(fit_idx),'held_out_rows':len(valid_idx),
                         'lambda':float(search.best_params_['lasso__alpha']),
                         'outer_RMSE_dollars':float(np.sqrt(np.mean((y[valid_idx]-pred[valid_idx])**2))*1000),
                         'nonzero_coefficients':int(np.sum(np.abs(search.best_estimator_.named_steps['lasso'].coef_)>1e-8)),
                         'training_store_ids':train.store_id.iloc[fit_idx].tolist(),
                         'held_out_store_ids':train.store_id.iloc[valid_idx].tolist()})
resid = y-pred
out = pd.DataFrame({'store_id':train.store_id,'outer_fold':fold_labels,
                    'observed_sales_dollars':y*1000,'OOF_prediction_dollars':pred*1000,
                    'OOF_residual_dollars':resid*1000})
out.to_csv('lasso_nested_cv_diagnostic_predictions.csv',index=False)

# Investigate the largest errors without deleting rows. Known synthetic equation
# is used only for this explanatory audit, never as a predictor or tuning input.
conditional_mean = (30000+4*(train.digital_ad_spend-1500)+70*(train.promotion_index-50)
                    +7*(train.floor_area-2000)+8*(train.staff_hours-300)
                    +180*(train.local_demand_index-50))
top = np.argsort(np.abs(resid))[-5:][::-1]
audits = []
for i in top:
    fold = fold_labels[i]
    others = X.loc[fold_labels != fold]
    outside = [c for c in names if X.loc[i,c]<others[c].min() or X.loc[i,c]>others[c].max()]
    audits.append({**out.iloc[i].to_dict(),
                   'known_synthetic_conditional_mean_dollars':float(conditional_mean.iloc[i]),
                   'known_synthetic_noise_dollars':float(y[i]*1000-conditional_mean.iloc[i]),
                   'predictors_outside_outer_training_range':outside,
                   'row_finite':bool(np.isfinite(train.loc[i,names+['weekly_sales']].to_numpy(dtype=float)).all())})
pd.DataFrame(audits).to_csv('lasso_large_error_investigation.csv',index=False)
bins = pd.qcut(pred,4,duplicates='drop')
binned = pd.DataFrame({'pred':pred,'resid':resid,'bin':bins}).groupby('bin',observed=True).agg(
    prediction=('pred','mean'),mean_residual=('resid','mean'),
    RMSE=('resid',lambda r: np.sqrt(np.mean(r**2))),count=('resid','size')).reset_index()
binned['bin'] = binned['bin'].astype(str)
binned.to_csv('lasso_nested_cv_residual_bins.csv',index=False)
result = {'dataset':'Original constructed sales, seed 4031',
          'procedure':'5 outer folds, each with fresh 5-fold inner CV over 121 penalties [.001,10], minimizing mean fold RMSE. Scaling fitted inside each inner training fold; outer refit uses outer training rows only.',
          'error_definition':'Observed minus predicted sales; all displayed errors are nested out-of-fold on 168 training stores.',
          'pooled_outer_RMSE_dollars':float(np.sqrt(np.mean(resid**2))*1000),
          'pooled_outer_MAE_dollars':float(np.mean(np.abs(resid))*1000),
          'mean_outer_residual_dollars':float(resid.mean()*1000),
          'spearman_absolute_residual_vs_prediction':float(stats.spearmanr(np.abs(resid),pred).statistic),
          'folds':fold_records,'largest_error_audits':audits,
          'missing_training_values':int(train.isna().sum().sum()),
          'duplicate_store_ids':int(train.store_id.duplicated().sum()),
          'limits':'Nested-CV diagnostic errors share training data; normal-reference Q-Q plot is descriptive. No OLS degrees of freedom, t-tests or post-selection confidence intervals are claimed. Rows retained; 72 reserved test stores unused.'}
with open('lasso_nested_cv_diagnostic_results.json','w') as f:
    json.dump(result,f,indent=2)

fig,axes = plt.subplots(2,2,figsize=(12,9))
ax=axes[0,0]
ax.scatter(pred,resid,s=23,alpha=.65)
ax.axhline(0,color='black',lw=1)
ax.plot(binned.prediction,binned.mean_residual,'o-',color='#bf4c28',lw=2,label='Mean in prediction quartile')
ax.set(xlabel='OOF predicted sales ($1,000)',ylabel='OOF residual ($1,000)',title='Mean pattern: look for curvature or bias')
ax.legend(fontsize=9)
for i in top[:3]:
    ax.annotate(str(train.store_id.iloc[i]),(pred[i],resid[i]),xytext=(4,5),textcoords='offset points',fontsize=9)
ax=axes[0,1]
ax.scatter(pred,np.abs(resid),s=23,alpha=.65)
ax.set(xlabel='OOF predicted sales ($1,000)',ylabel='Absolute OOF residual ($1,000)',title='Error spread: look for a widening funnel')
ax=axes[1,0]
(q,ordered),(slope,intercept,r) = stats.probplot(resid,dist='norm')
ax.scatter(q,ordered,s=22,alpha=.7)
ax.plot(q,slope*q+intercept,color='#bf4c28',lw=1.5)
ax.set(xlabel='Theoretical normal quantile',ylabel='Ordered OOF residual ($1,000)',title='Q–Q plot: inspect asymmetry and tails')
ax=axes[1,1]
ax.scatter(pred,y,s=23,alpha=.65)
low=min(y.min(),pred.min()); high=max(y.max(),pred.max())
ax.plot([low,high],[low,high],color='black',lw=1)
ax.set(xlabel='OOF predicted sales ($1,000)',ylabel='Observed sales ($1,000)',title='Prediction agreement: diagonal means exact')
for ax in axes.flat:
    ax.grid(alpha=.15)
fig.suptitle('Tuned Lasso diagnostics: constructed sales, nested out-of-fold errors',fontsize=15,y=.98)
fig.subplots_adjust(left=.08,right=.97,top=.90,bottom=.15,hspace=.4,wspace=.28)
fig.text(.08,.035,'168 training stores. Each outer held-out store is excluded from scaling, tuning, and fitting used for its prediction.\nFive outer folds × five inner folds; labels identify the three largest absolute errors. All rows retained.\nThese diagnose the tuning procedure, not one full-training fit. The 72 reserved test stores remain unused.',fontsize=10)
fig.savefig('lasso_nested_cv_diagnostic_checks.png',dpi=160)
plt.close(fig)
print(json.dumps({k:result[k] for k in ['pooled_outer_RMSE_dollars','pooled_outer_MAE_dollars','mean_outer_residual_dollars','spearman_absolute_residual_vs_prediction','missing_training_values','duplicate_store_ids']},indent=2))
print(json.dumps({'fold_penalties':[r['lambda'] for r in fold_records],'largest_errors':audits[:3]},indent=2))
print(binned.to_string(index=False))
