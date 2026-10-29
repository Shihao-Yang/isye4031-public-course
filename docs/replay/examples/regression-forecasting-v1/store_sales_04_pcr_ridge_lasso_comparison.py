from pathlib import Path
import os
os.environ['MPLCONFIGDIR'] = str(Path.cwd() / '.mplcache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold, GridSearchCV

cols = ['size_sqft','advertising_thousand_usd_month','staff_count',
        'area_household_income_thousand_usd_year','nearby_competitor_count']
d = pd.read_csv('store_sales_01_data.csv')
d = d.loc[d['split'].eq('training')].reset_index(drop=True)
# Fixed, pointwise feature construction learns nothing from validation data.
X = d[cols].to_numpy()
X[:,1] -= 28
X = np.column_stack([X, X[:,1]**2])
y = d.sales_thousand_usd_month.to_numpy()
folds = pd.read_csv('store_sales_02_shared_five_folds.csv')
fold_ids = d[['store_id']].merge(folds,on='store_id',validate='one_to_one').validation_fold.to_numpy()
outer = [(np.flatnonzero(fold_ids!=k),np.flatnonzero(fold_ids==k)) for k in range(1,6)]

def spec(name):
    if name == 'Curved OLS':
        return Pipeline([('scale',StandardScaler()),('reg',LinearRegression())]), None
    if name == 'PCR':
        return Pipeline([('scale',StandardScaler()),('pca',PCA()),('reg',LinearRegression())]), {'pca__n_components':list(range(1,7))}
    if name == 'Ridge':
        return Pipeline([('scale',StandardScaler()),('reg',Ridge())]), {'reg__alpha':np.logspace(-2,3,16)}
    return Pipeline([('scale',StandardScaler()),('reg',Lasso(max_iter=20000,tol=1e-8))]), {'reg__alpha':np.logspace(-3,1,17)}
names = ['Curved OLS','PCR','Ridge','Lasso']
rows, summaries, settings = [], [], {}
predictions = d[['store_id']].copy()
for name in names:
    pred = np.empty(len(y))
    for k,(tr,va) in enumerate(outer,1):
        model,grid = spec(name)
        if grid:
            inner = list(KFold(5,shuffle=True,random_state=403120+k).split(X[tr]))
            fitted = GridSearchCV(model,grid,cv=inner,scoring='neg_mean_squared_error',n_jobs=1).fit(X[tr],y[tr])
            params = fitted.best_params_
        else:
            fitted = model.fit(X[tr],y[tr])
            params = {}
        pred[va] = fitted.predict(X[va])
        rows.append({'model':name,'fold':k,'rmse':float(np.sqrt(np.mean((y[va]-pred[va])**2))), 'selected_parameters':json.dumps(params)})
    predictions[name] = pred
    summaries.append({'model':name,'nested_validation_rmse':float(np.sqrt(np.mean((y-pred)**2))),
                      'nested_validation_mae':float(np.mean(np.abs(y-pred))),
                      'mean_fold_rmse':float(np.mean([r['rmse'] for r in rows if r['model']==name]))})
    model,grid = spec(name)
    if grid:
        fitted = GridSearchCV(model,grid,cv=outer,scoring='neg_mean_squared_error',n_jobs=1).fit(X,y)
        settings[name] = fitted.best_params_
        cv = pd.DataFrame(fitted.cv_results_)
        cv.to_csv('store_sales_04_'+name.lower()+'_training_tuning.csv',index=False)
    else:
        fitted = model.fit(X,y)
        settings[name] = {}
    if name == 'Lasso':
        settings[name]['retained_coefficients'] = int(np.count_nonzero(fitted.best_estimator_[-1].coef_))
results = pd.DataFrame(summaries)
fold_results = pd.DataFrame(rows)
results.to_csv('store_sales_04_nested_validation_results.csv',index=False)
fold_results.to_csv('store_sales_04_nested_fold_results.csv',index=False)
predictions.to_csv('store_sales_04_nested_validation_predictions.csv',index=False)
Path('store_sales_04_comparison_design.json').write_text(json.dumps({
    'features':cols+['centered_advertising_squared'],
    'advertising_center':28,'training_stores':len(d),'reserved_test_used':False,
    'outer_folds':'Saved five-fold split from store_sales_02_shared_five_folds.csv',
    'inner_folds':'Five shared inner folds per outer training set; seed=403120+outer_fold_number',
    'tuning_objective':'Mean squared error; scaling and PCA fitted inside each training fold',
    'final_training_settings':settings,
    'note':'Full-training tuning selects candidate settings, with no test evaluation. Nested validation evaluates the tuning procedures.'},indent=2))
plt.rcParams.update({'font.size':12,'axes.titlesize':14})
fig,ax = plt.subplots(figsize=(8.5,4.7))
colors = ['#333333','#2166ac','#c45118','#41853c']
for name,color in zip(names,colors):
    r = fold_results.loc[fold_results.model.eq(name)]
    ax.plot(r.fold,r.rmse,'o-',lw=2,color=color,label=name)
ax.set(xticks=range(1,6),xlabel='Shared outer validation fold',ylabel='Sales prediction RMSE ($1,000)',
       title='Prediction errors after tuning within each training fold')
ax.grid(alpha=.2)
ax.legend(ncol=2,fontsize=10)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
fig.savefig('store_sales_04_nested_validation_comparison.png',dpi=180)
print(results.to_string(index=False))
print(fold_results.to_string(index=False))
print(json.dumps(settings,indent=2))
