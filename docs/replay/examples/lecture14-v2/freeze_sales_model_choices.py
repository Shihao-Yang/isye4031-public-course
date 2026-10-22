"""Freeze model specifications and training fits before any final test use."""
import os
os.environ['MPLCONFIGDIR']='.mplcache'
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,Lasso,ElasticNet

with open('ridge_lasso_enet_training_comparison.json') as f:
    comparison=json.load(f)['comparison']
with open('lasso_fivefold_validation_results.json') as f:
    lasso_lambda=json.load(f)['fits']['minimum_CV']['lambda']
with open('elastic_net_demonstration_results.json') as f:
    enet=json.load(f)
ridge_lambda=next(r['lambda'] for r in comparison if r['model']=='Ridge')
data=pd.read_csv('constructed_sales_seed4031.csv')
split=pd.read_csv('fixed_sales_split.csv')
train=data.loc[split['split'].eq('train')]
names=[c for c in train if c not in ['store_id','weekly_sales']]
estimators={'Lasso':Lasso(alpha=lasso_lambda,max_iter=100000,tol=1e-9),
            'Ridge':Ridge(alpha=len(train)*ridge_lambda),
            'Elastic Net':ElasticNet(alpha=enet['best_lambda'],l1_ratio=enet['best_rho'],max_iter=100000,tol=1e-9)}
models={}
for name,estimator in estimators.items():
    models[name]=Pipeline([('scale',StandardScaler()),('model',estimator)]).fit(train[names],train.weekly_sales.to_numpy()/1000)
joblib.dump(models,'frozen_sales_training_models.joblib')
manifest={'dataset':'Original constructed sales, seed 4031',
          'dataset_sha256':hashlib.sha256(open('constructed_sales_seed4031.csv','rb').read()).hexdigest(),
          'primary_model':'Lasso','primary_lambda':lasso_lambda,
          'decision':'Freeze minimum-CV Lasso for prediction: simpler than the other tuned fits (9 versus 10 or 12 nonzero inputs) with only about $5 higher mean tuning RMSE than Elastic Net. This is a practical parsimony choice, not a statistical equivalence claim or causal variable-selection claim.',
          'locked_comparison_candidates':{'Lasso':{'lambda':lasso_lambda,'rho':1},
                                           'Ridge':{'lambda':ridge_lambda,'sklearn_alpha':len(train)*ridge_lambda},
                                           'Elastic Net':{'lambda':enet['best_lambda'],'rho':enet['best_rho']}},
          'predictors_in_order':names,'preprocessing':'StandardScaler fit on all 168 training stores; no feature removal or transformations beyond standardization',
          'response':'weekly_sales / 1000; multiply model predictions by 1000 to report dollars',
          'intercept':'Unpenalized for all three models',
          'training_store_ids':train.store_id.tolist(),
          'reserved_test_store_ids':split.loc[split['split'].eq('test'),'store_id'].tolist(),
          'test_status':'Unused; no predictions, summaries, scoring or tuning on reserved test outcomes',
          'planned_final_evaluation':'Only when explicitly requested: one locked comparison of these three training fits on the same 72 reserved stores. Primary metric RMSE in dollars; descriptive secondary MAE. Do not retune or switch the primary choice based on test results.',
          'saved_models':'frozen_sales_training_models.joblib',
          'model_file_sha256':hashlib.sha256(open('frozen_sales_training_models.joblib','rb').read()).hexdigest()}
with open('frozen_sales_model_decision.json','w') as f:
    json.dump(manifest,f,indent=2)
print(json.dumps({'primary_model':manifest['primary_model'],'lambda':lasso_lambda,'test_status':manifest['test_status']},indent=2))
