"""Single final evaluation of already frozen models on reserved stores."""
import os
os.environ['MPLCONFIGDIR']='.mplcache'
import json
import hashlib
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

output=Path('frozen_sales_final_test_results.json')
if output.exists():
    raise RuntimeError('Final evaluation already saved; inspect existing results without evaluating again.')
with open('frozen_sales_model_decision.json') as f:
    manifest=json.load(f)
for path,key in [('constructed_sales_seed4031.csv','dataset_sha256'),
                 (manifest['saved_models'],'model_file_sha256')]:
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==manifest[key], 'Frozen input hash mismatch'
models=joblib.load(manifest['saved_models'])
data=pd.read_csv('constructed_sales_seed4031.csv').set_index('store_id')
test=data.loc[manifest['reserved_test_store_ids']]
assert len(test)==72
assert set(manifest['training_store_ids']).isdisjoint(test.index)
X=test[manifest['predictors_in_order']]
y=test.weekly_sales.to_numpy()
predictions=pd.DataFrame({'store_id':test.index,'observed_sales_dollars':y})
metrics=[]
for name in ['Lasso','Ridge','Elastic Net']:
    prediction=models[name].predict(X)*1000
    residual=y-prediction
    predictions[f'{name}_prediction_dollars']=prediction
    predictions[f'{name}_residual_dollars']=residual
    metrics.append({'model':name,'test_n':len(y),
                    'RMSE_dollars':float(np.sqrt(np.mean(residual**2))),
                    'MAE_dollars':float(np.mean(np.abs(residual))),
                    'mean_residual_dollars':float(residual.mean())})
predictions.to_csv('frozen_sales_final_test_predictions.csv',index=False)
pd.DataFrame(metrics).to_csv('frozen_sales_final_test_metrics.csv',index=False)
lasso_rmse=metrics[0]['RMSE_dollars']
result={'dataset':'Original constructed sales, seed 4031',
        'evaluation':'Single final evaluation on 72 previously reserved stores using hash-verified frozen training fits. No refitting, retuning, preprocessing updates or model switching.',
        'primary_model':manifest['primary_model'],'primary_metric':'RMSE in dollars; MAE is secondary',
        'metrics':metrics,
        'other_RMSE_minus_Lasso_dollars':{m['model']:m['RMSE_dollars']-lasso_rmse for m in metrics[1:]},
        'frozen_models_sha256':manifest['model_file_sha256'],
        'limits':'Independent held-out evaluation of the fixed training fits under this synthetic population. One 72-store split does not establish population ranking or equivalence. No confidence intervals are claimed; no causal effects or true-variable recovery are established.',
        'decision':'Retain preselected Lasso as primary model; report all locked comparison results. Test outcomes will not be used to revise the model.',
        'test_status':'Consumed for final evaluation; no longer an untouched test set.'}
with output.open('x') as f:
    json.dump(result,f,indent=2)

fig,axes=plt.subplots(1,2,figsize=(11,5.8))
colors=['#1764a0','#888888','#c3872c']
bars=axes[0].bar([m['model'] for m in metrics],[m['RMSE_dollars'] for m in metrics],color=colors,width=.6)
for bar,m in zip(bars,metrics):
    axes[0].text(bar.get_x()+bar.get_width()/2,bar.get_height()+45,f"${m['RMSE_dollars']:,.0f}",ha='center',fontsize=11)
axes[0].set(ylabel='Final test RMSE ($ per store-week)',ylim=(0,max(m['RMSE_dollars'] for m in metrics)*1.18),title='Locked comparison: same 72 reserved stores')
prediction=predictions['Lasso_prediction_dollars'].to_numpy()/1000
observed=y/1000
axes[1].scatter(prediction,observed,s=32,alpha=.75,color=colors[0])
low=min(prediction.min(),observed.min())-1
high=max(prediction.max(),observed.max())+1
axes[1].plot([low,high],[low,high],color='black',lw=1)
axes[1].set(xlabel='Frozen Lasso predicted sales ($1,000)',ylabel='Observed sales ($1,000)',
            xlim=(low,high),ylim=(low,high),title='Primary model: final test prediction agreement')
for ax in axes:
    ax.grid(axis='y',alpha=.17)
    ax.set_axisbelow(True)
fig.suptitle('Final held-out evaluation: constructed sales data',fontsize=15,y=.97)
fig.subplots_adjust(left=.085,right=.98,bottom=.24,top=.86,wspace=.28)
fig.text(.085,.055,'Models and preprocessing were frozen using 168 training stores before this evaluation.\nLasso remains the preselected primary model. No tuning or model switching used these test outcomes.\nResults describe one 72-store holdout from the constructed population; the test set is now consumed.',fontsize=10)
fig.savefig('frozen_sales_final_test_comparison.png',dpi=160)
plt.close(fig)
print(json.dumps(result,indent=2))
