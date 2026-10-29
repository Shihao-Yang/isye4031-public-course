from pathlib import Path
import os
os.environ['MPLCONFIGDIR'] = str(Path.cwd() / '.mplcache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import hashlib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

cols = ['size_sqft','advertising_thousand_usd_month','staff_count',
        'area_household_income_thousand_usd_year','nearby_competitor_count']
# Record the fixed choice and previous comparison before opening the data file.
manifest = {'selected_model':'Curved ordinary least squares', 'selection_reason':'User chose simplicity after training-only comparison',
    'original_inputs':cols,'additional_term':'(advertising_thousand_usd_month - 28)^2',
    'training_role':'training','evaluation_role':'test','retune_after_test':False,
    'comparison_sha256':hashlib.sha256(Path('store_sales_04_nested_validation_results.csv').read_bytes()).hexdigest(),
    'comparison_scope':'Frozen. Evaluate selected model only; training-mean reference is a fixed baseline.'}
Path('store_sales_06_frozen_selection.json').write_text(json.dumps(manifest,indent=2))
d = pd.read_csv('store_sales_01_data.csv')
train = d.loc[d.split.eq('training')].copy()
test = d.loc[d.split.eq('test')].copy()
def design(frame):
    x = frame[cols].to_numpy().copy()
    x[:,1] -= 28
    return np.column_stack([x,x[:,1]**2])
X = design(train)
y = train.sales_thousand_usd_month.to_numpy()
model = make_pipeline(StandardScaler(),LinearRegression()).fit(X,y)
# Persist the fitted coefficients before computing held-out predictions.
coef = model[-1].coef_/model[0].scale_
intercept = model[-1].intercept_ - coef @ model[0].mean_
Path('store_sales_06_frozen_coefficients.json').write_text(json.dumps({'intercept':float(intercept),
    'feature_names':cols+['centered_advertising_squared'],'coefficients':coef.tolist(),
    'advertising_center':28,'training_rows':len(train)},indent=2))
truth = test.sales_thousand_usd_month.to_numpy()
pred = model.predict(design(test))
baseline = np.full(len(test), y.mean())
def metrics(actual,p):
    return {'rmse_thousand_usd':float(np.sqrt(mean_squared_error(actual,p))),
            'mae_thousand_usd':float(mean_absolute_error(actual,p)),
            'r_squared':float(r2_score(actual,p)),
            'mean_predicted_minus_actual_thousand_usd':float(np.mean(p-actual))}
results = {'selected_model':'Curved OLS','test_stores':len(test),'test_metrics':metrics(truth,pred),
           'training_mean_reference_test_metrics':metrics(truth,baseline),
           'training_advertising_range_thousand_usd':[float(train[cols[1]].min()),float(train[cols[1]].max())]}
validation = pd.read_csv('store_sales_04_nested_validation_results.csv')
results['previous_validation_rmse_thousand_usd'] = float(validation.loc[validation.model.eq('Curved OLS'),'nested_validation_rmse'].iloc[0])
reference = train[cols].median().to_frame().T
scenario = pd.concat([reference]*2,ignore_index=True)
scenario[cols[1]] = [100,150]
scenario['predicted_sales_thousand_usd_month'] = model.predict(design(scenario))
scenario.to_csv('store_sales_06_extrapolation_scenarios.csv',index=False)
results['extrapolation_illustrations'] = scenario[[cols[1],'predicted_sales_thousand_usd_month']].to_dict('records')
Path('store_sales_06_test_results.json').write_text(json.dumps(results,indent=2))
pd.DataFrame({'store_id':test.store_id,'actual_sales_thousand_usd':truth,
              'predicted_sales_thousand_usd':pred,'residual_actual_minus_predicted':truth-pred}).to_csv('store_sales_06_reserved_store_predictions.csv',index=False)
plt.rcParams.update({'font.size':11,'axes.titlesize':13,'axes.labelsize':12})
fig,axes = plt.subplots(1,2,figsize=(12,4.8))
lo = min(truth.min(),pred.min())-10
hi = max(truth.max(),pred.max())+10
axes[0].scatter(truth,pred,s=33,alpha=.8,color='#2166ac',edgecolors='white',linewidth=.4)
axes[0].plot([lo,hi],[lo,hi],color='gray',ls='--',lw=1.5)
axes[0].set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Actual sales ($1,000/month)',
            ylabel='Predicted sales ($1,000/month)',title='Frozen curved fit: 75 reserved stores')
grid = np.linspace(0,160,300)
curve = pd.concat([reference]*len(grid),ignore_index=True)
curve[cols[1]] = grid
amin,amax = results['training_advertising_range_thousand_usd']
axes[1].axvspan(amin,amax,color='#2166ac',alpha=.12,label='Training advertising range')
axes[1].plot(grid,model.predict(design(curve)),color='#c45118',lw=2.5,label='Fitted quadratic extension')
axes[1].scatter([100,150],scenario.predicted_sales_thousand_usd_month,color='#c45118',zorder=3)
axes[1].axhline(0,color='gray',ls=':',lw=1)
axes[1].set(xlabel='Advertising ($1,000/month)',ylabel='Predicted sales ($1,000/month)',title='Extrapolation with other inputs at training medians')
axes[1].legend(fontsize=9,loc='lower left')
for ax in axes:
    ax.grid(alpha=.2)
    ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
fig.savefig('store_sales_06_test_and_extrapolation.png',dpi=180)
print(json.dumps(results,indent=2))
