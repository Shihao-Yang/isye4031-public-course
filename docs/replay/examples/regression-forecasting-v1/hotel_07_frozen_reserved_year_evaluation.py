from pathlib import Path
import os
os.environ['MPLCONFIGDIR']=str(Path.cwd()/'.mplcache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import hashlib

manifest={'origin':156,'forecast_months':[157,168],'horizon':12,
 'selected_model':'OLS log occupancy on linear time trend and 11 month indicators, January reference',
 'time_term':'t/12','retransformation':'exp(predicted log occupancy) * mean(exp(training log residual))',
 'benchmark':'Seasonal naive: forecast(t)=observed(t-12)',
 'training_months':[1,156],'updates_during_forecast_year':False,
 'comparison_frozen':True,'no_post_test_tuning':True,
 'prior_validation_sha256':hashlib.sha256(Path('hotel_04_seasonal_naive_comparison.csv').read_bytes()).hexdigest(),
 'ar1_comparison_sha256':hashlib.sha256(Path('hotel_06_ar1_validation_metrics.csv').read_bytes()).hexdigest()}
Path('hotel_07_frozen_forecast_design.json').write_text(json.dumps(manifest,indent=2))
d=pd.read_csv('hotel_monthly.csv')
tr=d.loc[d.t<=156].copy()
def design(ts,ms):
    return np.column_stack([np.ones(len(ts)),np.asarray(ts)/12,
                            np.asarray(ms)[:,None]==np.arange(2,13)])
x=design(tr.t,tr.month); logy=np.log(tr.rooms_per_day.to_numpy())
b=np.linalg.lstsq(x,logy,rcond=None)[0]
smear=float(np.mean(np.exp(logy-x@b)))
future_t=np.arange(157,169); future_month=(future_t-1)%12+1
pred=np.exp(design(future_t,future_month)@b)*smear
naive=tr.set_index('t').loc[future_t-12,'rooms_per_day'].to_numpy(dtype=float)
forecast=pd.DataFrame({'t':future_t,'month':future_month,'horizon':np.arange(1,13),
                       'log_seasonal_forecast':pred,'repeat_last_year_forecast':naive})
# Persist every forecast and fitted parameter before accessing reserved outcomes.
forecast.to_csv('hotel_07_frozen_reserved_year_forecasts.csv',index=False)
Path('hotel_07_frozen_coefficients.json').write_text(json.dumps({'intercept':float(b[0]),
 'trend_per_year':float(b[1]),'month_effects_Feb_to_Dec':b[2:].tolist(),
 'smearing_factor':smear},indent=2))
actual=d.set_index('t').loc[future_t,'rooms_per_day'].to_numpy(dtype=float)
evaluation=forecast.copy(); evaluation['actual']=actual
rows=[]
for name,p in [('Log trend + months',pred),('Repeat last year',naive)]:
    e=actual-p
    rows.append({'method':name,'rmse_rooms_per_day':float(np.sqrt(np.mean(e**2))),
                 'mae_rooms_per_day':float(np.mean(np.abs(e))),
                 'mean_forecast_minus_actual_rooms_per_day':float(np.mean(p-actual)),
                 'max_absolute_error_rooms_per_day':float(np.max(np.abs(e)))})
evaluation['log_seasonal_error_actual_minus_forecast']=actual-pred
evaluation['repeat_last_year_error_actual_minus_forecast']=actual-naive
evaluation.to_csv('hotel_07_reserved_year_evaluation_by_month.csv',index=False)
metrics=pd.DataFrame(rows)
metrics.to_csv('hotel_07_reserved_year_metrics.csv',index=False)
summary={'test_months':12,'test_origin':156,'forecasts_updated':False,
 'log_model_rmse_reduction_percent_vs_benchmark':float(100*(1-rows[0]['rmse_rooms_per_day']/rows[1]['rmse_rooms_per_day'])),
 'months_log_model_has_lower_absolute_error':int(np.sum(np.abs(actual-pred)<np.abs(actual-naive))),
 'actual_average_rooms_per_day':float(actual.mean()),
 'log_model_forecast_average_rooms_per_day':float(pred.mean()),
 'benchmark_forecast_average_rooms_per_day':float(naive.mean()),
 'limits':['Only one final 12-month block; errors within it may be dependent',
           'Earlier residual dependence remains; calibrated prediction intervals unresolved',
           'Future stability of exponential trend and seasonal ratios unverified',
           'No hotel capacity or external predictors provided; occupancy rates cannot be inferred',
           'Final block is now evaluated and cannot be reused as untouched evidence after revisions']}
Path('hotel_07_reserved_year_summary.json').write_text(json.dumps(summary,indent=2))
plt.rcParams.update({'font.size':12,'axes.titlesize':14,'axes.labelsize':12})
fig,ax=plt.subplots(figsize=(10,4.8))
ax.plot(future_month,actual,'o-',color='#222222',lw=2,label='Reserved-year observations')
ax.plot(future_month,pred,'s--',color='#c45118',lw=2,ms=5,label='Frozen log seasonal forecast')
ax.plot(future_month,naive,'^--',color='#2166ac',lw=1.8,ms=5,label='Frozen repeat-last-year forecast')
ax.set(xticks=range(1,13),xticklabels=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
       xlabel='Month of reserved year 14',ylabel='Average rooms occupied per day',
       title='Forecasts issued at month 156, with no updates during the reserved year')
ax.grid(alpha=.2); ax.spines[['top','right']].set_visible(False)
ax.legend(fontsize=10,loc='upper left')
fig.tight_layout(); fig.savefig('hotel_07_reserved_year_forecast_comparison.png',dpi=180)
print(metrics.to_string(index=False))
print(evaluation[['month','actual','log_seasonal_forecast','repeat_last_year_forecast']].to_string(index=False))
print(json.dumps(summary,indent=2))
