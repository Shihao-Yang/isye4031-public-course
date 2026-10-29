from pathlib import Path
import os
os.environ['MPLCONFIGDIR'] = str(Path.cwd() / '.mplcache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

d=pd.read_csv('hotel_monthly.csv').query('t <= 156').copy()
def design(t,m):
    return np.column_stack([np.ones(len(t)),np.asarray(t)/12,
                            np.asarray(m)[:,None]==np.arange(2,13)])
rows=[]; metrics=[]; example={}
for origin in [96,108,120,132,144]:
    tr=d.loc[d.t<=origin]; va=d.loc[(d.t>origin)&(d.t<=origin+12)]
    x=design(tr.t,tr.month); z=np.log(tr.rooms_per_day.to_numpy())
    b=np.linalg.lstsq(x,z,rcond=None)[0]
    smear=float(np.mean(np.exp(z-x@b)))
    log_eta=design(va.t,va.month)@b
    model=np.exp(log_eta)*smear
    # All twelve seasonal-naive inputs are known at the forecast origin.
    naive=d.set_index('t').loc[va.t.to_numpy()-12,'rooms_per_day'].to_numpy(dtype=float)
    actual=va.rooms_per_day.to_numpy(dtype=float)
    for j,tt in enumerate(va.t):
        rows.append({'origin':origin,'t':int(tt),'horizon':int(tt-origin),
                     'month':int(va.month.iloc[j]),'actual':actual[j],
                     'log_seasonal_forecast':model[j],'repeat_last_year_forecast':naive[j]})
    for name,pred in [('Log trend + months',model),('Repeat last year',naive)]:
        metrics.append({'origin':str(origin),'method':name,
                        'rmse':float(np.sqrt(np.mean((actual-pred)**2))),
                        'mae':float(np.mean(np.abs(actual-pred)))})
    if origin==120:
        j=int(np.flatnonzero(va.t.to_numpy()==128)[0])
        example={'origin':origin,'training_t':[1,120],'forecast_t':128,'month':'August of year 11',
                 'horizon_months':8,'intercept':float(b[0]),'trend_coefficient_per_month':float(b[1]/12),
                 'trend_contribution':float(b[1]*128/12),'august_coefficient':float(b[8]),
                 'predicted_log_occupancy':float(log_eta[j]),'smearing_factor':smear,
                 'exp_log_forecast':float(np.exp(log_eta[j])),
                 'mean_forecast_rooms_per_day':float(model[j]),'repeat_last_year_value':float(naive[j]),
                 'observed_later':float(actual[j]),
                 'model_actual_minus_forecast':float(actual[j]-model[j]),
                 'naive_actual_minus_forecast':float(actual[j]-naive[j]),
                 'annual_fitted_growth_percent':float(100*np.expm1(b[1])),
                 'final_reserved_used':False}
predictions=pd.DataFrame(rows)
for name,col in [('Log trend + months','log_seasonal_forecast'),('Repeat last year','repeat_last_year_forecast')]:
    e=predictions.actual-predictions[col]
    metrics.append({'origin':'pooled five origins','method':name,'rmse':float(np.sqrt(np.mean(e**2))),
                    'mae':float(np.mean(np.abs(e)))})
predictions.to_csv('hotel_04_earlier_origin_forecasts.csv',index=False)
results=pd.DataFrame(metrics)
results.to_csv('hotel_04_seasonal_naive_comparison.csv',index=False)
Path('hotel_04_august_forecast_calculation.json').write_text(json.dumps(example,indent=2))
a=predictions.loc[predictions.origin.eq(120)]
history=d.loc[(d.t>=109)&(d.t<=120)]
plt.rcParams.update({'font.size':12,'axes.titlesize':14,'axes.labelsize':12})
fig,ax=plt.subplots(figsize=(10,4.8))
ax.plot(history.t,history.rooms_per_day,'o-',color='#555555',lw=1.5,ms=4,label='Known history')
ax.plot(a.t,a.actual,'o-',color='#222222',lw=1.8,ms=4,label='Observed later')
ax.plot(a.t,a.log_seasonal_forecast,'s--',color='#c45118',lw=2,ms=4,label='Log trend + months forecast')
ax.plot(a.t,a.repeat_last_year_forecast,'^--',color='#2166ac',lw=1.8,ms=4,label='Repeat last year forecast')
ax.axvline(120.5,color='#777777',ls=':',lw=1.5)
ax.axvspan(120.5,132.5,color='#999999',alpha=.08)
ax.set(xlim=(108.5,132.5),xlabel='Month index t',ylabel='Average rooms occupied per day',
       title='Forecasts issued after month 120: no updates during the next year')
ax.set_xticks([109,112,115,118,121,124,127,130,132])
ax.grid(alpha=.2)
ax.legend(fontsize=9,ncol=2,loc='upper left')
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
fig.savefig('hotel_04_origin_120_forecast_comparison.png',dpi=180)
print(json.dumps(example,indent=2))
print(results.to_string(index=False))
