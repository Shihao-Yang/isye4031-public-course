from pathlib import Path
import os
os.environ['MPLCONFIGDIR'] = str(Path.cwd() / '.mplcache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

d = pd.read_csv('hotel_monthly.csv').query('t <= 156').copy()
t=d.t.to_numpy(); month=d.month.to_numpy(); y=d.rooms_per_day.to_numpy(dtype=float)
assert np.all(y>0)
origins=[96,108,120,132,144]
month_names=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
def design(ts,ms,seasonal=True):
    base=np.column_stack([np.ones(len(ts)),np.asarray(ts)/12])
    return np.column_stack([base,np.asarray(ms)[:,None]==np.arange(2,13)]) if seasonal else base

def fit_predict(ts,ms,ys,new_t,new_m,logged,seasonal):
    target=np.log(ys) if logged else ys
    x=design(ts,ms,seasonal)
    coef=np.linalg.lstsq(x,target,rcond=None)[0]
    eta=design(new_t,new_m,seasonal)@coef
    smear=float(np.mean(np.exp(target-x@coef))) if logged else 1.0
    pred=np.exp(eta)*smear if logged else eta
    return pred,coef,smear,target-x@coef

specs=[('Linear trend only',False,False),('Trend + months: level',False,True),('Trend + months: log',True,True)]
rows=[]; folds=[]; outputs=[]; full={}; coefficients=[]
for name,logged,seasonal in specs:
    errs=[]
    for origin in origins:
        tr=t<=origin; va=(t>origin)&(t<=origin+12)
        pred,b,smear,_=fit_predict(t[tr],month[tr],y[tr],t[va],month[va],logged,seasonal)
        error=y[va]-pred
        errs.extend(error)
        folds.append({'model':name,'training_end':origin,'validation_start':origin+1,'validation_end':origin+12,
                      'rmse_rooms_per_day':float(np.sqrt(np.mean(error**2))),'training_smearing_factor':smear})
        for tt,actual,pp in zip(t[va],y[va],pred):
            outputs.append({'model':name,'origin':origin,'t':int(tt),'actual':actual,'forecast':pp})
    pred,b,smear,resid=fit_predict(t,month,y,t,month,logged,seasonal)
    full[name]={'pred':pred,'coef':b,'smear':smear,'resid':resid}
    rows.append({'model':name,'development_rmse_rooms_per_day':float(np.sqrt(np.mean((y-pred)**2))),
                 'validation_rmse_rooms_per_day':float(np.sqrt(np.mean(np.array(errs)**2))),
                 'validation_mae_rooms_per_day':float(np.mean(np.abs(errs)))})
    if seasonal:
        delta=np.r_[0,b[2:]]
        for m,c in enumerate(delta,1):
            coefficients.append({'model':name,'month':m,'month_name':month_names[m-1],
                                 'coefficient_vs_january':float(c),
                                 'percent_vs_january':float(100*np.expm1(c)) if logged else None})
results=pd.DataFrame(rows)
results.to_csv('hotel_03_seasonal_validation_results.csv',index=False)
pd.DataFrame(folds).to_csv('hotel_03_expanding_validation_details.csv',index=False)
pd.DataFrame(outputs).to_csv('hotel_03_validation_forecasts.csv',index=False)
pd.DataFrame(coefficients).to_csv('hotel_03_seasonal_coefficients.csv',index=False)
fit_out=d.copy()
for name,_,_ in specs:
    fit_out[name+'_fitted']=full[name]['pred']
    fit_out[name+'_model_scale_residual']=full[name]['resid']
fit_out.to_csv('hotel_03_development_fits.csv',index=False)
summary={'development_t':[1,156],'reserved_t':[157,168],'reserved_used':False,
         'january_reference':True,'time_predictor':'t/12, so trend coefficients are per year',
         'validation_origins':origins,'forecast_horizon':12,
         'retransformation':'exp(log prediction) times mean(exp(training log residual)); recomputed within each training block',
         'level_seasonal_trend_per_month':float(full['Trend + months: level']['coef'][1]/12),
         'log_seasonal_trend_per_month':float(full['Trend + months: log']['coef'][1]/12),
         'log_seasonal_growth_percent_per_year':float(100*np.expm1(full['Trend + months: log']['coef'][1])),
         'full_development_smearing_factor':full['Trend + months: log']['smear']}
Path('hotel_03_seasonal_log_summary.json').write_text(json.dumps(summary,indent=2))
plt.rcParams.update({'font.size':11,'axes.titlesize':13,'axes.labelsize':11})
fig,axes=plt.subplots(2,1,figsize=(10.5,7.3))
recent=t>=109
axes[0].plot(t[recent],y[recent],'o-',color='#333333',lw=1.5,ms=3,label='Observed')
for name,color in [('Trend + months: level','#2166ac'),('Trend + months: log','#c45118')]:
    axes[0].plot(t[recent],full[name]['pred'][recent],color=color,lw=2,label=name)
axes[0].set(title='Seasonal fits: last four development years',xlabel='Month index t',ylabel='Average rooms occupied per day')
axes[0].legend(fontsize=10,ncol=3,loc='upper left')
# Residuals shown on each model's fitted scale to reveal temporal structure.
axes[1].plot(t,full['Trend + months: log']['resid'],color='#c45118',lw=1.5)
axes[1].axhline(0,color='gray',lw=1)
axes[1].set(title='Log trend + month indicators: residuals still have structure',
            xlabel='Month index t',ylabel='Log occupancy minus fitted log occupancy')
for ax in axes:
    ax.grid(alpha=.2)
    ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
fig.savefig('hotel_03_seasonal_fits_and_log_residuals.png',dpi=180)
print(results.to_string(index=False))
print(pd.DataFrame(coefficients).query('month in [2,7,8,11]').to_string(index=False))
print(json.dumps(summary,indent=2))
