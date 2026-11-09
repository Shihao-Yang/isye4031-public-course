import csv
import json
import numpy as np
from scipy.optimize import minimize_scalar
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('paper_towels.csv',newline='') as f:
    rows=list(csv.reader(f))
y=np.array([float(r[1]) for r in rows[1:]])
def mafit(levels):
    w=np.diff(levels)
    def errors(theta):
        prev=0.; e=[]
        for value in w:
            prev=value-theta*prev
            e.append(prev)
        return np.array(e)
    opt=minimize_scalar(lambda theta:float(np.sum(errors(theta)**2)),bounds=(-.999,.999),method='bounded')
    return float(opt.x),errors(opt.x)
theta,e=mafit(y)
lb=acorr_ljungbox(e,lags=[6,12,18],model_df=1,return_df=True)
# Expanding-window, one-step forecasts; every model sees the same observations.
# Fit through week 90, predict 91; refit after each newly revealed observation.
preds={'Random walk':[],'ARIMA(0,1,1)':[],'ARIMA(1,1,0)':[]}
params=[]
for stop in range(90,len(y)):
    history=y[:stop]
    th,err=mafit(history)
    w=np.diff(history)
    phi=float(np.clip(np.dot(w[:-1],w[1:])/np.dot(w[:-1],w[:-1]),-.999,.999))
    preds['Random walk'].append(float(history[-1]))
    preds['ARIMA(0,1,1)'].append(float(history[-1]+th*err[-1]))
    preds['ARIMA(1,1,0)'].append(float(history[-1]+phi*w[-1]))
    params.append({'forecast_week':stop+1,'theta':th,'phi':phi})
metrics={}
for name,p in preds.items():
    ee=y[90:]-np.array(p)
    metrics[name]={'MAE':float(np.mean(abs(ee))),'RMSE':float(np.sqrt(np.mean(ee**2))),
                   'mean_error_actual_minus_forecast':float(np.mean(ee))}
aa=acf(e,nlags=20,fft=False)
results={'full_sample_theta':theta,'conditional_error_initialization':'e1=0; errors for weeks 2-120',
    'residual_mean':float(e.mean()),'residual_RMSE':float(np.sqrt(np.mean(e**2))),
    'residual_acf_lags_1_to_6':aa[1:7].tolist(),
    'Ljung_Box_model_df_1':{str(lag):{'Q':float(row.lb_stat),'df':int(lag-1),'p':float(row.lb_pvalue)} for lag,row in lb.iterrows()},
    'comparison':'Retrospective expanding-window one-step forecasts, weeks 91-120; models refit each week; no drift in all models',
    'forecast_metrics_in_recorded_units':metrics,'rolling_parameters':params}
with open('paper_towels_diagnostics_07_results.json','w') as f:
    json.dump(results,f,indent=2)
with open('paper_towels_diagnostics_07_forecasts.csv','w',newline='') as f:
    writer=csv.writer(f)
    writer.writerow(['week','actual']+list(preds))
    for i in range(30):
        writer.writerow([91+i,y[90+i]]+[p[i] for p in preds.values()])
fig,axes=plt.subplots(2,1,figsize=(10,6.2),constrained_layout=True)
axes[0].plot(range(2,121),e,color='#21618c',lw=1.2)
axes[0].axhline(0,color='gray',lw=.8)
axes[0].set(title='Full-sample ARIMA(0,1,1): fitted one-step errors',xlabel='Week',ylabel='Error (10,000 rolls)')
band=1.96/np.sqrt(len(e))
axes[1].axhspan(-band,band,alpha=.1,color='#21618c')
axes[1].vlines(range(1,21),0,aa[1:],color='#21618c',lw=1.6)
axes[1].plot(range(1,21),aa[1:],'o',ms=3,color='#21618c')
axes[1].axhline(0,color='gray',lw=.8)
axes[1].set(title='Residual ACF: approximate white-noise reference bands',xlabel='Lag (weeks)',ylabel='Correlation',ylim=(-.4,.4),xticks=[1,5,10,15,20])
for ax in axes:
    ax.grid(axis='y',alpha=.2)
    ax.spines[['top','right']].set_visible(False)
fig.savefig('paper_towels_diagnostics_07.png',dpi=170)
print(json.dumps({k:v for k,v in results.items() if k!='rolling_parameters'},indent=2))
