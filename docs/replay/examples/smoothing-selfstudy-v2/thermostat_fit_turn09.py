"""Fit Holt alpha and gamma with fixed OLS initialization. No randomness."""
from pathlib import Path
import csv
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
with (root/'thermostat.csv').open() as f:
    raw = list(csv.DictReader(f))
y = np.array([float(r['y']) for r in raw])
t = np.arange(1, len(y)+1)
assert y[23] == 206
x = t[:26]
b0 = ((x-x.mean())*(y[:26]-y[:26].mean())).sum()/((x-x.mean())**2).sum()
l0 = y[:26].mean()-b0*x.mean()
with (root/'thermostat_holt_updates_turn08.csv').open() as f:
    previous = list(csv.DictReader(f))
assert all(float(r['sales']) == actual for r, actual in zip(previous,y))
assert float(previous[23]['sales']) == 206

def holt(params):
    alpha, gamma = params
    level, trend = l0, b0
    rows = []
    for week, actual in zip(t,y):
        forecast = level+trend
        error = actual-forecast
        new_level = forecast+alpha*error
        new_trend = gamma*(new_level-level)+(1-gamma)*trend
        level,trend = new_level,new_trend
        rows.append([week, actual, forecast, error, level, trend])
    return np.array(rows)
def loss(params):
    return float(np.sum(holt(params)[:,3]**2))

# Full square scan plus deterministic local refinement and boundary searches.
grid = np.linspace(0,1,51)
candidates = sorted([(loss([a,g]),a,g) for a in grid for g in grid])
starts = [[r[1],r[2]] for r in candidates[:10]]
starts += [[a,g] for a in [0,.2,.5,1] for g in [0,.1,.5,1]]
fits = [minimize(loss,start,method='L-BFGS-B',bounds=[(0,1),(0,1)],
                 options={'ftol':1e-14,'gtol':1e-7,'maxiter':2000}) for start in starts]
best = min([np.array([r[1],r[2]]) for r in candidates[:1]]+[r.x for r in fits], key=loss)
for label, params in [('Original', [.2,.1]), ('Fitted', best)]:
    rs = holt(params)
    if label == 'Original':
        assert np.allclose(rs[:,2], [float(r['one_step_prediction']) for r in previous])
    e = rs[:,3]
    print(label, 'alpha,gamma=',params,'SSE=',loss(params),'MSD=',np.mean(e**2),
          'MAD=',np.mean(abs(e)),'MAPE=',100*np.mean(abs(e)/y),
          'L52=',rs[-1,4],'B52=',rs[-1,5])
    print('Forecasts:', [(h,rs[-1,4]+h*rs[-1,5]) for h in [1,2,4,8]])
print('Verified week24=206 in CSV and previous calculations; first26 sum=', y[:26].sum())

allrows=[]
forecasts=[]
summaries=[]
for label, params in [('Original', [.2,.1]), ('Fitted', best)]:
    rs=holt(params)
    allrows.extend([[label,*r] for r in rs])
    forecasts.extend([[label,52+h,h,rs[-1,4]+h*rs[-1,5]] for h in range(1,9)])
    e=rs[:,3]
    summaries.append([label,*params,loss(params),np.mean(e**2),np.mean(abs(e)),100*np.mean(abs(e)/y),rs[-1,4],rs[-1,5]])
for name,header,rows in [
    ('thermostat_fit_summary_turn09.csv',['method','alpha','gamma','SSE','MSD','MAD','MAPE_percent','L52','B52'],summaries),
    ('thermostat_fit_updates_turn09.csv',['method','week','actual','prediction','error','level','trend'],allrows),
    ('thermostat_forecasts_turn09.csv',['method','target_week','horizon','forecast'],forecasts)]:
    with (root/name).open('w',newline='') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)
fig,axes=plt.subplots(2,1,figsize=(11,7.8),sharex=True,gridspec_kw={'height_ratios':[2,1]})
axes[0].plot(t,y,'o-',color='#176b91',lw=1,markersize=3,label='Observed weekly sales')
for label,params,color in [('Original',[.2,.1],'#b85416'),('Fitted',best,'#7b4397')]:
    rs=holt(params)
    axes[0].plot(t,rs[:,2],color=color,lw=1.7,label=f'{label}: α={params[0]:.4f}, γ={params[1]:.4f}')
    axes[0].plot(range(53,61),[rs[-1,4]+h*rs[-1,5] for h in range(1,9)],'--',color=color,lw=2)
    axes[1].plot(t,rs[:,5],color=color,lw=1.7,label=label)
axes[0].set(title='Thermostat sales: original and fitted Holt models',ylabel='Sales (units)')
axes[0].legend(loc='upper left',fontsize=9)
axes[1].set(xlabel='Week',ylabel='Estimated trend\n(units per week)',xlim=(0,61))
axes[1].axhline(0,color='gray',lw=.8)
axes[1].legend(loc='upper left',fontsize=9)
for ax in axes:
    ax.axvspan(.5,26.5,color='gray',alpha=.08)
    ax.axvline(52.5,color='gray',ls=':',lw=1)
    ax.grid(axis='y',alpha=.2)
axes[1].set_xticks([1,13,26,39,52,56,60])
fig.tight_layout(pad=1.8)
fig.savefig(root/'thermostat_fit_turn09.png',dpi=170)
plt.close(fig)
