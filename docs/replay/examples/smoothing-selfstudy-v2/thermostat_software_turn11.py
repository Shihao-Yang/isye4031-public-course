"""Replot transcribed reports and compare explicit initializations.
Deterministic formula recreations, not current Minitab output; no seed used.
"""
from pathlib import Path
import csv
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
with (root/'thermostat.csv').open() as f:
    y=np.array([float(r['y']) for r in csv.DictReader(f)])
with (root/'software_reference.csv').open() as f:
    refs=[r for r in csv.DictReader(f) if r['example']=='thermostat']
with (root/'thermostat_fit_summary_turn09.csv').open() as f:
    fit=next(r for r in csv.DictReader(f) if r['method']=='Fitted')
with (root/'thermostat_intervals_turn10.csv').open() as f:
    ours=[r for r in csv.DictReader(f) if r['origin_week']=='52']
t=np.arange(1,53)
x=t[:26]
b0=((x-x.mean())*(y[:26]-y[:26].mean())).sum()/((x-x.mean())**2).sum()
l0=y[:26].mean()-b0*x.mean()
states=[('First-26 regression',l0,b0),('First-26 mean, zero trend',y[:26].mean(),0.),
        ('First observation, zero trend',y[0],0.)]
def run(params,initial):
    a,g=params
    l,b=initial
    pred=[]
    for actual in y:
        p=l+b
        error=actual-p
        l=p+a*error
        b=b+a*g*error
        pred.append(p)
    return np.array(pred),l,b
results=[]
for name,l,b in states:
    def loss(params):
        return np.sum((y-run(params,(l,b))[0])**2)
    fits=[minimize(loss,[a,g],bounds=[(0,1),(0,1)],method='L-BFGS-B',
                   options={'ftol':1e-14,'gtol':1e-7})
          for a in [.05,.25,.5,.9] for g in [0,.1,.5,1]]
    best=min(fits,key=lambda r:r.fun).x
    for mode,params in [('Fixed reported constants',(.247,.095)),('Refitted',best)]:
        p,lf,bf=run(params,(l,b))
        e=y-p
        row=[name,mode,l,b,*params,np.mean(abs(e)),np.mean(e**2),100*np.mean(abs(e)/y),lf,bf,*[lf+h*bf for h in [1,2,3]]]
        results.append(row)
        print(row)
for label,params in [('Rounded Solver constants',(.247,.095)),('Reported automatic constants',(.489596,.066912))]:
    p,l,b=run(params,(l0,b0))
    print(label,'our initialization:', 'MSD=',np.mean((y-p)**2),'forecasts=',[l+h*b for h in [1,2,3]])
with (root/'thermostat_initialization_turn11.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['initialization','parameter_choice','L0','B0','alpha','gamma','MAD','MSD','MAPE','L52','B52','forecast53','forecast54','forecast55']);w.writerows(results)
fig,axes=plt.subplots(1,2,figsize=(12.7,5.8))
colors=['#176b91','#b85416','#7b4397']
for method,color,offset in zip(['Minitab_with_Solver_constants','Minitab_automatic'],colors,[-.12,0]):
    rs=[r for r in refs if r['reported_method']==method]
    xs=np.array([52+int(r['horizon']) for r in rs])+offset
    p=np.array([float(r['forecast']) for r in rs])
    lo=np.array([float(r['lower95']) for r in rs]);hi=np.array([float(r['upper95']) for r in rs])
    label='Reference: supplied constants' if method.endswith('constants') else 'Reference: automatic'
    axes[0].errorbar(xs,p,yerr=[p-lo,hi-p],fmt='o-',capsize=4,color=color,label=label)
p=np.array([float(r['forecast']) for r in ours])
lo=np.array([float(r['lower95']) for r in ours]);hi=np.array([float(r['upper95']) for r in ours])
axes[0].errorbar(np.arange(53,56)+.12,p,yerr=[p-lo,hi-p],fmt='s--',capsize=4,color=colors[2],label='Our fitted Holt + normal intervals')
axes[0].set(title='Transcribed reference forecasts vs our calculation',xlabel='Target week (positions offset for visibility)',ylabel='Thermostats sold (units)',xticks=[53,54,55])
axes[0].legend(fontsize=8,loc='upper left')
for state,color in zip(states,colors):
    row=next(r for r in results if r[0]==state[0] and r[1]=='Refitted')
    axes[1].plot([53,54,55],row[-3:],'o-',color=color,label=f'{state[0]}\nα={row[4]:.3f}, γ={row[5]:.3f}')
axes[1].set(title='Change initialization, then refit α and γ',xlabel='Target week',ylabel='Point forecast (units)',xticks=[53,54,55])
axes[1].legend(fontsize=8,loc='upper left')
for ax in axes: ax.grid(axis='y',alpha=.2)
fig.tight_layout(pad=1.8,w_pad=2)
fig.savefig(root/'thermostat_software_turn11.png',dpi=170)
plt.close(fig)
