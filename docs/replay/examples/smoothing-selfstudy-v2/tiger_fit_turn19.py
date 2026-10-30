"""Multiplicative HW fit with fixed turn-18 initialization. No randomness."""
from pathlib import Path
import csv
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
with (root/'tiger.csv').open() as f:y=np.array([float(r['y']) for r in csv.DictReader(f)])
t=np.arange(1,17);v=y[:16]
b0=np.sum((t-t.mean())*(v-v.mean()))/np.sum((t-t.mean())**2);l0=v.mean()-b0*t.mean()
raw=((v/(l0+b0*t)).reshape(4,4)).mean(axis=0);s0=raw/raw.mean()
def run(params):
    a,g,d=params;l,b=l0,b0;s=list(s0);rows=[]
    for i,actual in enumerate(y):
        q=i%4;p=(l+b)*s[q];error=actual-p
        new_l=a*(actual/s[q])+(1-a)*(l+b)
        new_b=g*(new_l-l)+(1-g)*b
        s[q]=d*actual/new_l+(1-d)*s[q]
        l,b=new_l,new_b
        rows.append([i+1,actual,p,error,l,b,s[q]])
    forecasts=[(l+h*b)*s[(len(y)+h-1)%4] for h in range(1,9)]
    return np.array(rows),forecasts
def loss(params):
    rs,_=run(params)
    return float(np.sum(rs[:,3]**2))
fits=[minimize(loss,[a,g,d],method='L-BFGS-B',bounds=[(0,1)]*3,
               options={'ftol':1e-14,'gtol':1e-7,'maxiter':2000})
      for a in [.05,.3,.7,.95] for g in [0,.1,.5,1] for d in [0,.1,.5,1]]
best=min(fits,key=lambda r:r.fun).x
variants=[('Initial',[.2,.1,.1]),('Optimized',best),('Historical supplied constants',[.336,.046,.134])]
summaries=[];allrows=[];fc=[];stored={}
for name,params in variants:
    rows,f=run(params);e=rows[:,3]
    if name=='Initial':
        with (root/'tiger_updates_turn18.csv').open() as ff:prev=list(csv.DictReader(ff))
        assert np.allclose(rows[:,2],[float(r['prediction']) for r in prev])
    summary=[name,*params,np.sum(e**2),np.mean(abs(e)),np.mean(e**2),100*np.mean(abs(e)/y),rows[-1,4],rows[-1,5]]
    summaries.append(summary);stored[name]=(rows,f)
    allrows.extend([[name,*r] for r in rows]);fc.extend([[name,32+h,h,p] for h,p in enumerate(f,1)])
    print('SUMMARY',summary);print('FORECAST',name,f)
for filename,header,rs in [
 ('tiger_fit_summary_turn19.csv',['method','alpha','gamma','delta','SSE','MAD','MSD','MAPE_percent','L32','B32'],summaries),
 ('tiger_fitted_values_turn19.csv',['method','quarter','actual','fitted','error','level','trend','seasonal'],allrows),
 ('tiger_forecasts_turn19.csv',['method','target_quarter','horizon','forecast'],fc)]:
    with (root/filename).open('w',newline='') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rs)
with (root/'software_reference.csv').open() as f:
    refs=[r for r in csv.DictReader(f) if r['example']=='tiger']
print('Historical references',refs)
# Sensitivity check motivated by the earlier bike comparison. Keep the
# first-16 seasonal multipliers, changing only L0 and B0 to first-year OLS.
saved_l,saved_b=l0,b0
tx=np.arange(1,5)
b0=np.sum((tx-tx.mean())*(y[:4]-y[:4].mean()))/np.sum((tx-tx.mean())**2)
l0=y[:4].mean()-b0*tx.mean()
print('First-year sensitivity states:',l0,b0)
sensitivity=[]
for label,params in [('Initial',[.2,.1,.1]),('Historical supplied constants',[.336,.046,.134])]:
    rs,ff=run(params);ee=rs[:,3]
    row=[label,l0,b0,np.mean(abs(ee)),np.mean(ee**2),100*np.mean(abs(ee)/y)]
    sensitivity.append(row);print('SENSITIVITY',row)
with (root/'tiger_init_sensitivity_turn19.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['method','L0','B0','MAD','MSD','MAPE_percent']);w.writerows(sensitivity)
l0,b0=saved_l,saved_b
fig,axes=plt.subplots(1,2,figsize=(12.5,5.6),gridspec_kw={'width_ratios':[1.5,1]})
axes[0].plot(range(1,33),y,'o-',color='#176b91',markersize=3.5,label='Observed sales')
for name,color in [('Initial','#b85416'),('Optimized','#7b4397')]:
    rows,f=stored[name]
    axes[0].plot(range(1,33),rows[:,2],color=color,lw=1.7,label=name+' fitted values')
    axes[0].plot(range(33,41),f,'s--',color=color,markersize=3.5,label=name+' forecasts')
axes[0].axvspan(.5,16.5,color='gray',alpha=.08)
axes[0].axvline(32.5,color='gray',ls=':',lw=1)
axes[0].set(title='Tiger: fitted curves and two-year forecasts',xlabel='Quarter index',ylabel='Sales (thousands of cases)',xticks=[1,8,16,24,32,36,40])
axes[0].legend(loc='upper left',fontsize=8)
xx=np.arange(2);width=.24
axes[1].bar(xx-width,[summaries[0][6],summaries[1][6]],width,color=['#b85416','#7b4397'],label='Our initial / optimized')
axes[1].bar(xx,[float(r['MSD']) for r in refs],width,color='#176b91',label='Historical reference MSD')
axes[1].bar([1+width],[summaries[2][6]],width,color='#558344',label='Reported constants, our initialization')
axes[1].set(title='Historical squared-error comparison',ylabel='MSD ((thousands of cases)²)',xticks=xx,xticklabels=['Initial','Supplied / fitted constants'])
axes[1].legend(loc='upper right',fontsize=7.5)
for ax in axes:ax.grid(axis='y',alpha=.2)
fig.tight_layout(pad=1.8,w_pad=2)
fig.savefig(root/'tiger_fit_turn19.png',dpi=170)
plt.close(fig)
