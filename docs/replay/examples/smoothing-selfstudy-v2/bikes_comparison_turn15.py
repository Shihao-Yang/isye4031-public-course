"""Initial/optimized HW fits and explicit historical-comparison attempts.
All calculations deterministic; no random seed used.
"""
from pathlib import Path
import csv
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
def read(name):
    with (root/name).open() as f:return np.array([float(r['y']) for r in csv.DictReader(f)])
def init(y,n,season_n):
    t=np.arange(1,n+1)
    b=np.sum((t-t.mean())*(y[:n]-y[:n].mean()))/np.sum((t-t.mean())**2)
    l=y[:n].mean()-b*t.mean()
    tt=np.arange(1,season_n+1)
    s=(y[:season_n]-l-b*tt).reshape(-1,4).mean(axis=0)
    return l,b,s
def run(y,params,state):
    a,g,d=params;l,b,s=state[0],state[1],list(state[2])
    rows=[]
    for t,actual in enumerate(y,1):
        j=(t-1)%4;old_s=s[j];p=l+b+old_s;e=actual-p
        old_l=l;l=l+b+a*e;b=g*(l-old_l)+(1-g)*b
        s[j]=d*(actual-l)+(1-d)*old_s
        rows.append([t,actual,p,e,l,b,s[j]])
    forecasts=[l+h*b+s[(len(y)+h-1)%4] for h in range(1,9)]
    return np.array(rows),forecasts
def optimize(y,state):
    def loss(p):return np.sum(run(y,p,state)[0][:,3]**2)
    starts=[[a,g,d] for a in [.1,.5,.9] for g in [0,.1,.8] for d in [0,.1,.8]]
    fits=[minimize(loss,p,bounds=[(0,1)]*3,method='L-BFGS-B',options={'ftol':1e-14,'gtol':1e-7,'maxiter':2000}) for p in starts]
    return min(fits,key=lambda r:r.fun).x
configs=[('Chapter 8: all-16 regression',read('bikes_ch8.csv'),16,16),
         ('Earlier series: first-year regression and seasons',read('bikes.csv'),4,4),
         ('Earlier series: first-year states, all-16 regression seasons',read('bikes.csv'),4,16)]
summaries=[];allrows=[];allforecasts=[];stored=[]
for name,y,n,sn in configs:
    state=init(y,n,sn)
    if n==4 and sn==16:
        state=(state[0],state[1],init(y,16,16)[2])
    best=optimize(y,state)
    print('INIT',name,state)
    variants=[('Initial',[.2,.1,.1]),('Optimized',best)]
    if n==4:variants.append(('Reported rounded constants',[.561,0,0]))
    for label,params in variants:
        rows,f=run(y,params,state);e=rows[:,3]
        if name==configs[0][0] and label=='Initial':
            with (root/'bikes_hw_updates_turn14.csv').open() as fprev:
                prev=list(csv.DictReader(fprev))
            assert np.allclose(rows[:,2],[float(r['prediction']) for r in prev])
        metrics=[np.mean(abs(e)),np.mean(e**2),100*np.mean(abs(e)/y)]
        summaries.append([name,label,*params,*metrics,*f[:4]])
        allrows.extend([[name,label,*r] for r in rows])
        allforecasts.extend([[name,label,16+h,h,v] for h,v in enumerate(f,1)])
        print('RESULT',summaries[-1])
        stored.append((name,label,rows,f))
for filename,header,rows in [
 ('bikes_fit_summary_turn15.csv',['initialization','fit','alpha','gamma','delta','MAD','MSD','MAPE_percent','forecast17','forecast18','forecast19','forecast20'],summaries),
 ('bikes_fitted_values_turn15.csv',['initialization','fit','quarter','actual','fitted','error','level','trend','seasonal'],allrows),
 ('bikes_forecasts_turn15.csv',['initialization','fit','target_quarter','horizon','forecast'],allforecasts)]:
    with (root/filename).open('w',newline='') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)
print('CHAPTER 8 FITTED VALUES')
ii=stored[0][2];oo=stored[1][2]
for i,o in zip(ii,oo):print(f'| {int(i[0])} | {i[1]:.0f} | {i[2]:.3f} | {o[2]:.3f} |')
fig,axes=plt.subplots(2,1,figsize=(11,9),sharex=True)
for ax,(name,y,n,sn) in zip(axes,[configs[0],configs[2]]):
    ax.plot(range(1,17),y,'o-',color='#176b91',markersize=4,label='Observed sales')
    variants=[('Initial','#b85416'),('Optimized','#7b4397')] if n==16 else [('Initial','#b85416'),('Reported rounded constants','#7b4397'),('Optimized','#558344')]
    for label,color in variants:
        _,_,rs,f=next(r for r in stored if r[0]==name and r[1]==label)
        ax.plot(range(1,17),rs[:,2],color=color,lw=1.8,label=label+' fitted values')
        ax.plot(range(17,25),f,'s--',color=color,markersize=3,label=label+' forecasts')
    ax.axvline(16.5,color='gray',ls=':',lw=1)
    ax.set(title=name,ylabel='Bikes sold (units)')
    ax.grid(axis='y',alpha=.2);ax.legend(loc='upper left',fontsize=8,ncol=2);ax.margins(y=.2)
axes[1].set(xlabel='Quarter index',xticks=[1,4,8,12,16,20,24])
fig.tight_layout(pad=1.8)
fig.savefig(root/'bikes_fits_forecasts_turn15.png',dpi=170)
plt.close(fig)
