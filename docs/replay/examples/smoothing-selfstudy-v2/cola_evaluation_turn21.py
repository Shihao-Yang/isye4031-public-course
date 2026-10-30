"""Chronological last-year evaluation; constants fixed at .20.
Initialization uses months 1–24 only, matching turn 20. No randomness.
Both fixed-origin 12-step and rolling one-step predictions are scored.
This is retrospective illustration on previously inspected data.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
with (root/'cola.csv').open() as f:y=np.array([float(r['y']) for r in csv.DictReader(f)])
train=y[:24];test=y[24:];t=np.arange(1,25);a=g=d=.2
b0=np.sum((t-t.mean())*(train-train.mean()))/np.sum((t-t.mean())**2)
l0=train.mean()-b0*t.mean();baseline=l0+b0*t
add=(train-baseline).reshape(2,12).mean(axis=0)
raw=(train/baseline).reshape(2,12).mean(axis=0);mul=raw/raw.mean()
def update(actual,l,b,s,name):
    p=l+b+s if name=='Additive HW' else (l+b)*s
    new_l=a*(actual-s)+(1-a)*(l+b) if name=='Additive HW' else a*actual/s+(1-a)*(l+b)
    new_b=g*(new_l-l)+(1-g)*b
    new_s=d*(actual-new_l)+(1-d)*s if name=='Additive HW' else d*actual/new_l+(1-d)*s
    return p,new_l,new_b,new_s
predictions={}
for name,initial in [('Additive HW',add),('Multiplicative HW',mul)]:
    l,b=l0,b0;s=list(initial)
    for i,actual in enumerate(train):
        p,l,b,s[i%12]=update(actual,l,b,s[i%12],name)
    print('Origin24',name,'L24',l,'B24',b)
    fixed=[]
    for h in range(1,13):
        q=(24+h-1)%12
        fixed.append(l+h*b+s[q] if name=='Additive HW' else (l+h*b)*s[q])
    rolling=[]
    for i,actual in enumerate(test,24):
        p,l,b,s[i%12]=update(actual,l,b,s[i%12],name)
        rolling.append(p)
    predictions[('Fixed origin: month 24',name)]=np.array(fixed)
    predictions[('Rolling one step',name)]=np.array(rolling)
for evaluation in ['Fixed origin: month 24','Rolling one step']:
    predictions[(evaluation,'Seasonal naive')]=y[12:24].copy()
metrics=[];rows=[]
for (evaluation,name),p in predictions.items():
    e=test-p
    r=[evaluation,name,np.mean(abs(e)),np.mean(e**2),100*np.mean(abs(e)/test),np.mean(e)]
    metrics.append(r);print('METRICS',r)
    for i,(actual,pred,error) in enumerate(zip(test,p,e),25):
        rows.append([evaluation,name,i,actual,pred,error,abs(error),error**2,100*abs(error)/actual])
    print('ERRORS',evaluation,name,np.round(e,2))
# Verify the rolling predictions are exactly the causal last-year predictions
# from turn 20, which used the identical first-24 initialization and constants.
with (root/'cola_fitted_values_turn20.csv').open() as f:
    previous=list(csv.DictReader(f))
for short,name in [('Additive','Additive HW'),('Multiplicative','Multiplicative HW')]:
    pp=[float(r['fitted']) for r in previous if r['model']==short and int(float(r['month']))>=25]
    assert np.allclose(pp,predictions[('Rolling one step',name)])
for fn,header,data in [
 ('cola_evaluation_metrics_turn21.csv',['evaluation','method','MAD','MSD','MAPE_percent','mean_error_actual_minus_forecast'],metrics),
 ('cola_evaluation_errors_turn21.csv',['evaluation','method','month','actual','forecast','error','absolute_error','squared_error','absolute_percentage_error'],rows)]:
    with (root/fn).open('w',newline='') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(data)
fig,axes=plt.subplots(2,2,figsize=(13,8.6),sharex=True,gridspec_kw={'height_ratios':[1.4,1]})
colors={'Additive HW':'#b85416','Multiplicative HW':'#7b4397','Seasonal naive':'#558344'}
xx=np.arange(25,37)
for col,evaluation in enumerate(['Fixed origin: month 24','Rolling one step']):
    axes[0,col].plot(xx,test,'o-',color='#176b91',lw=2,label='Actual sales')
    for name,color in colors.items():
        p=predictions[(evaluation,name)]
        axes[0,col].plot(xx,p,'s--',markersize=3.5,color=color,label=name)
        axes[1,col].plot(xx,test-p,'o-',markersize=3.5,color=color,label=name)
    axes[0,col].set(title=evaluation,ylabel='Sales (hundreds of cases)')
    axes[0,col].legend(fontsize=8,loc='upper left')
    axes[1,col].axhline(0,color='black',lw=1)
    axes[1,col].set(ylabel='Error: actual − forecast\n(hundreds of cases)',xlabel='Third-year month')
    axes[1,col].set_xticks(xx,['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],rotation=45,ha='right')
for ax in axes.flat:ax.grid(axis='y',alpha=.2)
fig.suptitle('Tasty Cola: train on months 1–24, evaluate months 25–36',fontsize=13)
fig.tight_layout(pad=1.8)
fig.savefig(root/'cola_evaluation_turn21.png',dpi=170)
plt.close(fig)
