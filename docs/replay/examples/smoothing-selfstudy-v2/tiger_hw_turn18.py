"""Multiplicative HW with first-16 OLS and normalized ratio seasons.
Sales units: thousands of cases. Deterministic; no seed used.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
with (root/'tiger.csv').open() as f:y=np.array([float(r['y']) for r in csv.DictReader(f)])
t=np.arange(1,17);yy=y[:16]
sxx=((t-t.mean())**2).sum();sxy=((t-t.mean())*(yy-yy.mean())).sum()
b0=sxy/sxx;l0=yy.mean()-b0*t.mean()
baseline=l0+b0*t;ratios=yy/baseline
raw=ratios.reshape(4,4).mean(axis=0);normalizer=raw.mean();initial=raw/normalizer
a,g,d=.2,.1,.1
level,trend=l0,b0;s=list(initial);rows=[]
for i,actual in enumerate(y):
    q=i%4;ol,ob,os=level,trend,s[q]
    prediction=(ol+ob)*os;error=actual-prediction
    level=a*(actual/os)+(1-a)*(ol+ob)
    trend=g*(level-ol)+(1-g)*ob
    s[q]=d*(actual/level)+(1-d)*os
    rows.append([i+1,q+1,actual,ol,ob,os,prediction,error,actual/os,level,trend,actual/level,s[q]])
print('sum_y',yy.sum(),'mean_y',yy.mean(),'mean_t',t.mean(),'Sxx',sxx,'Sxy',sxy)
print('L0',l0,'B0',b0,'raw seasonal',raw,'normalizer',normalizer,'normalized',initial,'mean',initial.mean())
for q in range(4):print('Q',q+1,'ratios',ratios[q::4])
for row in rows[:5]:print('UPDATE',[round(float(v),9) for v in row])
for filename,header,rs in [
 ('tiger_initialization_turn18.csv',['quarter','season','sales','baseline','ratio','raw_seasonal','normalized_seasonal'],
  [[i+1,i%4+1,yy[i],baseline[i],ratios[i],raw[i%4],initial[i%4]] for i in range(16)]),
 ('tiger_updates_turn18.csv',['quarter','season','sales','prior_level','prior_trend','prior_seasonal','prediction','error','deseasonalized_sales','level','trend','sales_divided_by_level','seasonal'],rows)]:
    with (root/filename).open('w',newline='') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rs)
fig,axes=plt.subplots(1,2,figsize=(12,5.3))
axes[0].plot(range(1,33),y,'o-',color='#176b91',markersize=3.5,label='Observed sales')
axes[0].plot(t,baseline,'--',color='#7b4397',label='First-16 regression baseline')
axes[0].plot(range(1,33),np.array(rows)[:,6],color='#b85416',lw=1.7,label='Multiplicative HW predictions')
axes[0].axvspan(.5,16.5,color='gray',alpha=.09,label='Initialization window')
axes[0].set(title='Tiger Sports Drink: level, trend, and seasonal swings',xlabel='Quarter index',ylabel='Sales (thousands of cases)',xticks=[1,4,8,12,16,20,24,28,32])
axes[0].legend(loc='upper left',fontsize=8)
axes[1].bar(range(1,5),initial,color=['#176b91','#b85416','#7b4397','#558344'])
axes[1].axhline(1,color='black',ls='--',lw=1,label='Neutral multiplier = 1')
for q,sval in enumerate(initial,1):axes[1].text(q,sval+.025,f'{sval:.4f}',ha='center',fontsize=9)
axes[1].set(title='Initial normalized seasonal multipliers',xlabel='Quarter of year',ylabel='Seasonal multiplier (dimensionless)',xticks=[1,2,3,4],xticklabels=['Q1','Q2','Q3','Q4'],ylim=(0,1.6))
axes[1].legend(fontsize=8)
for ax in axes:ax.grid(axis='y',alpha=.2)
fig.tight_layout(pad=1.8,w_pad=2)
fig.savefig(root/'tiger_hw_turn18.png',dpi=170)
plt.close(fig)
