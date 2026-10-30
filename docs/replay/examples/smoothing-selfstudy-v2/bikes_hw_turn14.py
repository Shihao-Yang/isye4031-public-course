"""All-16 OLS initialization and additive Holt-Winters updates.
Deterministic formula calculation; no random seed used.
Seasonality uses y_t-L_t, with delta=.10; trend uses gamma=.10.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
with (root/'bikes_ch8.csv').open() as f:
    y=np.array([float(r['y']) for r in csv.DictReader(f)])
t=np.arange(1,17)
sxx=((t-t.mean())**2).sum()
sxy=((t-t.mean())*(y-y.mean())).sum()
b0=sxy/sxx
l0=y.mean()-b0*t.mean()
baseline=l0+b0*t
detrended=y-baseline
season=detrended.reshape(4,4).mean(axis=0)
assert np.isclose(season.sum(),0)
a,g,d=.2,.1,.1
level,trend=l0,b0
factors=list(season)
rows=[]
for week,actual in zip(t,y):
    q=(week-1)%4
    old_l,old_b,old_s=level,trend,factors[q]
    pred=old_l+old_b+old_s
    error=actual-pred
    level=a*(actual-old_s)+(1-a)*(old_l+old_b)
    trend=g*(level-old_l)+(1-g)*old_b
    factors[q]=d*(actual-level)+(1-d)*old_s
    rows.append([week,q+1,actual,old_l,old_b,old_s,pred,error,level,trend,factors[q]])
print(f'sum_t={t.sum()}, sum_y={y.sum()}, mean_t={t.mean()}, mean_y={y.mean()}, Sxx={sxx}, Sxy={sxy}')
print('L0=',l0,'B0=',b0,'seasonal=',season,'sum=',season.sum())
for q in range(4): print('Q',q+1,'detrended=',detrended[q::4].tolist(),'mean=',season[q])
for row in rows[:5]: print('UPDATE', [round(float(v),9) for v in row])
print('Final level/trend/seasons:',level,trend,factors)
forecasts=[]
for h in range(1,5):
    target=16+h
    forecast=level+h*trend+factors[(target-1)%4]
    forecasts.append([target,h,forecast]); print('FORECAST',forecasts[-1])
for filename,header,rs in [
    ('bikes_hw_initialization_turn14.csv',['quarter_index','season','actual','regression_baseline','detrended','initial_seasonal_factor'],
     [[int(i),int((i-1)%4+1),v,base,res,season[(i-1)%4]] for i,v,base,res in zip(t,y,baseline,detrended)]),
    ('bikes_hw_updates_turn14.csv',['quarter_index','season','actual','prior_level','prior_trend','prior_seasonal','prediction','error','updated_level','updated_trend','updated_seasonal'],rows),
    ('bikes_hw_forecasts_turn14.csv',['target_quarter_index','horizon','forecast'],forecasts)]:
    with (root/filename).open('w',newline='') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rs)
fig,axes=plt.subplots(1,2,figsize=(12,5.2))
axes[0].plot(t,y,'o-',color='#176b91',label='Chapter 8 observed sales')
axes[0].plot(t,baseline,'--',color='#7b4397',label='All-16 regression baseline')
axes[0].set(title='Initialize level and trend from the regression',xlabel='Quarter index',ylabel='Bikes sold (units)',xticks=[1,4,8,12,16])
axes[1].plot(t,y,'o-',color='#176b91',label='Observed sales')
axes[1].plot(t,np.array(rows)[:,6],'s--',color='#b85416',markersize=4,label='Holt–Winters fitted values')
axes[1].plot([r[0] for r in forecasts],[r[2] for r in forecasts],'s--',color='#7b4397',label='Forecasts made at quarter 16')
axes[1].axvline(16.5,color='gray',ls=':',lw=1)
axes[1].set(title='Additive updates: α=.20, γ=.10, δ=.10',xlabel='Quarter index',ylabel='Bikes sold (units)',xticks=[1,4,8,12,16,20])
for ax in axes:
    ax.grid(axis='y',alpha=.2);ax.legend(loc='upper left',fontsize=8);ax.margins(y=.2)
fig.tight_layout(pad=1.8,w_pad=2)
fig.savefig(root/'bikes_hw_turn14.png',dpi=170)
plt.close(fig)
