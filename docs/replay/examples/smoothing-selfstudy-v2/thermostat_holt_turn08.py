"""OLS time-zero initialization, Holt updates. Deterministic; no seed used."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
with (root/'thermostat.csv').open() as f:
    y=np.array([float(r['y']) for r in csv.DictReader(f)])
t=np.arange(1,len(y)+1)
x,yy=t[:26],y[:26]
sxx=((x-x.mean())**2).sum()
sxy=((x-x.mean())*(yy-yy.mean())).sum()
b0=sxy/sxx
l0=yy.mean()-b0*x.mean()
alpha,gamma=.2,.1
level,trend=l0,b0
rows=[]
for week,actual in zip(t,y):
    old_level,old_trend=level,trend
    forecast=old_level+old_trend
    error=actual-forecast
    level=alpha*actual+(1-alpha)*forecast
    change=level-old_level
    trend=gamma*change+(1-gamma)*old_trend
    rows.append([week,actual,old_level,old_trend,forecast,error,level,change,trend,level+trend])
print(f'sum_t={x.sum()}, sum_y={yy.sum()}, mean_t={x.mean():.9f}, mean_y={yy.mean():.9f}, Sxx={sxx:.9f}, Sxy={sxy:.9f}')
print(f'L0={l0:.12f}; B0={b0:.12f}; week1 forecast={l0+b0:.12f}')
for row in rows[:3]+rows[25:28]+rows[-1:]:
    print('row', [round(float(v),9) for v in row])
with (root/'thermostat_holt_updates_turn08.csv').open('w',newline='') as f:
    w=csv.writer(f)
    w.writerow(['week','sales','prior_level','prior_trend','one_step_prediction','error','updated_level','level_change','updated_trend','next_week_forecast'])
    w.writerows(rows)
with (root/'thermostat_holt_init_turn08.csv').open('w',newline='') as f:
    w=csv.writer(f); w.writerow(['alpha','gamma','initial_level_at_zero','initial_trend','Sxx','Sxy'])
    w.writerow([alpha,gamma,l0,b0,sxx,sxy])
rs=np.array(rows)
fig,axes=plt.subplots(2,1,figsize=(11,7.6),sharex=True,gridspec_kw={'height_ratios':[2,1]})
axes[0].plot(t,y,'o-',color='#176b91',markersize=3.5,lw=1.2,label='Observed weekly sales')
axes[0].plot(t,rs[:,4],color='#b85416',lw=2,label='Holt prediction before each update')
axes[0].plot(x,l0+b0*x,':',color='#7b4397',lw=2,label='Regression used to initialize L₀ and B₀')
axes[0].plot(range(53,57),[level+h*trend for h in range(1,5)],'s--',color='#b85416',markersize=4,label='Forecasts made at week 52')
axes[0].set(ylabel='Thermostats sold (units)',title='Weekly thermostat sales and Holt’s changing level and trend',ylim=(135,385))
axes[0].legend(loc='upper left',fontsize=8)
axes[1].plot([0]+list(t),[b0]+list(rs[:,8]),color='#7b4397',lw=2,label='Updated trend Bₜ')
axes[1].axhline(0,color='gray',lw=1)
axes[1].set(xlabel='Week',ylabel='Estimated trend\n(units per week)',xlim=(0,57))
axes[1].legend(loc='upper left',fontsize=8)
for ax in axes:
    ax.axvspan(0,26.5,color='gray',alpha=.1)
    ax.axvline(26.5,color='gray',ls=':',lw=1)
    ax.grid(axis='y',alpha=.2)
axes[1].set_xticks([0,1,13,26,39,52,56])
fig.tight_layout(pad=1.8)
fig.savefig(root/'thermostat_holt_turn08.png',dpi=170)
plt.close(fig)
