"""Conditional additive-normal Holt intervals and supplied week-53 update.
Deterministic calculations; no randomness or seed.
"""
from pathlib import Path
import csv
from statistics import NormalDist
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
with (root/'thermostat_fit_summary_turn09.csv').open() as f:
    fit=next(r for r in csv.DictReader(f) if r['method']=='Fitted')
a,g=float(fit['alpha']),float(fit['gamma'])
l,b=float(fit['L52']),float(fit['B52'])
sse=float(fit['SSE'])
s2=sse/(52-2)
s=np.sqrt(s2)
z=NormalDist().inv_cdf(.975)
actual=330.
error=actual-(l+b)
new_l=l+b+a*error
new_b=b+a*g*error
print(f'alpha={a:.12f}; gamma={g:.12f}; alpha*gamma={a*g:.12f}; SSE={sse:.9f}; variance={s2:.9f}; s={s:.9f}')
print(f'error={error:.12f}; level correction={a*error:.12f}; trend correction={a*g*error:.12f}; L53={new_l:.12f}; B53={new_b:.12f}')
rows=[]
for origin,level,trend in [(52,l,b),(53,new_l,new_b)]:
    for h in range(1,4):
        factor=1+sum((a+j*a*g)**2 for j in range(1,h))
        se=s*np.sqrt(factor)
        point=level+h*trend
        lo,hi=point-z*se,point+z*se
        rows.append([origin,origin+h,h,point,se,lo,hi,factor])
        print(f'origin {origin}, week {origin+h}, h={h}: point={point:.9f}, SE={se:.9f}, PI=[{lo:.9f},{hi:.9f}], variance factor={factor:.9f}')
for target in [54,55,56]:
    h=target-53
    old=l+(target-52)*b
    new=new_l+h*new_b
    print(f'target {target}: old={old:.9f}, new={new:.9f}, difference={new-old:.9f}, expected={(a+h*a*g)*error:.9f}')
with (root/'thermostat_intervals_turn10.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['origin_week','target_week','horizon','forecast','prediction_SE','lower95','upper95','variance_factor']);w.writerows(rows)
with (root/'thermostat_update_turn10.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['alpha','gamma','L52','B52','week53_sales','error','level_correction','trend_correction','L53','B53','fixed_error_variance'])
    w.writerow([a,g,l,b,actual,error,a*error,a*g*error,new_l,new_b,s2])
with (root/'thermostat.csv').open() as f:
    y=[float(r['y']) for r in csv.DictReader(f)]
fig,axes=plt.subplots(1,2,figsize=(12,5.4),sharey=True)
for ax,origin in zip(axes,[52,53]):
    rs=np.array([r for r in rows if r[0]==origin])
    ax.plot(range(47,53),y[46:],'o-',color='#176b91',label='Observed sales')
    ax.fill_between(rs[:,1],rs[:,5],rs[:,6],color='#b85416',alpha=.2,label='Approximate 95% prediction interval')
    ax.plot(rs[:,1],rs[:,3],'s--',color='#b85416',lw=2,label='Point forecast')
    ax.axvline(origin+.5,color='gray',lw=1)
    ax.scatter([53],[actual],marker='*',s=110,color='#7b4397',zorder=5,
               label='Week 53: 330'+(' (revealed afterward)' if origin==52 else ' (now observed)'))
    if origin==53:
        ax.plot([52,53],[y[-1],actual],color='#176b91',lw=1)
        ax.plot([54,55,56],[l+(week-52)*b for week in [54,55,56]],':',color='gray',label='Previous forecasts')
    ax.set(title=f'Information through week {origin}',xlabel='Week',xlim=(46.5,56.5),ylim=(245,410))
    ax.set_xticks(range(47,57))
    ax.grid(axis='y',alpha=.2)
    ax.legend(loc='upper left',fontsize=7.6)
axes[0].set_ylabel('Thermostats sold (units)')
fig.suptitle('Holt forecasts before and after observing week 53',fontsize=13)
fig.tight_layout(pad=1.8)
fig.savefig(root/'thermostat_intervals_turn10.png',dpi=170)
plt.close(fig)
