"""Eight-quarter conditional additive-normal HW prediction intervals.
Variance estimated by SSE/16 for both fits; no parameter uncertainty.
Deterministic; no random seed used.
"""
from pathlib import Path
import csv
from statistics import NormalDist
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
with (root/'bikes_ch8.csv').open() as f:y=np.array([float(r['y']) for r in csv.DictReader(f)])
with (root/'bikes_fit_summary_turn15.csv').open() as f:
    fits=[r for r in csv.DictReader(f) if r['initialization']=='Chapter 8: all-16 regression']
t=np.arange(1,17);b0=np.sum((t-t.mean())*(y-y.mean()))/np.sum((t-t.mean())**2);l0=y.mean()-b0*t.mean()
s0=(y-l0-b0*t).reshape(4,4).mean(axis=0)
z=NormalDist().inv_cdf(.975)
out=[]
fig,axes=plt.subplots(1,2,figsize=(12.5,5.6),sharey=True)
for ax,fit in zip(axes,fits):
    name=fit['fit'];a,g,d=[float(fit[k]) for k in ['alpha','gamma','delta']]
    beta=a*g;c=d*(1-a);l,b=l0,b0;s=list(s0);pred=[];errors=[]
    for i,actual in enumerate(y):
        q=i%4;p=l+b+s[q];e=actual-p
        l=l+b+a*e;b=b+beta*e;s[q]=s[q]+c*e
        pred.append(p);errors.append(e)
    var=np.mean(np.array(errors)**2)
    assert np.isclose(var,float(fit['MSD']))
    print(name,'a,g,d=',a,g,d,'beta,c=',beta,c,'sigma=',np.sqrt(var),'L16 B16',l,b,'seasons=',s)
    current=[]
    for h in range(1,9):
        point=l+h*b+s[(16+h-1)%4]
        factor=1+sum((a+j*beta+(c if j%4==0 else 0))**2 for j in range(1,h))
        se=np.sqrt(var*factor);lo,hi=point-z*se,point+z*se
        row=[name,16+h,h,(h-1)%4+1,point,se,lo,hi,factor,var]
        out.append(row);current.append(row)
        print(f'| {h} | {point:.3f} | [{lo:.3f}, {hi:.3f}] | SE={se:.6f}, variance factor={factor:.9f}')
    rs=np.array([r[1:] for r in current],dtype=float)
    ax.plot(range(9,17),y[8:],'o-',color='#176b91',markersize=4,label='Observed Chapter 8 sales')
    ax.fill_between(rs[:,0],rs[:,5],rs[:,6],color='#b85416',alpha=.2,label='Approximate 95% prediction interval')
    ax.plot(rs[:,0],rs[:,3],'s--',color='#b85416',lw=1.8,markersize=4,label='Forecast made at quarter 16')
    ax.axvline(16.5,color='gray',ls=':',lw=1)
    ax.set(title=f'{name}: α={a:.4f}, γ={g:.2f}, δ={d:.2f}',xlabel='Quarter index',xlim=(8.5,24.5))
    ax.set_xticks([9,12,16,17,20,21,24]);ax.grid(axis='y',alpha=.2)
    ax.legend(loc='upper left',fontsize=8)
axes[0].set_ylabel('Bikes sold (units)')
fig.suptitle('Chapter 8 bikes: forecasts for the next two years',fontsize=13)
fig.tight_layout(pad=1.8)
fig.savefig(root/'bikes_intervals_turn16.png',dpi=170)
plt.close(fig)
with (root/'bikes_intervals_turn16.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['fit','target_quarter','horizon','season','point_forecast','prediction_SE','lower95','upper95','variance_factor','innovation_variance']);w.writerows(out)
