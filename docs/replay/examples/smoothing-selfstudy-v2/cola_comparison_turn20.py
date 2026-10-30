"""Additive/multiplicative seasonal HW, fixed .20 constants.
First-24 OLS initialization. Independent additive normal innovations for
both models; variance SSE/(36-3). Multiplicative intervals simulated with
seed 20261030 and 100000 paths. No parameter/initialization uncertainty.
"""
from pathlib import Path
import csv
import numpy as np
from statistics import NormalDist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
with (root/'cola.csv').open() as f:y=np.array([float(r['y']) for r in csv.DictReader(f)])
t=np.arange(1,25);v=y[:24];m=12;a=g=d=.2
b0=np.sum((t-t.mean())*(v-v.mean()))/np.sum((t-t.mean())**2);l0=v.mean()-b0*t.mean()
baseline=l0+b0*t
add_s=(v-baseline).reshape(2,12).mean(axis=0)
raw=(v/baseline).reshape(2,12).mean(axis=0);mul_s=raw/raw.mean()
z=NormalDist().inv_cdf(.975);seed=20261030;npaths=100000
rng=np.random.default_rng(seed)
# Common standard-normal draws reduce noise when comparing simulations.
draws=rng.standard_normal((12,npaths))
summary=[];updates=[];intervals=[];stored={}
print('Initialization L0=',l0,'B0=',b0,'additive seasons=',add_s,'multiplicative seasons=',mul_s)
for name,init in [('Additive',add_s),('Multiplicative',mul_s)]:
    l,b=l0,b0;s=list(init);rs=[]
    for i,actual in enumerate(y):
        q=i%m
        p=l+b+s[q] if name=='Additive' else (l+b)*s[q]
        error=actual-p
        new_l=a*(actual-s[q])+(1-a)*(l+b) if name=='Additive' else a*actual/s[q]+(1-a)*(l+b)
        new_b=g*(new_l-l)+(1-g)*b
        s[q]=d*(actual-new_l)+(1-d)*s[q] if name=='Additive' else d*actual/new_l+(1-d)*s[q]
        l,b=new_l,new_b;rs.append([i+1,actual,p,error,l,b,s[q]])
    rs=np.array(rs);e=rs[:,3];sse=np.sum(e**2);var=sse/33
    summary.append([name,a,g,d,sse,np.mean(abs(e)),np.mean(e**2),100*np.mean(abs(e)/y),var,np.sqrt(var),l,b])
    updates.extend([[name,*r] for r in rs])
    point=np.array([l+h*b+s[(36+h-1)%12] if name=='Additive' else (l+h*b)*s[(36+h-1)%12] for h in range(1,13)])
    if name=='Additive':
        beta=a*g;c=d*(1-a)
        se=np.array([np.sqrt(var*(1+sum((a+j*beta+(c if j%12==0 else 0))**2 for j in range(1,h)))) for h in range(1,13)])
        lo,hi=point-z*se,point+z*se
    else:
        ll=np.full(npaths,l);bb=np.full(npaths,b);ss=np.tile(np.array(s),(npaths,1))
        paths=[]
        for h in range(1,13):
            q=(36+h-1)%12
            old=ss[:,q].copy()
            pp=(ll+bb)*old
            actual=pp+np.sqrt(var)*draws[h-1]
            new_l=a*actual/old+(1-a)*(ll+bb)
            new_b=g*(new_l-ll)+(1-g)*bb
            ss[:,q]=d*actual/new_l+(1-d)*old
            ll,bb=new_l,new_b
            paths.append(actual)
        paths=np.array(paths)
        lo,hi=np.quantile(paths,[.025,.975],axis=1)
        print('Simulated nonpositive sales:',int((paths<=0).sum()))
    for h in range(1,13):intervals.append([name,36+h,h,point[h-1],lo[h-1],hi[h-1]])
    stored[name]=(rs,point,lo,hi)
    print('SUMMARY',summary[-1])
for h in range(12):
    aa=[r for r in intervals if r[0]=='Additive'][h];mm=[r for r in intervals if r[0]=='Multiplicative'][h]
    print(f'| {h+1} | {aa[3]:.1f} | [{aa[4]:.1f}, {aa[5]:.1f}] | {mm[3]:.1f} | [{mm[4]:.1f}, {mm[5]:.1f}] |')
for filename,header,rows in [
 ('cola_fit_summary_turn20.csv',['model','alpha','gamma','delta','SSE','MAD','MSD','MAPE_percent','innovation_variance','innovation_sd','L36','B36'],summary),
 ('cola_fitted_values_turn20.csv',['model','month','actual','fitted','error','level','trend','seasonal'],updates),
 ('cola_forecasts_intervals_turn20.csv',['model','target_month','horizon','point_forecast','lower95','upper95'],intervals),
 ('cola_initialization_turn20.csv',['month_of_year','additive_factor','multiplicative_factor','L0','B0'],[[j+1,add_s[j],mul_s[j],l0,b0] for j in range(12)])]:
    with (root/filename).open('w',newline='') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)
fig,axes=plt.subplots(2,1,figsize=(11,9),sharex=True)
for ax,name,color in zip(axes,['Additive','Multiplicative'],['#b85416','#7b4397']):
    rs,point,lo,hi=stored[name]
    ax.plot(range(1,37),y,'o-',color='#176b91',markersize=3,label='Observed sales')
    ax.plot(range(1,37),rs[:,2],color=color,lw=1.8,label=name+' fitted values')
    ax.plot(range(37,49),point,'s--',color=color,markersize=3,label='Next-year point forecast')
    ax.fill_between(range(37,49),lo,hi,color=color,alpha=.2,label='Approximate 95% prediction interval')
    ax.axvspan(.5,24.5,color='gray',alpha=.07)
    ax.axvline(36.5,color='gray',ls=':',lw=1)
    ax.set(title=name+' seasonal Holt–Winters: α = γ = δ = 0.20',ylabel='Sales (hundreds of cases)')
    ax.grid(axis='y',alpha=.2);ax.legend(loc='upper left',fontsize=8,ncol=2)
axes[1].set(xlabel='Month index (1, 13, 25, 37 = January)',xticks=[1,12,24,36,37,42,45,48],xlim=(.5,48.5))
fig.tight_layout(pad=1.8)
fig.savefig(root/'cola_comparison_turn20.png',dpi=170)
plt.close(fig)
