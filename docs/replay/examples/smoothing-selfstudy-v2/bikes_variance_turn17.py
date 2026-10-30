"""Recompute turn-16 intervals with SSE/(16-3). No randomness or seed."""
from pathlib import Path
import csv
from statistics import NormalDist
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
with (root/'bikes_intervals_turn16.csv').open() as f:old=list(csv.DictReader(f))
with (root/'bikes_ch8.csv').open() as f:y=[float(r['y']) for r in csv.DictReader(f)]
z=NormalDist().inv_cdf(.975)
scale=np.sqrt(16/13)
out=[]
fig,axes=plt.subplots(1,2,figsize=(12.5,5.6),sharey=True)
for ax,name in zip(axes,['Initial','Optimized']):
    rs=[r for r in old if r['fit']==name]
    new=[]
    for r in rs:
        point=float(r['point_forecast']);se=float(r['prediction_SE'])*scale
        var=float(r['innovation_variance'])*16/13
        lo,hi=point-z*se,point+z*se
        row=[name,int(r['target_quarter']),int(r['horizon']),point,se,lo,hi,var]
        out.append(row);new.append(row)
        if int(r['horizon'])<=3:print(name,'h=',r['horizon'],'point=',point,'SE=',se,'PI=',lo,hi)
    print(name,'SSE=',float(rs[0]['innovation_variance'])*16,'new variance=',var,'new s=',np.sqrt(var))
    xx=np.array([int(r['target_quarter']) for r in rs])
    pp=np.array([float(r['point_forecast']) for r in rs])
    newlo=np.array([r[5] for r in new]);newhi=np.array([r[6] for r in new])
    ax.plot(range(9,17),y[8:],'o-',color='#176b91',markersize=3.5,label='Observed sales')
    ax.fill_between(xx,newlo,newhi,color='#b85416',alpha=.2,label='95% interval: SSE/13')
    ax.plot(xx,newlo,color='#b85416',lw=1)
    ax.plot(xx,newhi,color='#b85416',lw=1)
    ax.plot(xx,[float(r['lower95']) for r in rs],':',color='#7b4397',lw=1.8,label='Previous bounds: SSE/16')
    ax.plot(xx,[float(r['upper95']) for r in rs],':',color='#7b4397',lw=1.8)
    ax.plot(xx,pp,'s--',color='#b85416',lw=1.7,markersize=4,label='Point forecast (unchanged)')
    ax.axvline(16.5,color='gray',ls=':',lw=1)
    ax.set(title=name+' parameter setting',xlabel='Quarter index',xlim=(8.5,24.5),xticks=[9,12,16,17,20,21,24])
    ax.grid(axis='y',alpha=.2);ax.legend(loc='upper left',fontsize=8)
axes[0].set_ylabel('Bikes sold (units)')
fig.suptitle('Chapter 8 bikes: effect of using SSE/(16 − 3) for variance',fontsize=13)
fig.tight_layout(pad=1.8)
fig.savefig(root/'bikes_variance_convention_turn17.png',dpi=170)
plt.close(fig)
with (root/'bikes_intervals_turn17.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['fit','target_quarter','horizon','forecast','prediction_SE','lower95','upper95','innovation_variance_SSE_over13']);w.writerows(out)
print('width multiplier=',scale,'percentage increase=',100*(scale-1))
