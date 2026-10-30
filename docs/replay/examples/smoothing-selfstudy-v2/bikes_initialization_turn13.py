"""Compare bike series; first-two-cycle additive initialization.
Explicit annual-average trend convention, not undocumented software output.
Deterministic; no random seed used.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
def read(name):
    with (root/name).open() as f:
        return np.array([float(r['y']) for r in csv.DictReader(f)])
a=read('bikes.csv');y=read('bikes_ch8.csv');m=4
years=y[:8].reshape(2,m)
means=years.mean(axis=1)
b=(means[1]-means[0])/m
l=means[0]-b*2.5
ts=np.arange(1,9)
residual=y[:8]-(l+b*ts)
season= residual.reshape(2,m).mean(axis=0)
assert np.isclose(season.sum(),0)
print('Year means:',means,'initial trend:',b,'L0:',l,'L8:',l+8*b)
print('Seasonal factors:',season,'sum:',season.sum())
for i in range(4):
    print('year',i+1,'Table6.8',a[i*4:(i+1)*4], 'Chapter8',y[i*4:(i+1)*4], 'difference',(y-a)[i*4:(i+1)*4])
for t,actual,base,res in zip(ts,y[:8],l+b*ts,residual):
    print(f'week index {t}: actual={actual}, trend baseline={base}, detrended={res}')
for h in range(1,5):
    target=8+h
    print('Forecast from initialized state at8:',target,l+target*b+season[(target-1)%4])
with (root/'bikes_difference_turn13.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['quarter_index','year','quarter_of_year','Table6_8','Chapter8','Chapter8_minus_Table6_8'])
    w.writerows([[i+1,i//4+1,i%4+1,a[i],y[i],y[i]-a[i]] for i in range(16)])
with (root/'bikes_init_turn13.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['season','year1_observation','year2_observation','year1_detrended','year2_detrended','initial_seasonal_factor','L0','B0','L8'])
    w.writerows([[j+1,years[0,j],years[1,j],residual[j],residual[j+4],season[j],l,b,l+8*b] for j in range(4)])
fig,axes=plt.subplots(1,2,figsize=(12,5.2))
axes[0].plot(range(1,17),a,'o-',lw=1.7,color='#176b91',label='Table 6.8: bikes.csv')
axes[0].plot(range(1,17),y,'s--',lw=1.7,color='#b85416',label='Chapter 8: bikes_ch8.csv')
axes[0].axvspan(.5,8.5,color='gray',alpha=.1,label='First two years used for initialization')
axes[0].set(title='Same first eight quarters; different later sales',xlabel='Quarter index',ylabel='Bikes sold (units)',xticks=[1,4,8,12,16])
axes[0].legend(loc='upper left',fontsize=8)
axes[1].bar(range(1,5),season,color=['#176b91','#b85416','#7b4397','#558344'])
axes[1].axhline(0,color='gray',lw=1)
for j,s in enumerate(season,1):
    axes[1].annotate(f'{s:+.4f}',(j,s),xytext=(0,5 if s>0 else -14),textcoords='offset points',ha='center',fontsize=9)
axes[1].set(title='Initial additive seasonal effects: Chapter 8',xlabel='Quarter of year',ylabel='Seasonal adjustment (bikes)',xticks=[1,2,3,4],xticklabels=['Q1','Q2','Q3','Q4'],ylim=(-20,24))
for ax in axes:ax.grid(axis='y',alpha=.2)
fig.tight_layout(pad=1.8,w_pad=2)
fig.savefig(root/'bikes_initialization_turn13.png',dpi=170)
plt.close(fig)
