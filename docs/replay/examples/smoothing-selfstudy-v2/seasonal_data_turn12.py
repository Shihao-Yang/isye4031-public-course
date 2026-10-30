"""Plot supplied seasonal series; descriptive calculations only.
No sampling or random seed. Bike series is bikes.csv, not bikes_ch8.csv.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
specs=[('bikes.csv','Mountain bikes',4,'Bikes sold (units)'),
       ('tiger.csv','Tiger Sports Drink',4,'Sales (thousands of cases)'),
       ('cola.csv','Tasty Cola',12,'Sales (hundreds of cases)')]
fig,axes=plt.subplots(3,2,figsize=(12,11.5),gridspec_kw={'width_ratios':[1.3,1]})
summary=[]
for i,(filename,title,m,unit) in enumerate(specs):
    with (root/filename).open() as f:
        y=np.array([float(r['y']) for r in csv.DictReader(f)])
    years=y.reshape(-1,m)
    print(title)
    if m==4:
        print('| Year | Q1 | Q2 | Q3 | Q4 |')
        for j,values in enumerate(years,1): print('| '+str(j)+' | '+' | '.join(f'{v:.0f}' for v in values)+' |')
    else:
        print('| Month | Year 1 | Year 2 | Year 3 |')
        months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        for j,label in enumerate(months): print('| '+label+' | '+' | '.join(f'{v:.0f}' for v in years[:,j])+' |')
    ax=axes[i,0]
    ax.plot(np.arange(1,len(y)+1),y,'o-',color='#176b91',lw=1.7,markersize=3.8)
    for boundary in range(m,len(y),m): ax.axvline(boundary+.5,color='gray',alpha=.22,lw=.8)
    ax.set(title=f'{title}: observations in time order',xlabel='Quarter index' if m==4 else 'Month index',ylabel=unit,xlim=(.5,len(y)+.5))
    ax=axes[i,1]
    for j,values in enumerate(years,1):
        ax.plot(range(1,m+1),values,'o-',lw=1.5,markersize=3.5,label=f'Year {j}',color=plt.cm.viridis((j-1)/max(1,len(years)-1)))
        summary.append([title,j,values.mean(),values.min(),values.max(),np.ptp(values),values.max()/values.min(),int(values.argmax())+1,int(values.argmin())+1])
    ax.set(title=f'{title}: compare seasons across years',xlabel='Quarter of year' if m==4 else 'Month of year',ylabel=unit)
    ax.set_xticks(range(1,m+1),[f'Q{j}' for j in range(1,5)] if m==4 else ['J','F','M','A','M','J','J','A','S','O','N','D'])
    ax.legend(fontsize=7,ncol=2,loc='upper left')
    print('First and last annual summaries:',summary[-len(years)],summary[-1])
for ax in axes.flat:
    ax.grid(axis='y',alpha=.2)
    ax.margins(y=.22)
fig.tight_layout(pad=2.0,h_pad=2.5,w_pad=2)
fig.savefig(root/'seasonal_patterns_turn12.png',dpi=170)
plt.close(fig)
with (root/'seasonal_summary_turn12.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['series','year_index','annual_mean','minimum','maximum','range','max_min_ratio','peak_season','trough_season']);w.writerows(summary)
