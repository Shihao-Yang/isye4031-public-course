import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('paper_towels.csv', newline='') as f:
    rows=list(csv.reader(f))
a=np.array([[float(v) for v in row] for row in rows[1:]])
t=a[:,0].astype(int)
y=a[:,1]
d=np.diff(y)
results={
    'observations':len(y), 'differences':len(d),
    'week_1_actual_rolls':100000+10000*y[0],
    'week_60_actual_rolls':100000+10000*y[59],
    'week_119_y':y[-2], 'week_120_y':y[-1],
    'last_weekly_change':d[-1], 'last_weekly_change_rolls':10000*d[-1],
    'mean_weekly_change':float(d.mean()),
    'level_lag1_sample_correlation':float(np.corrcoef(y[:-1],y[1:])[0,1]),
    'change_lag1_sample_correlation':float(np.corrcoef(d[:-1],d[1:])[0,1]),
}
with open('paper_towels_changes_03_results.json','w') as f:
    json.dump(results,f,indent=2)
fig,axes=plt.subplots(2,1,figsize=(10,6.5),sharex=True,constrained_layout=True)
axes[0].plot(t,y,color='#21618c',lw=1.6)
axes[0].axhline(0,color='gray',lw=.8,ls=':')
axes[0].set(title='Paper-towel sales: 120 observed weeks',ylabel='Sales above 100,000 rolls\n(units of 10,000 rolls)')
axes[1].plot(t[1:],d,color='#b45f06',lw=1.25)
axes[1].axhline(0,color='gray',lw=.8)
axes[1].set(title='First differences: change from the previous week',xlabel='Week index (calendar dates unspecified)',ylabel='Weekly change\n(units of 10,000 rolls)')
for ax in axes:
    ax.grid(axis='y',alpha=.2)
    ax.spines[['top','right']].set_visible(False)
axes[1].set_xlim(1,120)
fig.savefig('paper_towels_changes_03.png',dpi=170)
print(json.dumps(results,indent=2))
