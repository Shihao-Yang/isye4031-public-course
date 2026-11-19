import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

base = Path(__file__).resolve().parent
a = np.genfromtxt(base / 'paper_towels.csv', delimiter=',', names=True)
y = a['sales_above_100000_in_10000_rolls']
n = len(y)
c = y-y.mean()
r = np.array([1.] + [float(c[k:] @ c[:-k] / (c @ c)) for k in range(1,15)])
lags = np.arange(1,15)
se = np.array([np.sqrt((1+2*np.sum(r[1:k]**2))/n) for k in lags])
limits = 2*se
rows = [{'lag':int(k),'acf':float(r[k]),'bartlett_se':float(s),'lower_limit':float(-b),'upper_limit':float(b),'outside_limits':bool(abs(r[k])>b)} for k,s,b in zip(lags,se,limits)]
result = {'series':'Original sales','n':n,'formula':'SE(r_k) = sqrt((1 + 2 * sum(r_j^2 for j=1,...,k-1)) / n)', 'multiplier':2,'rows':rows,'interpretation':'Approximate pointwise zero-correlation reference limits; stationarity assumption is questionable for original sales.'}
(base / 'turn07_bartlett_acf_results.json').write_text(json.dumps(result,indent=2)+'\n')
plt.rcParams.update({'font.size':17,'axes.titlesize':22,'axes.labelsize':18})
fig,ax = plt.subplots(figsize=(12,6.5))
ax.fill_between(lags,-limits,limits,color='#176a9a',alpha=.10)
ax.plot(lags,limits,'--',color='#b45c20',lw=2,label='±2 Bartlett standard errors')
ax.plot(lags,-limits,'--',color='#b45c20',lw=2)
marker,stems,baseline=ax.stem(lags,r[1:],basefmt=' ')
plt.setp(marker,color='#176a9a',markersize=7)
plt.setp(stems,color='#176a9a',linewidth=2.2)
ax.axhline(0,color='#333333',lw=1)
ax.set(title='Original sales: sample ACF at lags 1–14',xlabel='Lag (weeks)',ylabel='Sample autocorrelation',xlim=(.6,14.4),ylim=(-.9,1.08))
ax.set_xticks(lags)
ax.legend(loc='lower left',frameon=False,fontsize=15)
ax.grid(axis='y',alpha=.2)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout(pad=2)
fig.savefig(base / 'turn07_bartlett_acf.png',dpi=160)
plt.close(fig)
print(json.dumps(result,indent=2))
