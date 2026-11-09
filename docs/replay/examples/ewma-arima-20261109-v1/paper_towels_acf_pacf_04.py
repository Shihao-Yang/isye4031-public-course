import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf

with open('paper_towels.csv',newline='') as f:
    rows=list(csv.reader(f))
y=np.array([float(r[1]) for r in rows[1:]])
w=np.diff(y)
series=[y,w]
results={}
fig,axes=plt.subplots(2,2,figsize=(10,6.7),constrained_layout=True)
for i,(values,name) in enumerate(zip(series,['Sales levels','Weekly changes'])):
    aa=acf(values,nlags=20,fft=False)
    pp=pacf(values,nlags=20,method='ywm')
    band=1.96/np.sqrt(len(values))
    results[name]={'n':len(values),'acf':aa.tolist(),'pacf_yule_walker_mle':pp.tolist(),'approx_white_noise_band':float(band)}
    for j,(coeff,label) in enumerate([(aa,'ACF'),(pp,'PACF')]):
        ax=axes[i,j]
        ax.axhspan(-band,band,color='#21618c',alpha=.10)
        ax.axhline(0,color='gray',lw=.8)
        ax.vlines(range(1,21),0,coeff[1:],color='#21618c',lw=1.6)
        ax.plot(range(1,21),coeff[1:],'o',color='#21618c',ms=3)
        ax.set(title=f'{name}: {label}',xlabel='Lag (weeks)',ylabel='Correlation',ylim=(-1,1),xticks=[1,5,10,15,20])
        ax.spines[['top','right']].set_visible(False)
# Residualize both ends of lag 2 on the intervening observation.
a=w[2:]; b=w[:-2]; mid=np.column_stack([np.ones(len(w)-2),w[1:-1]])
ra=a-mid@np.linalg.lstsq(mid,a,rcond=None)[0]
rb=b-mid@np.linalg.lstsq(mid,b,rcond=None)[0]
results['changes_lag2_comparison']={
    'raw_same_rows_correlation':float(np.corrcoef(a,b)[0,1]),
    'partial_correlation_controlling_intervening_change':float(np.corrcoef(ra,rb)[0,1])}
fig.savefig('paper_towels_acf_pacf_04.png',dpi=170)
with open('paper_towels_acf_pacf_04_results.json','w') as f:
    json.dump(results,f,indent=2)
for name in ['Sales levels','Weekly changes']:
    print(name, 'ACF lags 1-6:',np.round(results[name]['acf'][1:7],4))
    print(name, 'PACF lags 1-6:',np.round(results[name]['pacf_yule_walker_mle'][1:7],4))
print(json.dumps(results['changes_lag2_comparison'],indent=2))
print('bands:', results['Sales levels']['approx_white_noise_band'],results['Weekly changes']['approx_white_noise_band'])
