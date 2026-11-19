import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

base=Path(__file__).resolve().parent
raw=np.genfromtxt(base/'paper_towels.csv',delimiter=',',names=True)
x=np.diff(raw['sales_above_100000_in_10000_rolls'])
n=len(x)
c=x-x.mean()
lags=np.arange(1,15)
r=np.array([1.]+[float(c[k:]@c[:-k]/(c@c)) for k in lags])
pacf=[]
for k in lags:
    # Solve the Yule-Walker equations from the same full-sample ACF.
    matrix=r[np.abs(np.arange(k)[:,None]-np.arange(k)[None,:])]
    coefficients=np.linalg.solve(matrix,r[1:k+1])
    pacf.append(float(coefficients[-1]))
pacf=np.array(pacf)
acf_se=np.array([np.sqrt((1+2*np.sum(r[1:k]**2))/n) for k in lags])
pacf_se=1/np.sqrt(n)
result={'series':'First differences','n':n,'acf_convention':'One overall mean; full-sample sum of squared deviations in denominator','pacf_method':'Yule-Walker equations: last coefficient for each lag','pacf_reference_se':float(pacf_se),'pacf_two_se_limit':float(2*pacf_se),'pacf_limit_assumption':'Approximate pointwise reference for zero partial correlations beyond a finite cutoff; not a universal standard error','rows':[{'lag':int(k),'acf':float(r[k]),'acf_two_se_limit':float(2*s),'pacf':float(p),'pacf_outside_reference':bool(abs(p)>2*pacf_se)} for k,s,p in zip(lags,acf_se,pacf)]}
(base/'turn12_acf_pacf_results.json').write_text(json.dumps(result,indent=2)+'\n')
plt.rcParams.update({'font.size':16,'axes.titlesize':20,'axes.labelsize':17})
fig,axes=plt.subplots(1,2,figsize=(15,6.3),sharey=True)
for ax,values,bounds,title in [(axes[0],r[1:],2*acf_se,'ACF: includes intervening associations'),(axes[1],pacf,np.repeat(2*pacf_se,14),'PACF: adjusts for intervening lags')]:
    ax.fill_between(lags,-bounds,bounds,color='#176a9a',alpha=.10)
    ax.plot(lags,bounds,'--',color='#b45c20',lw=1.8)
    ax.plot(lags,-bounds,'--',color='#b45c20',lw=1.8)
    marker,stems,baseline=ax.stem(lags,values,basefmt=' ')
    plt.setp(marker,color='#176a9a',markersize=6)
    plt.setp(stems,color='#176a9a',linewidth=2)
    ax.axhline(0,color='#333333',lw=1)
    ax.set(title=title,xlabel='Lag (weeks)',xlim=(.5,14.5),ylim=(-.45,.5))
    ax.set_xticks(lags)
    ax.grid(axis='y',alpha=.2)
    ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Sample correlation')
fig.suptitle('First differences: dashed curves are ±2 approximate standard errors',fontsize=19,y=1.01)
fig.tight_layout(pad=2,w_pad=2)
fig.savefig(base/'turn12_acf_pacf.png',dpi=160,bbox_inches='tight')
plt.close(fig)
print(json.dumps(result,indent=2))
