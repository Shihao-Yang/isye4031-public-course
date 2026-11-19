import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf


def main():
    data = pd.read_csv('hotel_monthly.csv')
    star = data.rooms_per_day.to_numpy(dtype=float)**.25
    v = star[12:] - star[:-12]
    a = acf(v,nlags=24,adjusted=False,fft=False)
    p = pacf(v,nlags=24,method='ywm')
    band = 1.96/np.sqrt(len(v))
    fig, axes = plt.subplots(2,1,figsize=(11,7),constrained_layout=True,sharex=True)
    lags = np.arange(1,25)
    colors = ['#c34f28' if k%12==0 else '#23689b' for k in lags]
    for ax, values, title in zip(axes,[a,p],['ACF of seasonal differences','PACF of seasonal differences (Yule–Walker, unadjusted)']):
        ax.vlines(lags,0,values[1:],colors=colors,linewidth=2)
        ax.scatter(lags,values[1:],c=colors,s=20)
        ax.axhspan(-band,band,color='gray',alpha=.15)
        ax.axhline(0,color='black',linewidth=.8)
        ax.set(ylabel='Sample correlation',title=title,ylim=(-.6,.6))
        ax.grid(alpha=.2)
    axes[-1].set(xlabel='Lag (months); orange marks lags 12 and 24',xticks=range(1,25))
    fig.suptitle('Hotel: v(t) = y*(t) − y*(t − 12), n = 156',fontsize=13)
    fig.savefig('turn07_seasonal_acf_pacf.png',dpi=170)
    plt.close(fig)
    results = {'series':'v_t = rooms_per_day_t ** 0.25 - rooms_per_day_(t-12) ** 0.25; seasonal difference only',
        'n':len(v),'acf_convention':'Demeaned, adjusted=False, fft=False, fixed lag-zero sum denominator',
        'pacf_convention':'statsmodels pacf method=ywm: Yule-Walker with unadjusted autocovariances',
        'pointwise_white_noise_band':float(band),'lags':list(range(25)), 'acf':a.tolist(),'pacf':p.tolist()}
    with open('turn07_seasonal_acf_pacf_results.json','w') as f:
        json.dump(results,f,indent=2)
    print('lag ACF PACF')
    for k in range(1,25):
        print(k,round(float(a[k]),4),round(float(p[k]),4))


if __name__=='__main__':
    main()
