import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf


def main():
    data = pd.read_csv('hotel_monthly.csv')
    star = data.rooms_per_day.to_numpy(dtype=float)**.25
    seasonal = star[12:] - star[:-12]
    combined = np.diff(seasonal)
    regular = np.diff(star)
    reverse = regular[12:] - regular[:-12]
    result = {'definition': 'w_t = y_star_t - y_star_(t-1) - y_star_(t-12) + y_star_(t-13)',
              'order_agreement_max_abs_error':float(np.max(np.abs(combined-reverse))),
              'acf_convention': 'Demean each series; lag-k cross-product sum divided by lag-zero sum, adjusted=False, fft=False',
              'bands': 'Pointwise approximate white-noise reference: plus/minus 1.96/sqrt(n)', 'series': {}}
    fig, axes = plt.subplots(2,2,figsize=(12,7.6),constrained_layout=True)
    for row, (key, values, times, title) in enumerate([
        ('seasonal_only',seasonal,data.t.to_numpy()[12:],'Seasonal difference only'),
        ('combined',combined,data.t.to_numpy()[13:],'Seasonal plus one-month difference')]):
        correlations = acf(values,nlags=48,adjusted=False,fft=False)
        band = 1.96/np.sqrt(len(values))
        axes[row,0].plot(times,values,color='#23689b',linewidth=1.3)
        axes[row,0].axhline(0,color='black',linewidth=.8)
        axes[row,0].axhline(values.mean(),color='#c34f28',linestyle='--',label='Sample mean')
        axes[row,0].set(xlabel='Month index (t)',ylabel='Difference on fourth-root scale',title=title,ylim=(-.15,.15))
        axes[row,0].legend(loc='lower left')
        lags=np.arange(1,49)
        colors=['#c34f28' if k%12==0 else '#23689b' for k in lags]
        axes[row,1].vlines(lags,0,correlations[1:],colors=colors,linewidth=1.6)
        axes[row,1].scatter(lags,correlations[1:],c=colors,s=12)
        axes[row,1].axhspan(-band,band,color='gray',alpha=.15)
        axes[row,1].axhline(0,color='black',linewidth=.8)
        axes[row,1].set(xlabel='Lag (months)',ylabel='Sample autocorrelation',title='ACF (orange: multiples of 12)',ylim=(-1,1),xticks=[1,6,12,18,24,30,36,42,48])
        result['series'][key]={'n':len(values),'mean':float(values.mean()),'sample_sd':float(values.std(ddof=1)),
            'first_36_mean':float(values[:36].mean()),'last_36_mean':float(values[-36:].mean()),
            'acf_lags_0_to_48':correlations.tolist(),'reference_band':float(band)}
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.savefig('turn05_combined_differences.png',dpi=170)
    plt.close(fig)
    result['first_combined_example']={'t':14,'y_star_t':float(star[13]),'y_star_t_minus_1':float(star[12]),
        'y_star_t_minus_12':float(star[1]),'y_star_t_minus_13':float(star[0]),'combined':float(combined[0])}
    with open('turn05_combined_results.json','w') as f:
        json.dump(result,f,indent=2)
    for key,s in result['series'].items():
        print(key,{k:v for k,v in s.items() if k!='acf_lags_0_to_48'})
        print('ACF', {k:round(s['acf_lags_0_to_48'][k],4) for k in [1,2,3,11,12,13,24,36]})
    print(result['first_combined_example'])


if __name__=='__main__':
    main()
