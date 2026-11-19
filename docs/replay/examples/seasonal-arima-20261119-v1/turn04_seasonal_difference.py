import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf


def main():
    data = pd.read_csv('hotel_monthly.csv')
    star = data.rooms_per_day.to_numpy(dtype=float) ** .25
    seasonal = star[12:] - star[:-12]
    correlations = acf(seasonal, nlags=48, adjusted=False, fft=False)
    band = 1.96 / np.sqrt(len(seasonal))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    axes[0].plot(data.t.to_numpy()[12:], seasonal, color='#23689b', linewidth=1.4)
    axes[0].axhline(0, color='black', linewidth=.8)
    axes[0].axhline(seasonal.mean(), color='#c34f28', linestyle='--', label='Sample mean')
    axes[0].set(xlabel='Month index (t)', ylabel='Year-over-year change\non fourth-root scale',
                title='Seasonal difference: y* at month t minus y* at month t − 12')
    axes[0].legend()
    lags = np.arange(1,49)
    colors = ['#c34f28' if k % 12 == 0 else '#23689b' for k in lags]
    axes[1].vlines(lags, 0, correlations[1:], colors=colors, linewidth=1.7)
    axes[1].scatter(lags, correlations[1:], c=colors, s=15)
    axes[1].axhspan(-band, band, color='gray', alpha=.15)
    axes[1].axhline(0, color='black', linewidth=.8)
    axes[1].set(xlabel='Lag (months)', ylabel='Sample autocorrelation', ylim=(-1,1),
                title='ACF of seasonal differences (orange: multiples of 12)',
                xticks=[1,6,12,18,24,30,36,42,48])
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig('turn04_seasonal_difference_acf.png', dpi=170)
    plt.close(fig)
    result = {'definition': 'v_t = y_star_t - y_star_(t-12); y_star_t = rooms_per_day_t ** 0.25; no one-month differencing applied',
              'n':len(seasonal), 't': data.t.to_numpy()[12:].tolist(),
              'seasonal_differences': seasonal.tolist(), 'mean': float(seasonal.mean()),
              'first_36_mean': float(seasonal[:36].mean()), 'last_36_mean': float(seasonal[-36:].mean()),
              'acf_lags_0_to_48': correlations.tolist(),
              'acf_convention': 'Subtract sample mean; adjusted=False; lag-k cross-product sum divided by lag-zero sum; fft=False',
              'pointwise_white_noise_reference_band': float(band)}
    with open('turn04_seasonal_difference_results.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k:result[k] for k in ['n','mean','first_36_mean','last_36_mean']}, indent=2))
    print({k:round(float(correlations[k]),4) for k in [1,2,3,6,12,24,36,48]})


if __name__ == '__main__':
    main()
