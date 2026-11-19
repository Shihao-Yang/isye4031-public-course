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
    diff = np.diff(star)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), constrained_layout=True)
    result = {'transformation': 'y_star_t = rooms_per_day_t ** 0.25',
              'difference': 'w_t = y_star_t - y_star_(t-1)',
              'acf_convention': 'Demean each series separately; lag-k cross-product sum divided by lag-0 sum (adjusted=False); lag zero omitted from figure.',
              'reference_band': 'plus/minus 1.96/sqrt(n), approximate white-noise reference, pointwise, not a stationarity test',
              'series': {}}
    for row, (values, times, key, title, label) in enumerate([
        (star, data.t.to_numpy(), 'fourth_root', 'Before: fourth-root series', 'Fourth-root rooms per day'),
        (diff, data.t.to_numpy()[1:], 'first_difference', 'After: one-month differences', 'Change in fourth-root rooms per day')]):
        axes[row, 0].plot(times, values, color='#23689b', linewidth=1.3)
        if row:
            axes[row, 0].axhline(0, color='gray', linewidth=.8)
        axes[row, 0].set(xlabel='Month index (t)', ylabel=label, title=title)
        correlations = acf(values, nlags=48, adjusted=False, fft=False)
        lags = np.arange(1,49)
        colors = ['#c34f28' if k % 12 == 0 else '#23689b' for k in lags]
        axes[row, 1].vlines(lags, 0, correlations[1:], colors=colors, linewidth=1.6)
        axes[row, 1].scatter(lags, correlations[1:], c=colors, s=12)
        band = 1.96 / np.sqrt(len(values))
        axes[row, 1].axhspan(-band, band, color='gray', alpha=.15)
        axes[row, 1].axhline(0, color='black', linewidth=.7)
        axes[row, 1].set(xlabel='Lag (months)', ylabel='Sample autocorrelation',
                         title='ACF (orange: multiples of 12)', ylim=(-1,1), xticks=[1,6,12,18,24,30,36,42,48])
        result['series'][key] = {'n':len(values), 'mean':float(np.mean(values)),
                                'acf_lags_0_to_48':correlations.tolist(), 'white_noise_band':float(band)}
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.savefig('turn03_first_difference_acf.png', dpi=170)
    plt.close(fig)
    with open('turn03_first_difference_results.json', 'w') as f:
        json.dump(result, f, indent=2)
    for key, values in result['series'].items():
        print(key, 'n=', values['n'], 'ACF:', {k:round(values['acf_lags_0_to_48'][k],4) for k in [1,2,6,12,24,36,48]})


if __name__ == '__main__':
    main()
