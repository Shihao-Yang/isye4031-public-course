import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    data = pd.read_csv('hotel_monthly.csv')
    y = data.rooms_per_day.to_numpy(dtype=float)
    z = y ** .25
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    for row, (values, label) in enumerate([(y, 'Occupied rooms per day'), (z, 'Fourth root of occupied rooms per day')]):
        axes[row, 0].plot(data.t, values, color='#23689b', linewidth=1.5)
        axes[row, 0].set(xlabel='Month index (t)', ylabel=label,
                         title='Original observations' if row == 0 else 'Fourth-root transformation; no differencing')
        for year, series in enumerate(values.reshape(14,12)):
            axes[row, 1].plot(range(1,13), series, color=plt.get_cmap('viridis')(year/13),
                              linewidth=1.2, marker='o', markersize=2.5, alpha=.8)
        axes[row, 1].set(xlabel='Month within each year', ylabel=label,
                         title='Observed years aligned by calendar month')
        axes[row, 1].set_xticks(range(1,13), months, rotation=45)
    for ax in axes.flat:
        ax.grid(alpha=.22)
    fig.savefig('turn02_fourth_root.png', dpi=170)
    plt.close(fig)
    result = {'transformation': 'z_t = y_t ** 0.25', 'differencing': 'none',
              'annual_summary_definition': 'Arithmetic mean and maximum minus minimum of each consecutive January-December block; ranges are descriptive, not isolated seasonal components.',
              'annual_summaries': []}
    for i, (raw, transformed) in enumerate(zip(y.reshape(14,12), z.reshape(14,12)), 1):
        result['annual_summaries'].append({'year': i, 'raw_mean': float(raw.mean()),
            'raw_range': float(np.ptp(raw)), 'fourth_root_mean': float(transformed.mean()),
            'fourth_root_range': float(np.ptp(transformed))})
    with open('turn02_fourth_root_results.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
