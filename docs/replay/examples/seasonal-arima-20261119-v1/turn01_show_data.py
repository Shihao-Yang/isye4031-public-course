import json
import platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    data = pd.read_csv('hotel_monthly.csv')
    y = data['rooms_per_day'].to_numpy()
    years = (data['t'].to_numpy() - 1) // 12 + 1
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), constrained_layout=True)
    axes[0].plot(data['t'], y, color='#23689b', linewidth=1.6)
    axes[0].set(xlabel='Month index (t)', ylabel='Occupied rooms per day',
                title='Traveler’s Rest: monthly average occupied rooms across four hotels')
    axes[0].set_xticks([1, 13, 25, 37, 49, 61, 73, 85, 97, 109, 121, 133, 145, 157, 168])
    axes[0].grid(alpha=.22)
    cmap = plt.get_cmap('viridis')
    for year in range(1, 15):
        rows = data.loc[years == year]
        axes[1].plot(rows['month'], rows['rooms_per_day'], marker='o', markersize=3,
                     linewidth=1.3, color=cmap((year-1)/13), alpha=.8)
    axes[1].set(xlabel='Month within each year', ylabel='Occupied rooms per day',
                title='Each line is one observed year (purple: early; yellow: late)')
    axes[1].set_xticks(range(1,13), ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    axes[1].grid(alpha=.22)
    fig.savefig('turn01_hotel_data.png', dpi=170)
    plt.close(fig)
    matrix = y.reshape(14,12)
    result = {
        'observations': len(y), 'years': 14,
        'min_rooms_per_day': int(y.min()), 'max_rooms_per_day': int(y.max()),
        'first_year_mean': float(matrix[0].mean()), 'last_year_mean': float(matrix[-1].mean()),
        'peak_month_by_year': (matrix.argmax(axis=1)+1).tolist(),
        'monthly_means_descriptive': dict(zip(map(str,range(1,13)), matrix.mean(axis=0).tolist())),
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'pandas': pd.__version__, 'matplotlib': matplotlib.__version__}
    }
    with open('turn01_data_summary.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
