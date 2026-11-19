import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    horizon = 60
    coefficient = 0.7
    shocks = np.zeros(horizon + 1)
    shocks[0] = 1.0
    ma = shocks.copy()
    ma[12:] += coefficient * shocks[:-12]
    ar = np.zeros(horizon + 1)
    for t in range(horizon + 1):
        ar[t] = shocks[t] + (coefficient * ar[t-12] if t >= 12 else 0.0)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.6), constrained_layout=True, sharex=True, sharey=True)
    for ax, response, title in zip(axes, [ma, ar], [
        'Constructed seasonal MA: x(t) = e(t) + 0.7 e(t − 12)',
        'Constructed seasonal AR: x(t) = 0.7 x(t − 12) + e(t)']):
        ax.axhline(0, color='gray', linewidth=.8)
        ax.vlines(np.arange(horizon+1), 0, response, color='#23689b', linewidth=2)
        ax.scatter(np.arange(horizon+1), response, color='#23689b', s=12)
        for t in range(0,horizon+1,12):
            ax.annotate(f'{response[t]:.3g}', (t,response[t]), xytext=(0,8),
                        textcoords='offset points', ha='center', fontsize=11)
        ax.set(title=title, ylabel='Effect of the initial surprise', ylim=(-.08,1.2))
        ax.grid(alpha=.2)
    axes[-1].set(xlabel='Months since one +1 surprise at month 0', xticks=range(0,61,12))
    fig.suptitle('Illustrations only: all later surprises and all earlier deviations set to zero', fontsize=12)
    fig.savefig('turn06_seasonal_mechanisms.png', dpi=170)
    plt.close(fig)
    results = {'illustration_only': True, 'fitted_to_hotel_data': False,
               'assumptions': 'One unit surprise at month 0; all other surprises and all earlier centered process values zero; no nonseasonal terms.',
               'ma_sign_convention': 'plus: x_t = e_t + Theta e_(t-12)',
               'Theta':coefficient, 'Phi':coefficient, 'month':list(range(horizon+1)),
               'seasonal_MA_response':ma.tolist(), 'seasonal_AR_response':ar.tolist()}
    with open('turn06_seasonal_mechanisms_results.json','w') as f:
        json.dump(results,f,indent=2)
    print({t:{'MA':float(ma[t]),'AR':float(ar[t])} for t in range(0,61,12)})


if __name__ == '__main__':
    main()
