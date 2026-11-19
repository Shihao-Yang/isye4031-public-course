import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import chi2


def calculate(ac, n, h, model_df=1):
    k = np.arange(1,h+1)
    rho = ac[1:h+1]
    bp = n*rho**2
    lb = n*(n+2)*rho**2/(n-k)
    df = h-model_df
    return {'h':h,'df':df,'box_pierce':float(bp.sum()),'ljung_box':float(lb.sum()),
            'box_pierce_p':float(chi2.sf(bp.sum(),df)), 'ljung_box_p':float(chi2.sf(lb.sum(),df)),
            'rows':[{'lag':int(j),'residual_acf':float(r),'squared_acf':float(r*r),
                     'BP_contribution':float(b),'LB_contribution':float(l)} for j,r,b,l in zip(k,rho,bp,lb)]}


def main():
    source = json.loads(Path('turn07_residuals.json').read_text())
    ac = np.array(source['acf'])
    n = len(source['standardized_errors'])
    calculations = [calculate(ac,n,h) for h in [6,12,18,24]]
    worked = calculations[1]
    critical = float(chi2.ppf(.95,worked['df']))
    result = {'source':'turn07_residuals.json; unchanged standardized exact-filter errors',
              'n':n,'model_df':1,'df_explanation':'h - p - q = h - 1; the differencing order is not subtracted',
              'calculations':calculations,'worked_h':12,'sum_squared_acf_h12':float(np.sum(ac[1:13]**2)),
              'sum_weighted_squared_acf_h12':float(np.sum(ac[1:13]**2/(n-np.arange(1,13)))),
              'chi_square_95_critical_df11':critical,
              'calibration':'Approximate asymptotic chi-square reference, not an exact finite-sample distribution. Lag cutoffs are illustrative sensitivity checks, not independent tests.'}
    fig,axs = plt.subplots(1,2,figsize=(10,3.8))
    rows = worked['rows']
    k = np.arange(1,13)
    axs[0].bar(k-.18,[r['BP_contribution'] for r in rows],width=.36,label='Box–Pierce')
    axs[0].bar(k+.18,[r['LB_contribution'] for r in rows],width=.36,label='Ljung–Box')
    axs[0].set(xlabel='Lag k',ylabel='Contribution to Q',title='Combining lags 1–12',xticks=k)
    axs[0].legend()
    x = np.linspace(0,32,500)
    q = worked['ljung_box']
    axs[1].plot(x,chi2.pdf(x,worked['df']))
    axs[1].fill_between(x,chi2.pdf(x,worked['df']),where=x>=q,alpha=.25,label=f'Right-tail p = {worked["ljung_box_p"]:.3f}')
    axs[1].axvline(q,color='crimson',label=f'Observed Q = {q:.2f}')
    axs[1].axvline(critical,color='gray',linestyle=':',label=f'95% cutoff = {critical:.2f}')
    axs[1].set(xlabel='Q statistic',ylabel='Reference density',title='Approximate chi-square reference: 11 df')
    axs[1].legend(fontsize=8)
    for ax in axs: ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig('turn08_portmanteau.png',dpi=170)
    plt.close(fig)
    Path('turn08_portmanteau.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
