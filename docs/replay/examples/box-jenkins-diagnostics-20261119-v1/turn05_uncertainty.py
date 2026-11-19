import json
import warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2
from statsmodels.tsa.arima.model import ARIMA


def fit_model(z, constant):
    model = ARIMA(z, order=(0,0,1), trend='c' if constant else 'n', enforce_invertibility=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fit = model.fit(method='statespace', cov_type='oim', method_kwargs={'maxiter': 300, 'disp': 0, 'pgtol': 1e-10, 'factr': 100.0})
    signs = np.array([1., -1., 1.] if constant else [-1., 1.])
    names = ['delta','theta','sigma2'] if constant else ['theta','sigma2']
    estimates = signs * fit.params
    covariance = np.outer(signs,signs)*fit.cov_params()
    rows = {}
    for name, estimate, se in zip(names, estimates, np.sqrt(np.diag(covariance))):
        statistic = estimate/se
        rows[name] = {'estimate': float(estimate), 'se': float(se), 'z': float(statistic),
                      'two_sided_normal_p': float(2*norm.sf(abs(statistic))),
                      'wald_95_interval': [float(estimate-1.96*se),float(estimate+1.96*se)]}
    return {'parameters': rows, 'covariance': covariance.tolist(), 'parameter_order': names,
            'covariance_method': 'observed information matrix, statsmodels oim',
            'loglike': float(fit.llf), 'aic': float(fit.aic), 'bic': float(fit.bic),
            'converged': bool(fit.mle_retvals['converged']), 'iterations': int(fit.mle_retvals['iterations']),
            'warnings': [str(w.message) for w in caught]}


def main():
    y = np.genfromtxt('paper_towels.csv', delimiter=',', skip_header=1)[:,1]
    z = np.diff(y)
    no = fit_model(z,False)
    yes = fit_model(z,True)
    lr = 2*(yes['loglike']-no['loglike'])
    delta = yes['parameters']['delta']
    result = {'n': len(z), 'no_constant': no, 'with_constant': yes,
              'likelihood_ratio': {'statistic': lr,'df':1,'asymptotic_p':float(chi2.sf(lr,1))},
              'delta_rolls_per_week': {'estimate':delta['estimate']*10000,
                                      'wald_95_interval':[v*10000 for v in delta['wald_95_interval']]},
              'note':'Exact Gaussian likelihood on first differences. No residual diagnostics. Wald intervals and tests are asymptotic, conditional on the specified model.'}
    fig,axs = plt.subplots(1,2,figsize=(9,3.5))
    for j,(label,fit) in enumerate([('No constant',no),('With constant',yes)]):
        row = fit['parameters']['theta']
        axs[0].errorbar(row['estimate'],j,xerr=1.96*row['se'],fmt='o',capsize=5,label=label)
    axs[0].set(yticks=[0,1],yticklabels=['No constant','With constant'],xlabel='Textbook theta',title='Approximate 95% Wald intervals',ylim=(-.6,1.6))
    axs[0].axvline(0,color='gray',linestyle=':')
    axs[1].errorbar(delta['estimate'],0,xerr=1.96*delta['se'],fmt='o',capsize=5)
    axs[1].axvline(0,color='gray',linestyle=':')
    axs[1].set(yticks=[],xlabel='Mean weekly change delta (10,000 rolls)',title='Estimated constant, with uncertainty',ylim=(-1,1))
    for ax in axs: ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig('turn05_uncertainty.png',dpi=170)
    plt.close(fig)
    Path('turn05_uncertainty.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
