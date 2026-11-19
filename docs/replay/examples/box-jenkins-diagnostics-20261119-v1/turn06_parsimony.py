import json
import warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2
from statsmodels.tsa.arima.model import ARIMA


def fit_model(z, p, start=None):
    model = ARIMA(z, order=(p,0,1), trend='n', enforce_stationarity=True, enforce_invertibility=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fit = model.fit(start_params=start, method='statespace', cov_type='oim',
                        method_kwargs={'maxiter':500,'disp':0,'pgtol':1e-10,'factr':100.0})
    signs = np.array([1.,-1.,1.] if p else [-1.,1.])
    names = ['phi','theta','sigma2'] if p else ['theta','sigma2']
    estimates = signs*fit.params
    cov = np.outer(signs,signs)*fit.cov_params()
    se = np.sqrt(np.diag(cov))
    cor = cov/np.outer(se,se)
    rows = {name:{'estimate':float(est),'se':float(s),'z':float(est/s),
                  'normal_p':float(2*norm.sf(abs(est/s))),
                  'wald_95_interval':[float(est-1.96*s),float(est+1.96*s)]}
            for name,est,s in zip(names,estimates,se)}
    return {'parameters':rows,'parameter_order':names,'covariance':cov.tolist(),'correlation':cor.tolist(),
            'loglike':float(fit.llf),'aic':float(fit.aic),'bic':float(fit.bic),
            'converged':bool(fit.mle_retvals['converged']),'iterations':int(fit.mle_retvals['iterations']),
            'warnings':[str(w.message) for w in caught]}


def main():
    y = np.genfromtxt('paper_towels.csv',delimiter=',',skip_header=1)[:,1]
    z = np.diff(y)
    ma = fit_model(z,0)
    fits = [fit_model(z,1,start) for start in [None,[0.,.35,1.07],[.6,-.2,1.07],[-.5,.7,1.07]]]
    arma = max(fits,key=lambda f:f['loglike'])
    lr = 2*(arma['loglike']-ma['loglike'])
    result = {'n':len(z),'convention':'z_t = phi*z_(t-1) + a_t - theta*a_(t-1), no constant',
              'covariance_method':'observed information matrix (oim)', 'ma1':ma,'arma11':arma,
              'arma11_multistart':fits,'likelihood_ratio':{'statistic':lr,'df':1,'asymptotic_p':float(chi2.sf(lr,1))}}
    fig,axs = plt.subplots(1,3,figsize=(12,3.7))
    for j,(label,f) in enumerate([('MA(1)',ma),('ARMA(1,1)',arma)]):
        row = f['parameters']['theta']
        axs[0].errorbar(row['estimate'],j,xerr=1.96*row['se'],fmt='o',capsize=5)
    axs[0].set(yticks=[0,1],yticklabels=['MA(1)','ARMA(1,1)'],xlabel='Textbook theta',title='Approximate 95% Wald intervals',ylim=(-.6,1.6))
    row = arma['parameters']['phi']
    axs[1].errorbar(row['estimate'],0,xerr=1.96*row['se'],fmt='o',capsize=5)
    axs[1].set(yticks=[],xlabel='AR coefficient phi',title='Added AR term',ylim=(-1,1))
    cov = np.array(arma['covariance'])[:2,:2]
    eig,vec = np.linalg.eigh(cov)
    angles = np.linspace(0,2*np.pi,400)
    ellipse = vec @ (np.sqrt(eig*chi2.ppf(.95,2))[:,None]*np.array([np.cos(angles),np.sin(angles)]))
    center = np.array([arma['parameters'][k]['estimate'] for k in ['phi','theta']])
    ellipse += center[:,None]
    axs[2].plot(ellipse[0],ellipse[1]); axs[2].plot(*center,'o')
    axs[2].set(xlabel='phi',ylabel='Textbook theta',title='Approximate 95% joint Wald ellipse')
    for ax in axs[:2]: ax.axvline(0,color='gray',linestyle=':')
    for ax in axs: ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig('turn06_parsimony.png',dpi=170)
    plt.close(fig)
    Path('turn06_parsimony.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='arma11_multistart'},indent=2))
    print('Multistart:',[(f['converged'],f['loglike'],f['warnings']) for f in fits])


if __name__ == '__main__':
    main()
