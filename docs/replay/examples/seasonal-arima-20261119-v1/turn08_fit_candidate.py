import json
import warnings
import platform
import numpy as np
import pandas as pd
import scipy
import statsmodels
from scipy.stats import chi2
from statsmodels.tsa.statespace.sarimax import SARIMAX


def fit_candidate(v, trend):
    model = SARIMAX(v, order=([1,3,5],0,0), seasonal_order=(0,0,1,12), trend=trend,
                    enforce_stationarity=True, enforce_invertibility=True)
    attempts=[]
    start=None
    for method in ['lbfgs','powell','lbfgs']:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fit=model.fit(start_params=start,method=method,maxiter=3000,disp=False)
        attempts.append({'method':method,'converged':bool(fit.mle_retvals.get('converged',False)),
                         'warnings':[str(w.message) for w in caught]})
        start=fit.params
        if attempts[-1]['converged']:
            break
    if not attempts[-1]['converged']:
        raise RuntimeError('Candidate failed to converge: '+str(attempts))
    params=dict(zip(fit.param_names, map(float,fit.params)))
    out={'parameters':params,'standard_errors':dict(zip(fit.param_names,map(float,fit.bse))),
         'confidence_intervals_95':dict(zip(fit.param_names,fit.conf_int().tolist())),
         'log_likelihood':float(fit.llf),'AIC':float(fit.aic),'BIC':float(fit.bic),
         'nobs':int(fit.nobs),'attempts':attempts,
         'minimum_AR_root_modulus':float(np.min(np.abs(fit.arroots))),
         'minimum_MA_root_modulus':float(np.min(np.abs(fit.maroots)))}
    if trend=='c':
        denom=1-sum(params['ar.L'+str(k)] for k in [1,3,5])
        mean=params['intercept']/denom
        grad=np.zeros(len(fit.params))
        grad[fit.param_names.index('intercept')]=1/denom
        for k in [1,3,5]:
            grad[fit.param_names.index('ar.L'+str(k))]=params['intercept']/denom**2
        se=float(np.sqrt(grad @ fit.cov_params() @ grad))
        out['implied_mean']={'estimate':float(mean),'delta_method_se':se,
                              'approx_95_interval':[float(mean-1.96*se),float(mean+1.96*se)]}
    return out


def main():
    data=pd.read_csv('hotel_monthly.csv')
    star=data.rooms_per_day.to_numpy(dtype=float)**.25
    v=star[12:]-star[:-12]
    with_c=fit_candidate(v,'c')
    without_c=fit_candidate(v,'n')
    lr=2*(with_c['log_likelihood']-without_c['log_likelihood'])
    out={'data_definition':'Seasonal differences of fourth-root occupied rooms per day; no regular difference; t=13,...,168',
         'equation':'v_t = c + phi_1 v_(t-1) + phi_3 v_(t-3) + phi_5 v_(t-5) + e_t + Theta e_(t-12)',
         'estimation':'statsmodels SARIMAX Gaussian state-space maximum likelihood, stationary initialization; stationarity and invertibility enforced; default OPG covariance',
         'sample_mean':float(v.mean()),'with_constant':with_c,'without_constant':without_c,
         'likelihood_ratio':{'statistic':float(lr),'df':1,'approx_p_value':float(chi2.sf(lr,1))},
         'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__, 'statsmodels':statsmodels.__version__}}
    with open('turn08_candidate_fit_results.json','w') as f:
        json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    main()
