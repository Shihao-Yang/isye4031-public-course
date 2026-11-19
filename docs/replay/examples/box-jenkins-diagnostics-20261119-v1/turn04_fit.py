import json
import warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from statsmodels.tsa.arima.model import ARIMA


def css_errors(z, theta):
    e = np.empty(len(z))
    previous = 0.0
    for i, value in enumerate(z):
        e[i] = value + theta*previous
        previous = e[i]
    return e


def css(z, theta):
    e = css_errors(z, theta)
    return float(e @ e)


def main():
    y = np.genfromtxt('paper_towels.csv', delimiter=',', skip_header=1)[:, 1]
    z = np.diff(y)
    centered = z-z.mean()
    r1 = float(centered[1:] @ centered[:-1] / (centered @ centered))
    moment = float(-2*r1/(1+np.sqrt(1-4*r1*r1))) if abs(r1) <= .5 else None
    result = {'n_levels': len(y), 'n_differences': len(z), 'sample_r1': r1,
              'moment_theta': moment, 'mean_difference': float(z.mean()),
              'model': 'z_t = a_t - theta*a_(t-1), no constant', 'warnings': []}
    trace = []
    def objective(theta):
        val = css(z, theta)
        trace.append({'theta': float(theta), 'SSE': val})
        return val
    opt = minimize_scalar(objective, bounds=(-.9999, .9999), method='bounded', options={'xatol': 1e-12})
    result['css'] = {'theta': float(opt.x), 'SSE': float(opt.fun), 'sigma2_SSE_over_n': float(opt.fun/len(z)),
                     'success': bool(opt.success), 'message': str(opt.message), 'evaluations': int(opt.nfev),
                     'initialization': 'a_0=0 at start of differenced sample', 'evaluation_trace': trace}
    model = ARIMA(z, order=(0,0,1), trend='n', enforce_invertibility=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fits = []
        for label, starttheta in [('moment', moment), ('zero', 0.0), ('negative', -.7), ('positive', .8)]:
            if starttheta is None:
                continue
            path = []
            def callback(params):
                constrained = model.transform_params(np.asarray(params))
                path.append({'theta': float(-constrained[0]), 'sigma2': float(constrained[1]),
                             'loglike': float(model.loglike(constrained))})
            fit = model.fit(start_params=[-starttheta, np.var(z)], method='statespace',
                            method_kwargs={'maxiter': 300, 'disp': 0, 'callback': callback, 'pgtol': 1e-10, 'factr': 100.0})
            fits.append({'start': label, 'start_theta': starttheta, 'theta': float(-fit.params[0]),
                         'sigma2': float(fit.params[1]), 'loglike': float(fit.llf),
                         'theta_se': float(fit.bse[0]), 'optimizer': fit.mle_retvals, 'iteration_trace': path})
        result['warnings'] = [str(w.message) for w in caught]
    result['exact_gaussian_fits'] = fits
    grid = np.linspace(-.95, .95, 381)
    ss = np.array([css(z, t) for t in grid])
    # Exact stationary Gaussian covariance: diagonal 1+theta², first off-diagonal -theta.
    # Its eigenvectors do not depend on theta, allowing an exact profile over sigma².
    n = len(z)
    i = np.arange(1,n+1)
    Q = np.sqrt(2/(n+1))*np.sin(np.pi*np.outer(i,i)/(n+1))
    projection = Q.T @ z
    def exact_profile(t):
        eig = 1+t*t-2*t*np.cos(np.pi*i/(n+1))
        variance = np.sum(projection**2/eig)/n
        return float(-.5*(n*(np.log(2*np.pi)+1+np.log(variance))+np.log(eig).sum())), float(variance)
    profile = np.array([exact_profile(t)[0] for t in grid])
    ep = minimize_scalar(lambda t: -exact_profile(t)[0], bounds=(-.9999,.9999), method='bounded', options={'xatol': 1e-12})
    result['independent_exact_profile'] = {'theta': float(ep.x), 'loglike': float(-ep.fun), 'sigma2': exact_profile(ep.x)[1], 'success': bool(ep.success)}
    result['grid'] = {'theta': grid.tolist(), 'conditional_SSE': ss.tolist(), 'exact_profile_loglike': profile.tolist()}
    fig, axs = plt.subplots(1,3,figsize=(13,3.9))
    axs[0].plot(grid,ss)
    axs[0].scatter([opt.x],[opt.fun],color='crimson',zorder=3)
    axs[0].set(xlabel='Textbook theta',ylabel='Conditional SSE',title='Conditional least-squares search')
    axs[1].plot(grid, profile)
    axs[1].scatter([ep.x],[-ep.fun],color='crimson',zorder=3)
    axs[1].set(xlabel='Textbook theta',ylabel='Profile log likelihood',title='Exact Gaussian likelihood')
    for f in fits:
        points = [f['start_theta']] + [p['theta'] for p in f['iteration_trace']]
        axs[2].plot(range(len(points)),points,marker='.',label=f['start'])
    axs[2].axhline(ep.x,color='black',linestyle=':',linewidth=1)
    axs[2].set(xlabel='Optimizer iteration (0 = start)',ylabel='Textbook theta',title='Exact-likelihood convergence')
    axs[2].legend(fontsize=8)
    for ax in axs:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig('turn04_search.png',dpi=170)
    plt.close(fig)
    def convert(v):
        if isinstance(v,np.ndarray): return v.tolist()
        if isinstance(v,np.generic): return v.item()
        raise TypeError(type(v).__name__)
    Path('turn04_fit.json').write_text(json.dumps(result,indent=2,default=convert))
    print(json.dumps({k:v for k,v in result.items() if k not in ['grid','exact_gaussian_fits','css']},indent=2))
    print('CSS', opt.x,opt.fun,opt.fun/len(z),opt.success,opt.nfev)
    for f in fits:
        print({k:v for k,v in f.items() if k != 'iteration_trace'})


if __name__ == '__main__':
    main()
