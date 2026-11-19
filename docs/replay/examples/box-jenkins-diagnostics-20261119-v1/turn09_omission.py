import json
import warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import chi2
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf


def summarize(errors, model_df):
    n = len(errors)
    ac = acf(errors,nlags=24,fft=False)
    pa = pacf(errors,nlags=24,method='ywm')
    tests = []
    for h in [6,12,18,24]:
        k = np.arange(1,h+1)
        bp = float(n*np.sum(ac[1:h+1]**2))
        lb = float(n*(n+2)*np.sum(ac[1:h+1]**2/(n-k)))
        df = h-model_df
        tests.append({'h':h,'df':df,'BP':bp,'BP_p':float(chi2.sf(bp,df)),
                      'LB':lb,'LB_p':float(chi2.sf(lb,df))})
    return {'standardized_errors':errors.tolist(),'acf':ac.tolist(),'pacf':pa.tolist(),
            'model_df':model_df,'tests':tests}


def main():
    y = np.genfromtxt('paper_towels.csv',delimiter=',',skip_header=1)[:,1]
    z = np.diff(y)
    prior = json.loads(Path('turn07_residuals.json').read_text())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fit = ARIMA(z,order=(0,0,0),trend='n').fit(method_kwargs={'disp':0,'maxiter':300})
    omitted = summarize(np.asarray(fit.filter_results.standardized_forecasts_error[0]),0)
    candidate = summarize(np.array(prior['standardized_errors']),1)
    omitted.update({'sigma2':float(fit.params[0]),'loglike':float(fit.llf),'aic':float(fit.aic),
                    'converged':bool(fit.mle_retvals['converged']), 'warnings':[str(w.message) for w in caught]})
    result = {'n':len(z),'comparison':'No-constant ARIMA(0,1,0) vs no-constant ARIMA(0,1,1). Same data, standardized one-step errors, same ACF definition and lag cutoffs.',
              'omitted_MA':omitted,'candidate_MA1':candidate,
              'reference_band':float(1.96/np.sqrt(len(z)))}
    fig,axs = plt.subplots(2,2,figsize=(11,6.5),sharex=True,sharey=True)
    for row,(name,values) in enumerate([('Omit MA: ARIMA(0,1,0)',omitted),('Candidate: ARIMA(0,1,1)',candidate)]):
        for col,key in enumerate(['acf','pacf']):
            ax = axs[row,col]
            ax.stem(np.arange(1,25),values[key][1:],basefmt=' ')
            band = result['reference_band']
            ax.axhspan(-band,band,color='steelblue',alpha=.15)
            ax.axhline(0,color='gray',linewidth=.8)
            ax.set(title=name+' | '+key.upper(),xlabel='Lag (weeks)',ylabel='Residual correlation',ylim=(-.23,.37))
            ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig('turn09_omission.png',dpi=170)
    plt.close(fig)
    Path('turn09_omission.json').write_text(json.dumps(result,indent=2))
    print('Omitted model convergence:',omitted['converged'],'warnings:',omitted['warnings'])
    for label,values in [('Omitted',omitted),('Candidate',candidate)]:
        print(label,'ACF1',values['acf'][1], 'PACF1',values['pacf'][1])
        print(json.dumps(values['tests'],indent=2))
    print('Omitted variance',omitted['sigma2'])


if __name__ == '__main__':
    main()
