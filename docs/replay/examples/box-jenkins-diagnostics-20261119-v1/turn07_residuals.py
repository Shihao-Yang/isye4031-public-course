import json
import warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf


def main():
    data = np.genfromtxt('paper_towels.csv',delimiter=',',skip_header=1)
    z = np.diff(data[:,1])
    weeks = data[1:,0].astype(int)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fit = ARIMA(z,order=(0,0,1),trend='n').fit(cov_type='oim',method_kwargs={'maxiter':300,'disp':0})
    e = np.asarray(fit.filter_results.standardized_forecasts_error[0])
    raw = np.asarray(fit.resid)
    ac = acf(e,nlags=24,fft=False)
    pa = pacf(e,nlags=24,method='ywm')
    jb = stats.jarque_bera(e)
    sw = stats.shapiro(e)
    top = np.argsort(abs(e))[::-1][:5]
    segments = []
    for indices in np.array_split(np.arange(len(e)),3):
        segments.append({'weeks':[int(weeks[indices[0]]),int(weeks[indices[-1]])],
                         'n':len(indices),'mean':float(e[indices].mean()),'sd':float(e[indices].std(ddof=1))})
    rolling = np.array([e[j-19:j+1].std(ddof=1) for j in range(19,len(e))])
    result = {'model':'ARIMA(0,1,1), no constant, fitted to 119 differences',
              'theta':float(-fit.params[0]),'sigma2':float(fit.params[1]),
              'converged':bool(fit.mle_retvals['converged']),'warnings':[str(w.message) for w in caught],
              'residual_definition':'Exact-filter one-step prediction error divided by its model-implied standard deviation; all 119 included; first residual uses stationary initialization.',
              'weeks':weeks.tolist(),'raw_prediction_errors':raw.tolist(),'standardized_errors':e.tolist(),
              'acf':ac.tolist(),'pacf_yule_walker_mle':pa.tolist(),
              'reference_band':float(1.96/np.sqrt(len(e))),
              'mean':float(e.mean()),'sd':float(e.std(ddof=1)),
              'skewness':float(stats.skew(e,bias=False)),'excess_kurtosis':float(stats.kurtosis(e,bias=False)),
              'jarque_bera':{'statistic':float(jb.statistic),'p':float(jb.pvalue)},
              'shapiro_wilk':{'statistic':float(sw.statistic),'p':float(sw.pvalue)},
              'thirds':segments,'rolling_sd_window':20,'rolling_sd':rolling.tolist(),
              'largest_absolute_errors':[{'week':int(weeks[j]),'standardized_error':float(e[j]),'raw_error':float(raw[j])} for j in top],
              'acf_outside_reference':[{'lag':j,'acf':float(ac[j])} for j in range(1,25) if abs(ac[j])>1.96/np.sqrt(len(e))],
              'pacf_outside_reference':[{'lag':j,'pacf':float(pa[j])} for j in range(1,25) if abs(pa[j])>1.96/np.sqrt(len(e))]}
    fig,axs = plt.subplots(3,2,figsize=(12,10))
    ax = axs[0,0]
    ax.plot(weeks,e,linewidth=.9,marker='.',markersize=3)
    ax.axhline(0,color='gray',linewidth=.8)
    for threshold in [-3,3]: ax.axhline(threshold,color='crimson',linestyle=':',linewidth=1)
    for j in top[:3]: ax.annotate(str(weeks[j]),(weeks[j],e[j]),xytext=(4,5),textcoords='offset points')
    ax.set(xlabel='Week',ylabel='Standardized error',title='One-step prediction errors (labels = weeks)')
    ax = axs[0,1]
    ax.plot(weeks[19:],rolling)
    ax.axhline(1,color='gray',linestyle=':')
    ax.set(xlabel='Week at end of trailing window',ylabel='Sample SD',title='Spread: trailing 20 errors')
    for ax,values,title in [(axs[1,0],ac,'Residual ACF'),(axs[1,1],pa,'Residual PACF (Yule–Walker)')]:
        ax.stem(np.arange(1,25),values[1:],basefmt=' ')
        band = 1.96/np.sqrt(len(e))
        ax.axhspan(-band,band,color='steelblue',alpha=.13)
        ax.axhline(0,color='gray',linewidth=.8)
        ax.set(xlabel='Lag (weeks)',ylabel='Correlation',title=title)
    ax = axs[2,0]
    ax.hist(e,bins=14,density=True,alpha=.65,edgecolor='white')
    x = np.linspace(-4,4,300)
    ax.plot(x,stats.norm.pdf(x),label='Standard normal')
    ax.set(xlabel='Standardized error',ylabel='Density',title='Distribution of prediction errors')
    ax.legend()
    stats.probplot(e,dist='norm',plot=axs[2,1])
    axs[2,1].set_title('Normal Q–Q plot (fitted reference line)')
    for ax in axs.flat: ax.grid(alpha=.18)
    fig.tight_layout()
    fig.savefig('turn07_residuals.png',dpi=170)
    plt.close(fig)
    Path('turn07_residuals.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['weeks','raw_prediction_errors','standardized_errors','acf','pacf_yule_walker_mle','rolling_sd']},indent=2))


if __name__ == '__main__':
    main()
