import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera


def main():
    data=pd.read_csv('hotel_monthly.csv')
    star=data.rooms_per_day.to_numpy(dtype=float)**.25
    v=star[12:]-star[:-12]
    with open('turn08_candidate_fit_results.json') as f:
        saved=json.load(f)
    params=saved['with_constant']['parameters']
    model=SARIMAX(v,order=([1,3,5],0,0),seasonal_order=(0,0,1,12),trend='c',
                  enforce_stationarity=True,enforce_invertibility=True)
    fit=model.filter([params[name] for name in model.param_names])
    residual=fit.filter_results.standardized_forecasts_error[0]
    times=data.t.to_numpy()[12:]
    a=acf(residual,nlags=36,adjusted=False,fft=False)
    band=1.96/np.sqrt(len(residual))
    lb=acorr_ljungbox(residual,lags=[12,24,36],model_df=4,return_df=True)
    sensitivity=acorr_ljungbox(residual[12:],lags=[12,24,36],model_df=4,return_df=True)
    jb=jarque_bera(residual)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    axes[0,0].plot(times,residual,color='#23689b',linewidth=1.2)
    axes[0,0].axhline(0,color='black',linewidth=.8)
    axes[0,0].axhspan(-2,2,color='gray',alpha=.12)
    axes[0,0].set(xlabel='Month index (t)',ylabel='Standardized prediction error',title='One-step errors, including stationary initialization')
    lags=np.arange(1,37)
    colors=['#c34f28' if k%12==0 else '#23689b' for k in lags]
    axes[0,1].vlines(lags,0,a[1:],colors=colors,linewidth=1.6)
    axes[0,1].scatter(lags,a[1:],c=colors,s=12)
    axes[0,1].axhspan(-band,band,color='gray',alpha=.15)
    axes[0,1].axhline(0,color='black',linewidth=.8)
    axes[0,1].set(xlabel='Lag (months)',ylabel='Sample autocorrelation',title='Residual ACF',ylim=(-.4,.4),xticks=[1,3,5,6,12,18,24,30,36])
    sm.qqplot(residual,line='45',ax=axes[1,0],markerfacecolor='#23689b',markeredgecolor='#23689b',markersize=4)
    axes[1,0].set(title='Normal Q–Q plot')
    axes[1,1].scatter(fit.fittedvalues,residual,color='#23689b',s=18,alpha=.75)
    axes[1,1].axhline(0,color='black',linewidth=.8)
    axes[1,1].set(xlabel='Fitted seasonal difference',ylabel='Standardized prediction error',title='Errors versus fitted values')
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.savefig('turn09_residual_diagnostics.png',dpi=170)
    plt.close(fig)
    idx=np.argsort(np.abs(residual))[-5:][::-1]
    result={'residual_definition':'Standardized one-step-ahead state-space forecast errors (error divided by its model-implied conditional standard deviation); full 156 observations',
        'parameters_source':'turn08_candidate_fit_results.json; filtered at saved parameters, no refitting',
        'ljung_box_convention':'model_df=4: three estimated AR coefficients plus one seasonal MA coefficient; chi-square reference df=lag-4; intercept and innovation variance not counted',
        'ljung_box':lb.reset_index(names='lag').to_dict(orient='records'),
        'sensitivity_omit_first_12':sensitivity.reset_index(names='lag').to_dict(orient='records'),
        'acf_lags_0_to_36':a.tolist(),'pointwise_reference_band':float(band),
        'jarque_bera':{'statistic':float(jb.statistic),'p_value':float(jb.pvalue)},
        'standardized_error_mean':float(residual.mean()),'standardized_error_sd':float(residual.std(ddof=1)),
        'first_half_mean_square':float(np.mean(residual[:78]**2)),
        'second_half_mean_square':float(np.mean(residual[78:]**2)),
        'largest_absolute_errors':[{'t':int(times[i]),'standardized_error':float(residual[i])} for i in idx],
        't':times.tolist(),'standardized_errors':residual.tolist()}
    with open('turn09_residual_diagnostics_results.json','w') as f:
        json.dump(result,f,indent=2)
    print(lb)
    print('Omit initial 12:',sensitivity)
    print('ACF exceeding band:',{k:round(float(a[k]),4) for k in range(1,37) if abs(a[k])>band})
    print('Seasonal ACF:',{k:round(float(a[k]),4) for k in [12,24,36]})
    print('JB',jb)
    print('Mean squares',result['first_half_mean_square'],result['second_half_mean_square'])
    print('Largest',result['largest_absolute_errors'])


if __name__=='__main__':
    main()
