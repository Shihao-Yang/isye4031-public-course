import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from turn08_fit_candidate import fit_candidate


def metrics(actual, predicted):
    errors=predicted-actual
    return {'MAE':float(np.mean(np.abs(errors))), 'RMSE':float(np.sqrt(np.mean(errors**2))),
            'mean_error_forecast_minus_actual':float(np.mean(errors))}


def main():
    data=pd.read_csv('hotel_monthly.csv')
    raw=data.rooms_per_day.to_numpy(dtype=float)
    train=raw[:156]
    actual=raw[156:]
    baseline=raw[144:156]
    root=train**.25
    v=root[12:]-root[:-12]
    fitted=fit_candidate(v,'c')
    model=SARIMAX(v,order=([1,3,5],0,0),seasonal_order=(0,0,1,12),trend='c',
                  enforce_stationarity=True,enforce_invertibility=True)
    fit=model.filter([fitted['parameters'][name] for name in model.param_names])
    forecast=fit.get_forecast(12)
    m=root[-12:]+np.asarray(forecast.predicted_mean)
    variance=np.asarray(forecast.var_pred_mean)
    mean=m**4+6*m*m*variance+3*variance**2
    median=m**4
    lower=(m-norm.ppf(.975)*np.sqrt(variance))**4
    upper=(m+norm.ppf(.975)*np.sqrt(variance))**4
    months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    rows=[{'t':157+i,'month':months[i],'actual':float(actual[i]),'candidate_mean':float(mean[i]),
           'candidate_median':float(median[i]),'seasonal_naive':float(baseline[i]),
           'candidate_lower_95':float(lower[i]),'candidate_upper_95':float(upper[i])} for i in range(12)]
    out={'design':'Single origin t=156; refit on t=1,...,156; predict all 12 months t=157,...,168 without updating.',
         'limitation':'Retrospective illustration: model structure and fourth-root transform were chosen with access to all observations. Not an untouched test.',
         'training_fit':fitted,'candidate_mean_metrics':metrics(actual,mean),
         'candidate_median_metrics':metrics(actual,median),'seasonal_naive_metrics':metrics(actual,baseline),
         'months_candidate_mean_has_smaller_absolute_error':int(np.sum(np.abs(mean-actual)<np.abs(baseline-actual))),
         'candidate_95_interval_coverage_count':int(np.sum((actual>=lower)&(actual<=upper))),
         'forecasts':rows}
    out['relative_MAE_reduction']=1-out['candidate_mean_metrics']['MAE']/out['seasonal_naive_metrics']['MAE']
    out['relative_RMSE_reduction']=1-out['candidate_mean_metrics']['RMSE']/out['seasonal_naive_metrics']['RMSE']
    with open('turn11_retrospective_results.json','w') as f:
        json.dump(out,f,indent=2)
    pd.DataFrame(rows).to_csv('turn11_retrospective_forecasts.csv',index=False)
    fig,axes=plt.subplots(2,1,figsize=(11,7.5),constrained_layout=True,sharex=True)
    x=np.arange(1,13)
    axes[0].fill_between(x,lower,upper,color='#c34f28',alpha=.15,label='Candidate pointwise 95% interval')
    axes[0].plot(x,actual,'o-',color='black',label='Observed year 14')
    axes[0].plot(x,mean,'o-',color='#c34f28',label='Candidate forecast mean')
    axes[0].plot(x,baseline,'s--',color='#23689b',label='Year 13 same-month value')
    axes[0].set(ylabel='Occupied rooms per day',title='Retrospective forecasts from the end of year 13')
    axes[0].legend(fontsize=9)
    axes[1].bar(x-.18,mean-actual,width=.36,color='#c34f28',label='Candidate')
    axes[1].bar(x+.18,baseline-actual,width=.36,color='#23689b',label='Same-month last year')
    axes[1].axhline(0,color='black',linewidth=.8)
    axes[1].set(xlabel='Month of year 14',ylabel='Forecast minus observed\n(rooms per day)',xticks=x,xticklabels=months)
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig('turn11_retrospective_comparison.png',dpi=170)
    plt.close(fig)
    print(json.dumps({k:value for k,value in out.items() if k not in ['training_fit','forecasts']},indent=2))
    print('Training fit:',fitted['parameters'],fitted['attempts'])
    for r in rows:
        print(r['month'],r['actual'],round(r['candidate_mean'],1),r['seasonal_naive'])


if __name__=='__main__':
    main()
