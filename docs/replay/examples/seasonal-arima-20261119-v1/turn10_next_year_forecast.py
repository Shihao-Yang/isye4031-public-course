import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.tsa.statespace.sarimax import SARIMAX


def main():
    data=pd.read_csv('hotel_monthly.csv')
    raw=data.rooms_per_day.to_numpy(dtype=float)
    star=raw**.25
    v=star[12:]-star[:-12]
    with open('turn08_candidate_fit_results.json') as f:
        saved=json.load(f)
    params=saved['with_constant']['parameters']
    model=SARIMAX(v,order=([1,3,5],0,0),seasonal_order=(0,0,1,12),trend='c',
                  enforce_stationarity=True,enforce_invertibility=True)
    fit=model.filter([params[name] for name in model.param_names])
    pred=fit.get_forecast(12)
    dv=np.asarray(pred.predicted_mean)
    variance=np.asarray(pred.var_pred_mean)
    means=star[-12:]+dv
    sd=np.sqrt(variance)
    z=norm.ppf(.975)
    lo_star=means-z*sd
    hi_star=means+z*sd
    median=means**4
    mean_rooms=means**4+6*means**2*variance+3*variance**2
    lower=lo_star**4
    upper=hi_star**4
    months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    rows=[]
    for h in range(12):
        rows.append({'t':169+h,'month':months[h],'forecast_seasonal_difference':float(dv[h]),
            'last_year_fourth_root':float(star[-12+h]),'fourth_root_mean':float(means[h]),
            'fourth_root_sd':float(sd[h]),'fourth_root_95_interval':[float(lo_star[h]),float(hi_star[h])],
            'rooms_per_day_median':float(median[h]),'rooms_per_day_mean':float(mean_rooms[h]),
            'rooms_per_day_95_prediction_interval':[float(lower[h]),float(upper[h])],
            'gaussian_probability_negative_fourth_root':float(norm.cdf(-means[h]/sd[h]))})
    contributions={'constant':params['intercept']}
    for lag in [1,3,5]:
        contributions['AR_lag_'+str(lag)]=float(params['ar.L'+str(lag)]*v[-lag])
    contributions['AR_plus_constant_sum']=sum(contributions.values())
    contributions['seasonal_innovation_state_contribution']=float(dv[0]-contributions['AR_plus_constant_sum'])
    # At future t=169, the lagged seasonal innovation is at t=157.
    contributions['implied_conditional_innovation_t157']=float(contributions['seasonal_innovation_state_contribution']/params['ma.S.L12'])
    result={'forecast_origin_t':168,'horizon':12,'parameters_source':'turn08_candidate_fit_results.json',
        'interval_convention':'Pointwise central 95% Gaussian state-space prediction intervals conditional on saved fitted parameters; state and future innovation uncertainty included; parameter, transformation and model selection uncertainty excluded.',
        'inverse_transform':'median=m^4; Gaussian fourth moment mean=m^4+6*m^2*s^2+3*s^4; limits=(m +/- 1.95996*s)^4, valid here because both root-scale limits are positive and negative root-scale probability is negligible.',
        'first_month_calculation':{'AR_lagged_values':{str(lag):float(v[-lag]) for lag in [1,3,5]},
            'contributions':contributions,'last_year_rooms_t157':float(raw[-12]),'last_year_root_t157':float(star[-12])},
        'forecasts':rows}
    with open('turn10_next_year_forecast_results.json','w') as f:
        json.dump(result,f,indent=2)
    pd.DataFrame([{'t':r['t'],'month':r['month'],'mean_rooms_per_day':r['rooms_per_day_mean'],
        'median_rooms_per_day':r['rooms_per_day_median'],'lower_95':r['rooms_per_day_95_prediction_interval'][0],
        'upper_95':r['rooms_per_day_95_prediction_interval'][1]} for r in rows]).to_csv('turn10_next_year_forecast.csv',index=False)
    fig,ax=plt.subplots(figsize=(11,5),constrained_layout=True)
    ax.plot(data.t.to_numpy()[-36:],raw[-36:],color='#23689b',label='Observed')
    future=np.arange(169,181)
    ax.plot(future,mean_rooms,color='#c34f28',marker='o',label='Forecast mean')
    ax.fill_between(future,lower,upper,color='#c34f28',alpha=.2,label='Pointwise 95% prediction interval')
    ax.axvline(168.5,color='gray',linestyle='--',linewidth=1)
    ax.set(xlabel='Month index (t)',ylabel='Monthly average occupied rooms per day',
           title='Traveler’s Rest: next-year forecasts across four hotels',xticks=[133,145,157,169,174,180])
    ax.legend(loc='upper left')
    ax.grid(alpha=.2)
    fig.savefig('turn10_next_year_forecast.png',dpi=170)
    plt.close(fig)
    print(json.dumps(result['first_month_calculation'],indent=2))
    print(json.dumps(rows[0],indent=2))
    for r in rows:
        print(r['month'],round(r['rooms_per_day_mean']),round(r['rooms_per_day_median']),[round(x) for x in r['rooms_per_day_95_prediction_interval']])


if __name__=='__main__':
    main()
