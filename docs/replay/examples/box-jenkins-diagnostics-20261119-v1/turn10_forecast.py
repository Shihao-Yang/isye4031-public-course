import json
import warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA


def main():
    data = np.genfromtxt('paper_towels.csv',delimiter=',',skip_header=1)
    y = data[:,1]
    z = np.diff(y)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fit = ARIMA(z,order=(0,0,1),trend='n').fit(cov_type='oim',method_kwargs={'maxiter':300,'disp':0})
    theta = float(-fit.params[0])
    variance = float(fit.params[1])
    # Sum the model's future differenced observation means and joint covariance.
    # Propagate the final filtered state covariance, preserving initial-state uncertainty.
    fr = fit.filter_results
    state = fr.predicted_state[:,-1].copy()
    state_cov = fr.predicted_state_cov[:,:,-1].copy()
    design = fr.design[:,:,0]
    transition = fr.transition[:,:,0]
    selection = fr.selection[:,:,0]
    process_cov = selection @ fr.state_cov[:,:,0] @ selection.T
    observation_cov = fr.obs_cov[:,:,0]
    horizon = 12
    means = []
    covs = []
    for j in range(horizon):
        means.append(float((design @ state)[0]))
        covs.append(state_cov.copy())
        state = transition @ state
        state_cov = transition @ state_cov @ transition.T + process_cov
    joint = np.zeros((horizon,horizon))
    for j in range(horizon):
        joint[j,j] = float((design @ covs[j] @ design.T + observation_cov)[0,0])
        for k in range(j+1,horizon):
            cross = covs[j] @ np.linalg.matrix_power(transition,k-j).T
            joint[j,k] = joint[k,j] = float((design @ cross @ design.T)[0,0])
    rows = []
    for j in range(horizon):
        h = j+1
        forecast = float(y[-1]+np.sum(means[:h]))
        var_h = float(joint[:h,:h].sum())
        se_rolls = 10000*np.sqrt(var_h)
        rolls = 100000+10000*forecast
        rows.append({'horizon':h,'week':120+h,'forecast_y':forecast,'forecast_rolls':rolls,
                     'variance_y':var_h,'se_rolls':float(se_rolls),
                     'textbook_variance_y':float(variance*(1+(h-1)*(1-theta)**2)),
                     'PI80_rolls':[float(rolls-norm.ppf(.9)*se_rolls),float(rolls+norm.ppf(.9)*se_rolls)],
                     'PI95_rolls':[float(rolls-norm.ppf(.975)*se_rolls),float(rolls+norm.ppf(.975)*se_rolls)]})
    result = {'model':'ARIMA(0,1,1), no drift, fitted to first differences',
              'n_levels':len(y),'theta':theta,'sigma2':variance,'last_observed_y':float(y[-1]),
              'last_observed_rolls':float(100000+10000*y[-1]),'last_filtered_innovation':float(fit.resid[-1]),
              'future_difference_means':means,'future_difference_covariance':joint.tolist(),
              'converged':bool(fit.mle_retvals['converged']),'warnings':[str(w.message) for w in caught],
              'interval_assumptions':'Pointwise Gaussian prediction intervals with fitted parameters fixed; exact filtered-state uncertainty included; no parameter, model-selection, structural-change uncertainty; no future observed values used.',
              'forecasts':rows}
    fig,axs = plt.subplots(1,2,figsize=(12,4.3),gridspec_kw={'width_ratios':[1.7,1]})
    weeks = np.array([r['week'] for r in rows])
    point = np.array([r['forecast_rolls'] for r in rows])/1000
    lo95,hi95 = np.array([r['PI95_rolls'] for r in rows]).T/1000
    lo80,hi80 = np.array([r['PI80_rolls'] for r in rows]).T/1000
    axs[0].plot(data[-40:,0],(100000+10000*y[-40:])/1000,color='black',label='Observed')
    axs[0].fill_between(weeks,lo95,hi95,alpha=.17,color='steelblue',label='95% pointwise PI')
    axs[0].fill_between(weeks,lo80,hi80,alpha=.35,color='steelblue',label='80% pointwise PI')
    axs[0].plot(np.r_[120,weeks],np.r_[result['last_observed_rolls']/1000,point],color='steelblue',marker='.',label='Forecast')
    axs[0].axvline(120,color='gray',linestyle=':')
    axs[0].set(xlabel='Week (calendar dates unspecified)',ylabel='Weekly sales (thousands of rolls)',title='Paper-towel sales: next 12 weeks')
    axs[0].legend(fontsize=8)
    axs[1].plot(np.arange(1,13),[r['se_rolls']/1000 for r in rows],marker='o')
    axs[1].set(xlabel='Forecast horizon h (weeks)',ylabel='Forecast-error SD (thousands of rolls)',title='Accumulating level uncertainty')
    for ax in axs: ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig('turn10_forecast.png',dpi=170)
    plt.close(fig)
    Path('turn10_forecast.json').write_text(json.dumps(result,indent=2))
    print('theta',theta,'sigma2',variance,'last sales',result['last_observed_rolls'],'last innovation',result['last_filtered_innovation'])
    print('converged',result['converged'],'warnings',result['warnings'])
    for r in rows:
        if r['horizon'] in [1,2,4,8,12]:
            print(r)


if __name__ == '__main__':
    main()
