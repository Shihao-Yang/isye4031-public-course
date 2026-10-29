from pathlib import Path
import os
os.environ['MPLCONFIGDIR'] = str(Path.cwd() / '.mplcache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

all_data = pd.read_csv('hotel_monthly.csv')
d = all_data.loc[all_data.t<=156].copy()
t = d.t.to_numpy()
y = d.rooms_per_day.to_numpy(dtype=float)
names = ['Constant','Linear','Quadratic']
origins = [96,108,120,132,144]
def design(times,degree):
    return np.vander(np.asarray(times)/12, N=degree+1, increasing=True)
rows, fold_rows, coefficient_rows = [], [], []
residuals = d.copy()
fits = {}
for degree,name in enumerate(names):
    b = np.linalg.lstsq(design(t,degree),y,rcond=None)[0]
    fitted = design(t,degree)@b
    fits[name] = fitted
    e = y-fitted
    residuals[name+'_fitted'] = fitted
    residuals[name+'_residual'] = e
    errors = []
    for origin in origins:
        tr = t<=origin
        va = (t>origin)&(t<=origin+12)
        inner_b = np.linalg.lstsq(design(t[tr],degree),y[tr],rcond=None)[0]
        error = y[va]-design(t[va],degree)@inner_b
        errors.extend(error)
        fold_rows.append({'model':name,'training_end':origin,'validation_start':origin+1,
                          'validation_end':origin+12,'rmse':float(np.sqrt(np.mean(error**2)))})
    rows.append({'model':name,'development_fit_rmse':float(np.sqrt(np.mean(e**2))),
                 'validation_rmse':float(np.sqrt(np.mean(np.array(errors)**2))),
                 'residual_lag1_correlation':float(np.corrcoef(e[:-1],e[1:])[0,1]),
                 'residual_lag12_correlation':float(np.corrcoef(e[:-12],e[12:])[0,1])})
    for power,value in enumerate(b):
        coefficient_rows.append({'model':name,'power_of_month_t':power,'coefficient':float(value/12**power)})
results = pd.DataFrame(rows)
results.to_csv('hotel_02_trend_comparison.csv',index=False)
pd.DataFrame(fold_rows).to_csv('hotel_02_expanding_validation.csv',index=False)
pd.DataFrame(coefficient_rows).to_csv('hotel_02_trend_coefficients.csv',index=False)
residuals.to_csv('hotel_02_development_fits_and_residuals.csv',index=False)
monthly = residuals.groupby('month')[[name+'_residual' for name in names]].mean()
monthly.to_csv('hotel_02_monthly_residual_means.csv')
e = residuals.Quadratic_residual.to_numpy()
acf = np.array([np.sum(e[k:]*e[:-k])/np.sum(e**2) for k in range(1,25)])
pd.DataFrame({'lag':np.arange(1,25),'quadratic_residual_acf':acf}).to_csv('hotel_02_quadratic_residual_acf.csv',index=False)
Path('hotel_02_analysis_scope.json').write_text(json.dumps({'development_t':[1,156],
 'final_holdout_t':[157,168],'final_holdout_used':False,'validation_origins':origins,
 'validation_horizon':12,'errors':'Actual minus fitted or forecast value',
 'acf_definition':'Sum of lagged residual products divided by sum of squared residuals, without bias adjustment',
 'scope':'Trend-only fits; residual summaries are development diagnostics, not forecast evaluation.'},indent=2))
plt.rcParams.update({'font.size':11,'axes.titlesize':13,'axes.labelsize':11})
fig,axes = plt.subplots(2,2,figsize=(11.5,8))
colors = ['#777777','#2166ac','#c45118']
axes[0,0].plot(t,y,color='#333333',alpha=.55,lw=1,label='Observed')
for name,color in zip(names,colors):
    axes[0,0].plot(t,fits[name],color=color,lw=2.2,label=name)
axes[0,0].set(title='Trend fits: development months only',xlabel='Month index t',ylabel='Average rooms occupied per day')
axes[0,0].legend(ncol=2,fontsize=9)
for name,color in zip(names[1:],colors[1:]):
    axes[0,1].plot(t,residuals[name+'_residual'],color=color,alpha=.8,lw=1.2,label=name)
axes[0,1].axhline(0,color='gray',lw=1)
axes[0,1].set(title='Residuals still rise and fall repeatedly',xlabel='Month index t',ylabel='Actual minus fitted (rooms/day)')
axes[0,1].legend(fontsize=9)
axes[1,0].bar(monthly.index,monthly.Quadratic_residual,color='#c45118')
axes[1,0].axhline(0,color='gray',lw=1)
axes[1,0].set(xticks=range(1,13),xticklabels=['J','F','M','A','M','J','J','A','S','O','N','D'],
              title='Quadratic residual means by calendar month',xlabel='Month of year',ylabel='Mean residual (rooms/day)')
axes[1,1].stem(np.arange(1,25),acf,basefmt=' ')
axes[1,1].axhline(0,color='gray',lw=1)
axes[1,1].set(xticks=[1,6,12,18,24],title='Quadratic residual autocorrelation',xlabel='Lag (months)',ylabel='Residual ACF')
for ax in axes.flat:
    ax.grid(alpha=.18)
    ax.spines[['top','right']].set_visible(False)
fig.tight_layout(pad=1.6)
fig.savefig('hotel_02_trends_and_unexplained_patterns.png',dpi=180)
print(results.to_string(index=False))
print(pd.DataFrame(coefficient_rows).to_string(index=False))
print(monthly.Quadratic_residual.to_string())
print('Quadratic residual ACF at lag 12:',acf[11])
