from pathlib import Path
import os
os.environ['MPLCONFIGDIR'] = str(Path.cwd() / '.mplcache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

d=pd.read_csv('hotel_03_development_fits.csv')
assert d.t.max()==156
r=d['Trend + months: log_model_scale_residual'].to_numpy()
e=r-r.mean()
lags=np.arange(1,25)
acf=np.array([np.dot(e[k:],e[:-k])/np.dot(e,e) for k in lags])
reference=1.96/np.sqrt(len(e))
sign=np.sign(e)
changes=np.flatnonzero(sign[1:]!=sign[:-1])+1
run_lengths=np.diff(np.r_[0,changes,len(e)])
summary={'model':'Linear trend plus January-reference month indicators fitted to log occupancy',
 'residual_definition':'log(observed occupancy) minus fitted log occupancy',
 'development_months':len(e),'final_reserved_used':False,
 'residual_acf_lag1':float(acf[0]),'residual_acf_lag2':float(acf[1]),
 'residual_acf_lag12':float(acf[11]),
 'adjacent_residual_pearson_correlation':float(np.corrcoef(e[:-1],e[1:])[0,1]),
 'durbin_watson':float(np.sum(np.diff(e)**2)/np.sum(e**2)),
 'longest_same_sign_run_months':int(run_lengths.max()),
 'white_noise_reference_band':float(reference),
 'reference_band_caveat':'Approximate white-noise guide only; fitted seasonal regression residuals do not have exactly independent sampling errors. No formal significance claim.',
 'interpretation':'Strong residual persistence contradicts treating independent errors as an adequate working assumption; not a proof of one particular error model.'}
pd.DataFrame({'lag':lags,'acf':acf,'approx_reference_lower':-reference,
              'approx_reference_upper':reference}).to_csv('hotel_05_log_seasonal_residual_acf.csv',index=False)
Path('hotel_05_residual_diagnostic_results.json').write_text(json.dumps(summary,indent=2))
plt.rcParams.update({'font.size':11,'axes.titlesize':13,'axes.labelsize':11})
fig,axes=plt.subplots(1,3,figsize=(13,4.2))
axes[0].plot(d.t,e,color='#c45118',lw=1.6)
axes[0].axhline(0,color='gray',lw=1)
axes[0].set(title='Runs above and below zero',xlabel='Month index t',ylabel='Log-scale residual')
axes[1].scatter(e[:-1],e[1:],color='#2166ac',alpha=.75,s=24)
axes[1].axhline(0,color='gray',lw=.8)
axes[1].axvline(0,color='gray',lw=.8)
axes[1].set(title='Adjacent residuals move together',xlabel='Residual at month t−1',ylabel='Residual at month t')
axes[2].axhspan(-reference,reference,color='#999999',alpha=.18,label='Approximate white-noise guide')
axes[2].stem(lags,acf,basefmt=' ')
axes[2].axhline(0,color='gray',lw=.8)
axes[2].set(xticks=[1,6,12,18,24],title='Residual autocorrelation',xlabel='Lag (months)',ylabel='ACF')
axes[2].legend(loc='upper right',fontsize=8)
for ax in axes:
    ax.grid(alpha=.18)
    ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
fig.savefig('hotel_05_log_seasonal_residual_dependence.png',dpi=180)
print(json.dumps(summary,indent=2))
