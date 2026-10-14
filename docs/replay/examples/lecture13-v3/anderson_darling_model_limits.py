import os
os.environ['MPLCONFIGDIR']='.mplcache'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import normal_ad
from scipy.stats import probplot
import json
actual=pd.read_csv('sales_territory.csv');m=sm.OLS(actual.Sales,sm.add_constant(actual.drop(columns='Sales'))).fit()
examples=pd.read_csv('constructed_same_summary_datasets.csv')
sets=[('Actual sales: full model',m.resid.to_numpy())]
for name,g in examples.groupby('dataset',sort=False):
    fit=sm.OLS(g.constructed_y,sm.add_constant(g.constructed_x)).fit();sets.append(('Constructed: '+name,fit.resid.to_numpy()))
rows=[];fig,axes=plt.subplots(2,2,figsize=(11,8))
for ax,(name,res) in zip(axes.flat,sets):
    stat,p=normal_ad(res);rows.append({'dataset':name,'n':len(res),'AD_statistic':float(stat),'nominal_pvalue':float(p)})
    (qx,qy),(slope,intercept,r)=probplot(res)
    ax.scatter(qx,qy);ax.plot(qx,intercept+slope*qx,color='gray');ax.set_title(f'{name}\nAD = {stat:.3f}, nominal p = {p:.3g}',fontsize=11);ax.set_xlabel('Normal theoretical quantile');ax.set_ylabel('Fitted residual')
fig.suptitle('Anderson–Darling screens residual distributions, not model adequacy',fontsize=14);fig.tight_layout();fig.savefig('anderson_darling_model_limits.png',dpi=180)
r={'test':'Anderson–Darling normality screen, mean and variance estimated (statsmodels normal_ad)','results':rows,'qualification':'Nominal iid normality p-values applied to fitted residuals. Residuals are correlated and their variances depend on leverage; constructed residuals also satisfy enforced constraints. These p-values are descriptive screens, not exact regression adequacy tests. No fitted-design bootstrap calibration used.'}
with open('anderson_darling_model_limits_results.json','w') as f:json.dump(r,f,indent=2)
pd.DataFrame(rows).to_csv('anderson_darling_model_limits_values.csv',index=False)
print(json.dumps(r,indent=2))
