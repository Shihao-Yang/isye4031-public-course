import os
os.environ['MPLCONFIGDIR'] = '.mplcache'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np
import json

df=pd.read_csv('sales_territory.csv')
other=df.drop(columns=['Sales','Accts'])
fit=sm.OLS(df.Accts,sm.add_constant(other)).fit()
res=fit.resid
out=pd.DataFrame({'row':np.arange(1,len(df)+1),'Accounts':df.Accts,'Accounts_fitted_from_other_seven':fit.fittedvalues,'Accounts_residual':res})
out.to_csv('accounts_after_adjustment_values.csv',index=False)
tss=float(np.sum((df.Accts-df.Accts.mean())**2))
rss=float(np.sum(res**2))
results={'model':'OLS Accts on intercept, Time, MktPoten, Adver, MktShare, Change, WkLoad, Rating','n':len(df),'r_squared':float(fit.rsquared),'remaining_variation_fraction':rss/tss,'VIF_Accounts':1/(1-fit.rsquared),'residual_min':float(res.min()),'residual_max':float(res.max()),'residual_sum':float(res.sum()),'coefficients':fit.params.to_dict()}
with open('accounts_after_adjustment_results.json','w') as f:json.dump(results,f,indent=2)
fig,axes=plt.subplots(1,2,figsize=(11,4.8))
axes[0].scatter(fit.fittedvalues,df.Accts,color='#326a9f')
lo=min(fit.fittedvalues.min(),df.Accts.min());hi=max(fit.fittedvalues.max(),df.Accts.max())
axes[0].plot([lo,hi],[lo,hi],color='gray',linestyle='--',label='Observed = fitted')
axes[0].set_xlabel('Accounts fitted from the other seven predictors')
axes[0].set_ylabel('Observed Accounts')
axes[0].set_title(f'Other predictors explain {fit.rsquared:.1%} of variation')
axes[0].legend()
axes[1].bar(out.row,res,color=['#326a9f' if v>=0 else '#bc402b' for v in res])
axes[1].axhline(0,color='black',lw=.8)
axes[1].set_xticks([1,5,10,15,20,25])
axes[1].set_xlabel('Observation (CSV row order)')
axes[1].set_ylabel('Accounts residual: observed minus fitted')
axes[1].set_title(f'{1-fit.rsquared:.1%} of variation remains after adjustment')
fig.suptitle('Accounts after linear adjustment: actual data, all 25 observations',fontsize=14)
fig.tight_layout()
fig.savefig('accounts_after_adjustment.png',dpi=180)
print(json.dumps(results,indent=2))
