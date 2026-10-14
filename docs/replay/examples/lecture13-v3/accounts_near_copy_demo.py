import os
os.environ['MPLCONFIGDIR']='.mplcache'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import json
rng=np.random.default_rng(403114)
df=pd.read_csv('sales_territory.csv');x=sm.add_constant(df.drop(columns='Sales'))
base=sm.OLS(df.Sales,x).fit()
noise=rng.normal(size=len(df));copy=df.Accts+0.1*noise
xx=x.copy();xx['Accounts_near_copy_constructed']=copy
fit=sm.OLS(df.Sales,xx).fit()
c=df.copy();c['Accounts_near_copy_constructed']=copy;c.to_csv('constructed_accounts_near_copy_data.csv',index=False)
trials=[]
for k in range(200):
    xp=x.copy();xp['Accounts_near_copy_constructed']=copy+rng.normal(0,.01,len(df))
    m=sm.OLS(df.Sales,xp).fit()
    trials.append({'trial':k+1,'Accounts_slope':m.params.Accts,'copy_slope':m.params.Accounts_near_copy_constructed,'sum':m.params.Accts+m.params.Accounts_near_copy_constructed,'fitted_sales_RMS_change_from_initial_near_copy_fit':np.sqrt(np.mean((m.fittedvalues-fit.fittedvalues)**2))})
tr=pd.DataFrame(trials);tr.to_csv('accounts_near_copy_sensitivity_trials.csv',index=False)
contrast=np.zeros(len(fit.params));contrast[list(fit.params.index).index('Accts')]=1;contrast[-1]=1
combined=fit.t_test(contrast)
r={'seed':403114,'construction':'Copy = actual Accounts + independent Normal(0, 0.1) noise; sensitivity trials add Normal(0, 0.01) noise to this copy only; Sales and all original predictors unchanged','n':len(df),'correlation_accounts_copy':float(np.corrcoef(df.Accts,copy)[0,1]),'design_rank':int(np.linalg.matrix_rank(xx)),'design_columns':xx.shape[1],'Accounts_slope':float(fit.params.Accts),'copy_slope':float(fit.params.iloc[-1]),'Accounts_SE':float(fit.bse.Accts),'copy_SE':float(fit.bse.iloc[-1]),'combined_slope':float(combined.effect.item()),'combined_SE':float(combined.sd.item()),'combined_ci95':combined.conf_int().tolist()[0],'original_accounts_slope':float(base.params.Accts),'original_accounts_SE':float(base.bse.Accts),'RMS_fitted_difference_from_original_model':float(np.sqrt(np.mean((fit.fittedvalues-base.fittedvalues)**2))),'sensitivity_summary':tr.drop(columns='trial').agg(['min','max','median']).to_dict(),'note':'Constructed sensitivity analysis, not validation. Classical OLS uncertainty.'}
with open('accounts_near_copy_demo_results.json','w') as f:json.dump(r,f,indent=2)
fig,axes=plt.subplots(1,2,figsize=(11,4.8))
axes[0].scatter(tr.Accounts_slope,tr.copy_slope,s=15,alpha=.6)
axes[0].set_xlabel('Accounts coefficient');axes[0].set_ylabel('Constructed copy coefficient')
axes[0].set_title('200 tiny perturbations of the constructed copy')
axes[1].scatter(base.fittedvalues,fit.fittedvalues,color='#326a9f')
lo=base.fittedvalues.min();hi=base.fittedvalues.max();axes[1].plot([lo,hi],[lo,hi],color='gray',linestyle='--')
axes[1].set_xlabel('Fitted Sales: original eight-predictor model');axes[1].set_ylabel('Fitted Sales: model with near copy')
axes[1].set_title('Fitted sales remain close, but are not identical')
fig.suptitle('Constructed near-copy experiment using all 25 actual observations',fontsize=14)
fig.tight_layout();fig.savefig('accounts_near_copy_demo.png',dpi=180)
print(json.dumps(r,indent=2))
