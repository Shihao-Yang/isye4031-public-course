import os
os.environ['MPLCONFIGDIR']='.mplcache'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import json
from scipy.stats import t
actual=pd.read_csv('sales_territory.csv')
constructed=pd.read_csv('constructed_accounts_near_copy_data.csv')
nx=constructed.drop(columns='Sales')
nraw=sm.OLS(actual.Sales,sm.add_constant(nx)).fit()
ncenter=sm.OLS(actual.Sales,sm.add_constant(nx-nx.mean())).fit()
x=actual.drop(columns='Sales');base=sm.OLS(actual.Sales,sm.add_constant(x)).fit()
raw=x.copy();raw['Accounts_x_Time_constructed']=x.Accts*x.Time
mr=sm.OLS(actual.Sales,sm.add_constant(raw)).fit()
c=x-x.mean();c['Accounts_x_Time_centered_constructed']=c.Accts*c.Time
mc=sm.OLS(actual.Sales,sm.add_constant(c)).fit()
ci=mc.conf_int().iloc[-1].tolist()
values=np.linspace(x.Time.min(),x.Time.max(),150)
b=[];lo=[];hi=[]
for val in values:
    contrast=np.zeros(len(mc.params));contrast[list(mc.params.index).index('Accts')]=1;contrast[-1]=val-x.Time.mean()
    estimate=float(contrast@mc.params);se=float(np.sqrt(contrast@mc.cov_params()@contrast));margin=t.ppf(.975,mc.df_resid)*se
    b.append(estimate);lo.append(estimate-margin);hi.append(estimate+margin)
pd.DataFrame({'Time':values,'conditional_Accounts_slope':b,'pointwise_ci95_low':lo,'pointwise_ci95_high':hi}).to_csv('accounts_time_interaction_slopes.csv',index=False)
r={'near_copy_centering':{'raw_Accounts_slope':float(nraw.params.Accts),'centered_Accounts_slope':float(ncenter.params.Accts),'raw_Accounts_SE':float(nraw.bse.Accts),'centered_Accounts_SE':float(ncenter.bse.Accts),'max_fitted_change':float(np.max(np.abs(nraw.fittedvalues-ncenter.fittedvalues)))},'actual_data_interaction':{'mean_Time':float(x.Time.mean()),'interaction_coefficient':float(mc.params.iloc[-1]),'interaction_SE':float(mc.bse.iloc[-1]),'interaction_ci95':ci,'interaction_pvalue':float(mc.pvalues.iloc[-1]),'raw_Accounts_coefficient_at_Time_zero':float(mr.params.Accts),'centered_Accounts_coefficient_at_mean_Time':float(mc.params.Accts),'centered_Accounts_SE':float(mc.bse.Accts),'max_fitted_difference_raw_vs_centered_interaction':float(np.max(np.abs(mr.fittedvalues-mc.fittedvalues))),'base_residual_sd':float(np.sqrt(base.mse_resid)),'interaction_residual_sd':float(np.sqrt(mc.mse_resid)),'residual_df':float(mc.df_resid)},'note':'All 25 observations retained. Interaction is a constructed product of actual predictors; no near copy is included in interaction models. Classical pointwise t intervals; exploratory in-sample analysis, not validation.'}
with open('centering_and_interaction_results.json','w') as f:json.dump(r,f,indent=2)
fig,axes=plt.subplots(1,2,figsize=(11,4.8))
for i,m in enumerate([nraw,ncenter]):
    interval=m.conf_int().loc['Accts'];bb=m.params.Accts
    axes[0].errorbar(bb,i,xerr=[[bb-interval.iloc[0]],[interval.iloc[1]-bb]],fmt='o',capsize=5)
axes[0].set_yticks([0,1],['Near-copy model','Same model, centered']);axes[0].set_ylim(-.6,1.6);axes[0].set_xlabel('Accounts coefficient and classical 95% interval');axes[0].set_title('Centering does not repair near duplication')
axes[1].plot(values,b,label='Accounts × Time model');axes[1].fill_between(values,lo,hi,alpha=.2,label='Pointwise 95% interval')
axes[1].axhline(base.params.Accts,color='gray',linestyle='--',label='Original additive model')
axes[1].axhline(0,color='black',lw=.6);axes[1].plot(x.Time,np.full(len(x),min(lo)), '|',color='black')
axes[1].set_xlabel('Time with company (months)');axes[1].set_ylabel('Conditional Accounts slope\n(hundreds of sales units per Accounts unit)');axes[1].set_title('Interaction changes the statistical question');axes[1].legend(fontsize=8)
fig.tight_layout();fig.savefig('centering_and_interaction_comparison.png',dpi=180)
print(json.dumps(r,indent=2))
