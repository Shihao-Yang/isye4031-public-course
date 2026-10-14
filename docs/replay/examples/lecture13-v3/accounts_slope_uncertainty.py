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

df=pd.read_csv('sales_territory.csv')
full=sm.OLS(df.Sales,sm.add_constant(df.drop(columns='Sales'))).fit()
simple=sm.OLS(df.Sales,sm.add_constant(df[['Accts']])).fit()
z=sm.add_constant(df.drop(columns=['Sales','Accts']))
a=sm.OLS(df.Accts,z).fit()
y=sm.OLS(df.Sales,z).fit()
partial=sm.OLS(y.resid,a.resid).fit()
def extract(m):
    ci=m.conf_int().loc['Accts']
    return {'slope':float(m.params.Accts),'standard_error':float(m.bse.Accts),'ci95':[float(ci.iloc[0]),float(ci.iloc[1])],'residual_df':float(m.df_resid),'residual_sd':float(np.sqrt(m.mse_resid))}
remaining=float(np.sum(a.resid**2)); total=float(np.sum((df.Accts-df.Accts.mean())**2))
se_ref=float(np.sqrt(full.mse_resid/total))
r={'simple_sales_on_accounts':extract(simple),'full_eight_predictor_sales_model':extract(full),'accounts_auxiliary_R_squared':float(a.rsquared),'accounts_residual_sum_squares':remaining,'accounts_total_sum_squares':total,'VIF':total/remaining,'standard_error_inflation_factor_holding_full_model_error_variance_fixed':float(np.sqrt(total/remaining)),'reference_standard_error_using_total_accounts_variation_and_full_model_error_variance':se_ref,'partial_regression_slope':float(partial.params.iloc[0]),'note':'Classical OLS standard errors and t intervals. Reference SE is an algebraic comparison holding full-model residual variance fixed, not a fitted alternative model.'}
with open('accounts_slope_uncertainty_results.json','w') as f:json.dump(r,f,indent=2)
pd.DataFrame({'row':np.arange(1,26),'Accounts_residual':a.resid,'Sales_residual':y.resid}).to_csv('accounts_sales_partial_residuals.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(11,4.8))
axes[0].scatter(a.resid,y.resid,color='#326a9f')
xx=np.linspace(a.resid.min(),a.resid.max(),100)
axes[0].plot(xx,full.params.Accts*xx,color='#bc402b',label=f'Adjusted slope = {full.params.Accts:.2f}')
axes[0].axhline(0,color='gray',lw=.6);axes[0].axvline(0,color='gray',lw=.6)
axes[0].set_xlabel('Accounts after removing other seven predictors')
axes[0].set_ylabel('Sales after removing other seven predictors\n(hundreds of units)')
axes[0].set_title('Partial regression: all 25 observations');axes[0].legend()
for i,(label,m) in enumerate([('Accounts only',simple),('All eight predictors',full)]):
    ci=m.conf_int().loc['Accts'];b=m.params.Accts
    axes[1].errorbar(b,i,xerr=[[b-ci.iloc[0]],[ci.iloc[1]-b]],fmt='o',capsize=6,color='#326a9f')
axes[1].set_yticks([0,1],['Accounts only','All eight predictors'])
axes[1].set_ylim(-.6,1.6)
axes[1].axvline(0,color='gray',linestyle='--')
axes[1].set_xlabel('Accounts slope (hundreds of sales units\nper unit of the Accounts measure)')
axes[1].set_title('Slope estimates and classical 95% intervals')
fig.tight_layout();fig.savefig('accounts_slope_uncertainty.png',dpi=180)
print(json.dumps(r,indent=2))
