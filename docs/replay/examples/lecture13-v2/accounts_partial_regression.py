import os
os.environ['MPLCONFIGDIR'] = '.mplcache'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import json

df=pd.read_csv('sales_territory.csv')
others=df.drop(columns=['Sales','Accts'])
controls=sm.add_constant(others)
afit=sm.OLS(df.Accts,controls).fit()
yfit=sm.OLS(df.Sales,controls).fit()
a=afit.resid
y=yfit.resid
full=sm.OLS(df.Sales,sm.add_constant(df.drop(columns='Sales'))).fit()
ci=full.conf_int().loc['Accts'].tolist()
results={'model':'OLS Sales on intercept and all eight original predictors; partial regression adjusts both Sales and Accts for the other seven predictors','accounts_total_centered_sum_squares':float(((df.Accts-df.Accts.mean())**2).sum()),'accounts_residual_sum_squares':float((a*a).sum()),'accounts_variation_fraction_remaining':float(1-afit.rsquared),'accounts_original_sample_sd':float(df.Accts.std()),'accounts_residual_rms_using_n_minus_1':float(np.sqrt((a*a).sum()/(len(df)-1))),'accounts_coefficient':float(full.params.Accts),'accounts_coefficient_se':float(full.bse.Accts),'accounts_coefficient_95ci':ci,'accounts_p_value':float(full.pvalues.Accts),'partial_correlation':float(np.corrcoef(a,y)[0,1]),'partial_r_squared':float(np.corrcoef(a,y)[0,1]**2),'full_model_residual_df':float(full.df_resid)}
with open('accounts_partial_regression_results.json','w') as f:json.dump(results,f,indent=2)
pd.DataFrame({'row':np.arange(1,len(df)+1),'Accts_observed':df.Accts,'Accts_predicted_by_other_seven':afit.fittedvalues,'Accts_residual':a,'Sales_residual':y}).to_csv('accounts_partial_regression_residuals.csv',index=False)
fig,ax=plt.subplots(figsize=(7,4.8))
ax.scatter(a,y,color='tab:blue')
x=np.array([a.min(),a.max()])
ax.plot(x,full.params.Accts*x,color='tab:orange',label=f'Full-model Accts slope = {full.params.Accts:.2f}')
ax.axhline(0,color='gray',lw=.7)
ax.axvline(0,color='gray',lw=.7)
ax.set_xlabel('Accts remaining after adjustment for other seven predictors')
ax.set_ylabel('Sales remaining after the same adjustment\n(hundreds of units)')
ax.set_title('Accounts added-variable plot: all 25 observations')
ax.legend()
fig.tight_layout()
fig.savefig('accounts_added_variable_plot.png',dpi=170)
print(json.dumps(results,indent=2))
