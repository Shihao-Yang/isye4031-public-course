import os
os.environ['MPLCONFIGDIR']='.mplcache'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import json

df=pd.read_csv('sales_territory.csv');x=df.drop(columns='Sales')
full=sm.OLS(df.Sales,sm.add_constant(x)).fit()
reduced=sm.OLS(df.Sales,sm.add_constant(x.drop(columns='Accts'))).fit()
rows=[]
for name in reduced.params.index:
    rows.append({'term':name,'full_coefficient':float(full.params[name]),'without_Accounts_coefficient':float(reduced.params[name]),'full_SE':float(full.bse[name]),'without_Accounts_SE':float(reduced.bse[name])})
pd.DataFrame(rows).to_csv('drop_accounts_coefficient_comparison.csv',index=False)
difference=reduced.fittedvalues-full.fittedvalues
pd.DataFrame({'row':np.arange(1,26),'observed_Sales':df.Sales,'full_fitted_Sales':full.fittedvalues,'without_Accounts_fitted_Sales':reduced.fittedvalues,'change_without_Accounts':difference}).to_csv('drop_accounts_fitted_changes.csv',index=False)
f,p,dfdiff=full.compare_f_test(reduced)
def metrics(m):return {'R_squared':float(m.rsquared),'adjusted_R_squared':float(m.rsquared_adj),'SSE':float(m.ssr),'residual_SD':float(np.sqrt(m.mse_resid)),'residual_df':float(m.df_resid)}
r={'full_eight_predictor_model':metrics(full),'seven_predictor_model_without_Accounts':metrics(reduced),'partial_F_test':{'F':float(f),'pvalue':float(p),'df_numerator':float(dfdiff),'df_denominator':float(full.df_resid)},'RMS_fitted_change':float(np.sqrt(np.mean(difference**2))),'max_absolute_fitted_change':float(np.max(np.abs(difference))),'row_of_max_absolute_change':int(np.argmax(np.abs(difference))+1),'coefficient_comparison':rows,'note':'Actual data, all 25 rows. In-sample nested model comparison, not validation. Classical OLS partial F test.'}
with open('drop_accounts_comparison_results.json','w') as file:json.dump(r,file,indent=2)
fig,axes=plt.subplots(1,2,figsize=(11,4.8))
axes[0].scatter(full.fittedvalues,reduced.fittedvalues,color='#326a9f')
lo=min(full.fittedvalues.min(),reduced.fittedvalues.min());hi=max(full.fittedvalues.max(),reduced.fittedvalues.max())
axes[0].plot([lo,hi],[lo,hi],linestyle='--',color='gray');axes[0].set_xlabel('Fitted Sales: full eight-predictor model');axes[0].set_ylabel('Fitted Sales: Accounts omitted');axes[0].set_title('Fitted sales with and without Accounts')
axes[1].bar(np.arange(1,26),difference,color=['#326a9f' if v>=0 else '#bc402b' for v in difference]);axes[1].axhline(0,color='black',lw=.7);axes[1].set_xlabel('Observation (CSV row order)');axes[1].set_ylabel('Change in fitted Sales\n(hundreds of units)');axes[1].set_title('Without Accounts minus full model')
fig.tight_layout();fig.savefig('drop_accounts_comparison.png',dpi=180)
print(json.dumps(r,indent=2))
