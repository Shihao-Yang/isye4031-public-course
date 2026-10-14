import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

df=pd.read_csv('sales_territory.csv')
x=df.drop(columns='Sales')
base=sm.OLS(df.Sales,sm.add_constant(x)).fit()
raw=x.copy();raw['Time_x_Accts']=x.Time*x.Accts
center=x-x.mean();center['Time_x_Accts']=center.Time*center.Accts
rdesign=sm.add_constant(raw);cdesign=sm.add_constant(center)
rfit=sm.OLS(df.Sales,rdesign).fit();cfit=sm.OLS(df.Sales,cdesign).fit()
vifs={label:{c:float(variance_inflation_factor(d.values,i)) for i,c in enumerate(d.columns) if c in ['Time','Accts','Time_x_Accts']} for label,d in [('raw',rdesign),('centered',cdesign)]}
f,p,dfdiff=cfit.compare_f_test(base)
out={'model':'Sales on all eight original main effects plus Time times Accts; all 25 observations retained','time_mean_months':float(x.Time.mean()),'Accts_mean':float(x.Accts.mean()),'interaction_coefficient':float(cfit.params.Time_x_Accts),'interaction_se':float(cfit.bse.Time_x_Accts),'interaction_95ci':cfit.conf_int().loc['Time_x_Accts'].tolist(),'interaction_p_value':float(cfit.pvalues.Time_x_Accts),'centered_Accts_coefficient_at_mean_Time':float(cfit.params.Accts),'centered_Accts_se':float(cfit.bse.Accts),'base_r_squared':float(base.rsquared),'interaction_r_squared':float(cfit.rsquared),'base_adjusted_r_squared':float(base.rsquared_adj),'interaction_adjusted_r_squared':float(cfit.rsquared_adj),'residual_df':float(cfit.df_resid),'nested_F':float(f),'nested_p':float(p),'vifs':vifs,'max_abs_fitted_difference_raw_vs_centered':float(np.max(np.abs(rfit.fittedvalues-cfit.fittedvalues)))}
with open('accounts_time_interaction_results.json','w') as f:json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
