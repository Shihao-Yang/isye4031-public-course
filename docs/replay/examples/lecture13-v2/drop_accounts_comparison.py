import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

df=pd.read_csv('sales_territory.csv')
x=df.drop(columns='Sales')
full=sm.OLS(df.Sales,sm.add_constant(x)).fit()
z=sm.add_constant(x.drop(columns='Accts'))
reduced=sm.OLS(df.Sales,z).fit()
f,p,d=full.compare_f_test(reduced)
change=reduced.fittedvalues-full.fittedvalues
out={'comparison':'Additive OLS full eight-predictor model versus seven-predictor model omitting Accts, both with intercept and all 25 observations; no interaction','full':{'r_squared':float(full.rsquared),'adjusted_r_squared':float(full.rsquared_adj),'residual_sd':float(np.sqrt(full.mse_resid))},'without_Accts':{'r_squared':float(reduced.rsquared),'adjusted_r_squared':float(reduced.rsquared_adj),'residual_sd':float(np.sqrt(reduced.mse_resid))},'nested_test':{'F':float(f),'p':float(p),'numerator_df':float(d),'denominator_df':float(full.df_resid)},'fitted_sales_change':{'rms':float(np.sqrt(np.mean(change**2))),'max_absolute':float(np.max(np.abs(change)))},'coefficients':{c:{'full_estimate':float(full.params[c]),'full_se':float(full.bse[c]),'without_Accts_estimate':float(reduced.params[c]),'without_Accts_se':float(reduced.bse[c])} for c in z.columns},'without_Accts_vifs':{c:float(variance_inflation_factor(z.values,i)) for i,c in enumerate(z.columns) if c!='const'}}
with open('drop_accounts_comparison_results.json','w') as file:json.dump(out,file,indent=2)
pd.DataFrame({'row':np.arange(1,len(df)+1),'full_fitted_sales':full.fittedvalues,'without_Accts_fitted_sales':reduced.fittedvalues,'change':change}).to_csv('drop_accounts_fitted_sales.csv',index=False)
print(json.dumps(out,indent=2))
