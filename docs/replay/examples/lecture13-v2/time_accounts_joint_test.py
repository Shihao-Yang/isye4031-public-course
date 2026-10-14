import json
import pandas as pd
import statsmodels.api as sm

df=pd.read_csv('sales_territory.csv')
x=df.drop(columns='Sales')
full=sm.OLS(df.Sales,sm.add_constant(x)).fit()
reduced=sm.OLS(df.Sales,sm.add_constant(x.drop(columns=['Time','Accts']))).fit()
f,p,q=full.compare_f_test(reduced)
out={'comparison':'Additive OLS full eight-predictor model versus model omitting both Time and Accts; intercept and all 25 observations retained; other six predictors held in model','null_hypothesis':'beta_Time = beta_Accts = 0','F':float(f),'p_value':float(p),'numerator_df':float(q),'denominator_df':float(full.df_resid),'full_r_squared':float(full.rsquared),'reduced_r_squared':float(reduced.rsquared),'full_SSE':float(full.ssr),'reduced_SSE':float(reduced.ssr),'partial_R_squared_for_pair':float((reduced.ssr-full.ssr)/reduced.ssr),'individual_full_model_tests':{c:{'coefficient':float(full.params[c]),'p_value':float(full.pvalues[c])} for c in ['Time','Accts']}}
with open('time_accounts_joint_test_results.json','w') as file:json.dump(out,file,indent=2)
print(json.dumps(out,indent=2))
