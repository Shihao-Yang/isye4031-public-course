import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

df=pd.read_csv('sales_territory.csv')
x=sm.add_constant(df.drop(columns='Sales'))
original=sm.OLS(df.Sales,x).fit()
duplicate=x.copy()
duplicate['Accts_copy_constructed']=df.Accts
fit=sm.OLS(df.Sales,duplicate).fit()
alternative=fit.params.copy()
alternative['Accts']+=10
alternative['Accts_copy_constructed']-=10
results={'experiment':'Constructed exact duplicate of Accts; all 25 original observations retained','original_design_columns':x.shape[1],'original_design_rank':int(np.linalg.matrix_rank(x)),'duplicated_design_columns':duplicate.shape[1],'duplicated_design_rank':int(np.linalg.matrix_rank(duplicate)),'original_Accts_coefficient':float(original.params.Accts),'pseudoinverse_Accts_coefficient':float(fit.params.Accts),'pseudoinverse_duplicate_coefficient':float(fit.params.Accts_copy_constructed),'alternative_Accts_coefficient':float(alternative.Accts),'alternative_duplicate_coefficient':float(alternative.Accts_copy_constructed),'max_absolute_fitted_difference_original_vs_duplicate':float(np.max(np.abs(original.fittedvalues-fit.fittedvalues))),'max_absolute_fitted_difference_after_coefficient_shift':float(np.max(np.abs(duplicate@alternative-fit.fittedvalues)))}
with open('exact_overlap_demonstration_results.json','w') as f:json.dump(results,f,indent=2)
print(json.dumps(results,indent=2))
