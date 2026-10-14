import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

df=pd.read_csv('sales_territory.csv')
x=df.drop(columns='Sales')
a=sm.add_constant(x)
b=sm.add_constant(x-x.mean())
m1=sm.OLS(df.Sales,a).fit()
m2=sm.OLS(df.Sales,b).fit()
v1={c:float(variance_inflation_factor(a.values,i)) for i,c in enumerate(a.columns) if c!='const'}
v2={c:float(variance_inflation_factor(b.values,i)) for i,c in enumerate(b.columns) if c!='const'}
r={'model':'Original full eight-predictor OLS with intercept, compared with same model after subtracting each predictor sample mean; Sales not centered','n':len(df),'original_intercept':float(m1.params['const']),'centered_intercept':float(m2.params['const']),'mean_sales':float(df.Sales.mean()),'original_Accts_slope':float(m1.params.Accts),'centered_Accts_slope':float(m2.params.Accts),'original_Accts_se':float(m1.bse.Accts),'centered_Accts_se':float(m2.bse.Accts),'original_vifs':v1,'centered_vifs':v2,'max_abs_slope_difference':float(np.max(np.abs(m1.params.iloc[1:]-m2.params.iloc[1:]))),'max_abs_fitted_difference':float(np.max(np.abs(m1.fittedvalues-m2.fittedvalues))),'original_design_condition_number':float(np.linalg.cond(a)),'centered_design_condition_number':float(np.linalg.cond(b))}
with open('centering_overlap_results.json','w') as f:json.dump(r,f,indent=2)
print(json.dumps(r,indent=2))
