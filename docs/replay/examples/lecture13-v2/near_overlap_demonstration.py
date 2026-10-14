import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
rng=np.random.default_rng(403113)
df=pd.read_csv('sales_territory.csv')
x=sm.add_constant(df.drop(columns='Sales'))
base=sm.OLS(df.Sales,x).fit()
noise=rng.normal(size=len(df))
noise=(noise-noise.mean())/noise.std(ddof=1)
rows=[]
for fraction in [0.1,0.01]:
 z=x.copy()
 z['Accts_near_copy_constructed']=df.Accts+fraction*df.Accts.std(ddof=1)*noise
 fit=sm.OLS(df.Sales,z).fit()
 c=np.zeros(z.shape[1]);c[z.columns.get_loc('Accts')]=1;c[-1]=1
 rows.append({'constructed_noise_sd_fraction_of_Accts_sd':fraction,'pair_correlation':float(z.Accts.corr(z.iloc[:,-1])),'Accts_vif':float(variance_inflation_factor(z.values,z.columns.get_loc('Accts'))),'Accts_coefficient':float(fit.params.Accts),'Accts_se':float(fit.bse.Accts),'near_copy_coefficient':float(fit.params.iloc[-1]),'near_copy_se':float(fit.bse.iloc[-1]),'coefficient_sum':float(c@fit.params),'coefficient_sum_se':float(np.sqrt(c@fit.cov_params()@c)),'training_r_squared':float(fit.rsquared),'design_rank':int(np.linalg.matrix_rank(z)),'design_columns':z.shape[1]})
result={'experiment':'Constructed near copies, using one fixed standardized random noise vector at two scales. Original data unchanged; all 25 observations retained.','seed':403113,'original_full_model_Accts_coefficient':float(base.params.Accts),'original_full_model_Accts_se':float(base.bse.Accts),'original_full_model_r_squared':float(base.rsquared),'constructed_models':rows}
with open('near_overlap_demonstration_results.json','w') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
