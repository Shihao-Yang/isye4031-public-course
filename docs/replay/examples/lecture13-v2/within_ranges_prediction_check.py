import itertools
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

df=pd.read_csv('sales_territory.csv');x=df.drop(columns='Sales')
# Constructed experiment: enumerate range-box corners, not actual territories.
means=x.mean();sd=x.std(ddof=1)
z=sm.add_constant((x-means)/sd)
fit=sm.OLS(df.Sales,z).fit()
corners=np.array(list(itertools.product(*[(x[c].min(),x[c].max()) for c in x.columns])))
design=np.column_stack([np.ones(len(corners)),(corners-means.values)/sd.values])
inverse=np.linalg.inv(z.values.T@z.values)
h=np.einsum('ij,jk,ik->i',design,inverse,design)
i=int(np.argmax(h));v=corners[i]
pred=fit.get_prediction(design[i:i+1]).summary_frame(alpha=.05).iloc[0]
observed=fit.get_influence().hat_matrix_diag
out={'model':'OLS Sales on intercept and all eight predictors, using centered standardized columns for numerically stable calculation','experiment':'Constructed range-box corner selected for maximum leverage among all 256 corners; not an observed or proposed real territory','constructed_predictors':dict(zip(x.columns,map(float,v))),'all_predictors_within_observed_ranges':bool(np.all(v>=x.min().values) and np.all(v<=x.max().values)),'constructed_leverage':float(h[i]),'leverage_at_predictor_means':1/len(df),'maximum_observed_leverage':float(observed.max()),'constructed_prediction':{k:float(val) for k,val in pred.items()},'observed_Time_Accts_correlation':float(x.Time.corr(x.Accts))}
with open('within_ranges_prediction_results.json','w') as f:json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
