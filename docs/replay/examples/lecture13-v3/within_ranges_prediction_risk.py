import os
os.environ['MPLCONFIGDIR']='.mplcache'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import json
from itertools import product

df=pd.read_csv('sales_territory.csv');x=df.drop(columns='Sales');X=sm.add_constant(x)
m=sm.OLS(df.Sales,X).fit()
# Constructed candidates use 5% or 95% of each observed range, strictly inside.
low=x.min()+.05*(x.max()-x.min());high=x.min()+.95*(x.max()-x.min())
candidates=np.array([[low.iloc[j] if bit==0 else high.iloc[j] for j,bit in enumerate(bits)] for bits in product([0,1],repeat=8)])
Z=np.column_stack([np.ones(len(candidates)),candidates]);inv=m.normalized_cov_params.to_numpy()
h=np.einsum('ij,jk,ik->i',Z,inv,Z);idx=int(np.argmax(h));risk=candidates[idx]
points=pd.DataFrame([x.mean().to_numpy(),risk],columns=x.columns,index=['Constructed mean territory','Constructed unusual combination'])
P=sm.add_constant(points,has_constant='add');pred=m.get_prediction(P).summary_frame(alpha=.05)
lev=np.einsum('ij,jk,ik->i',P.to_numpy(),inv,P.to_numpy())
rows=[]
for i,label in enumerate(points.index):
    rows.append({'territory':label,'predictors':points.iloc[i].to_dict(),'leverage':float(lev[i]),**pred.iloc[i].to_dict()})
points.to_csv('constructed_within_ranges_territories.csv')
r={'model':'Original full eight-predictor OLS model, all 25 actual observations','construction':'Choose highest-leverage combination among 256 candidates at 5% or 95% of each observed predictor range. Comparison point is vector of predictor means. Both territories are hypothetical.','observed_ranges':{'min':x.min().to_dict(),'max':x.max().to_dict()},'observed_leverage_max':float(m.get_influence().hat_matrix_diag.max()),'territories':rows,'note':'Model-based classical 95% intervals, not validation; prediction interval concerns one new observation under the same model and error assumptions.'}
with open('within_ranges_prediction_risk_results.json','w') as f:json.dump(r,f,indent=2)
fig,axes=plt.subplots(1,2,figsize=(12,5))
axes[0].scatter(x.Time,x.Accts,color='#326a9f',label='Actual observations')
axes[0].scatter(risk[0],risk[list(x.columns).index('Accts')],marker='X',s=110,color='#bc402b',label='Constructed unusual territory')
axes[0].scatter(x.Time.mean(),x.Accts.mean(),marker='D',s=65,color='#4a8751',label='Constructed mean territory')
axes[0].set_xlim(x.Time.min(),x.Time.max());axes[0].set_ylim(x.Accts.min(),x.Accts.max());axes[0].set_xlabel('Time (months)');axes[0].set_ylabel('Accounts measure');axes[0].set_title('Inside each range, unusual jointly\nTwo of eight predictors shown');axes[0].legend(fontsize=8)
for i,row in enumerate(rows):
    b=row['mean'];axes[1].plot([row['obs_ci_lower'],row['obs_ci_upper']],[i,i],color='gray',lw=3,label='95% prediction interval' if i==0 else None)
    axes[1].plot([row['mean_ci_lower'],row['mean_ci_upper']],[i,i],color='#326a9f',lw=7,label='95% mean-response interval' if i==0 else None);axes[1].scatter(b,i,color='black',zorder=3)
axes[1].set_yticks([0,1],['Constructed mean','Constructed unusual']);axes[1].set_ylim(-.6,1.6);axes[1].set_xlabel('Predicted Sales (hundreds of units)');axes[1].set_title('Uncertainty in the full eight-predictor model');axes[1].legend(fontsize=8)
fig.tight_layout();fig.savefig('within_ranges_prediction_risk.png',dpi=180)
print(json.dumps(r,indent=2))
