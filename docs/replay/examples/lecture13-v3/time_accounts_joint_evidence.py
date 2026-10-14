import os
os.environ['MPLCONFIGDIR']='.mplcache'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f
import json

df=pd.read_csv('sales_territory.csv');x=df.drop(columns='Sales')
full=sm.OLS(df.Sales,sm.add_constant(x)).fit()
reduced=sm.OLS(df.Sales,sm.add_constant(x.drop(columns=['Time','Accts']))).fit()
F,p,q=full.compare_f_test(reduced)
names=['Time','Accts'];b=full.params[names].to_numpy();cov=full.cov_params().loc[names,names].to_numpy()
vals,vec=np.linalg.eigh(cov)
theta=np.linspace(0,2*np.pi,400)
ellipse=b[:,None]+vec@np.diag(np.sqrt(vals*2*f.ppf(.95,2,full.df_resid)))@np.vstack([np.cos(theta),np.sin(theta)])
pd.DataFrame({'Time_coefficient':ellipse[0],'Accounts_coefficient':ellipse[1]}).to_csv('time_accounts_joint_confidence_region.csv',index=False)
ind={name:{'coefficient':float(full.params[name]),'SE':float(full.bse[name]),'pvalue':float(full.pvalues[name])} for name in names}
r={'hypothesis':'Time and Accts coefficients both zero, conditional on the other six predictors','n':len(df),'full_SSE':float(full.ssr),'reduced_SSE':float(reduced.ssr),'SSE_reduction':float(reduced.ssr-full.ssr),'partial_R_squared':float((reduced.ssr-full.ssr)/reduced.ssr),'F':float(F),'numerator_df':int(q),'denominator_df':int(full.df_resid),'pvalue':float(p),'individual_tests_in_full_model':ind,'coefficient_covariance':cov.tolist(),'note':'Classical OLS joint F test and 95% joint confidence ellipse under independent normal errors with constant variance. Actual data; all rows retained. Not validation or a causal test.'}
with open('time_accounts_joint_evidence_results.json','w') as file:json.dump(r,file,indent=2)
fig,ax=plt.subplots(figsize=(7.5,5.5))
ax.fill(ellipse[0],ellipse[1],alpha=.16,color='#326a9f');ax.plot(ellipse[0],ellipse[1],color='#326a9f',label='95% joint confidence region')
ax.scatter(*b,color='#326a9f',label='Full-model estimates',zorder=3)
ax.scatter(0,0,color='#bc402b',marker='x',s=90,label='Joint null: both slopes zero',zorder=4)
ax.axhline(0,color='gray',lw=.7);ax.axvline(0,color='gray',lw=.7)
ax.set_xlabel('Time coefficient (hundreds of sales units per month)');ax.set_ylabel('Accounts coefficient\n(hundreds of sales units per Accounts unit)')
ax.set_title(f'Time and Accounts: joint evidence in the full model\nF(2, 16) = {F:.2f}, p = {p:.4f}')
ax.legend(fontsize=9);fig.tight_layout();fig.savefig('time_accounts_joint_evidence.png',dpi=180)
print(json.dumps(r,indent=2))
