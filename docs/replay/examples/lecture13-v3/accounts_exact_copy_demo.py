import os
os.environ['MPLCONFIGDIR']='.mplcache'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import json

df=pd.read_csv('sales_territory.csv')
x=sm.add_constant(df.drop(columns='Sales'))
base=sm.OLS(df.Sales,x).fit()
xc=x.copy();xc['Accounts_exact_copy_constructed']=df.Accts
copy=sm.OLS(df.Sales,xc).fit()
constructed=df.copy();constructed['Accounts_exact_copy_constructed']=df.Accts
constructed.to_csv('sales_with_constructed_accounts_exact_copy.csv',index=False)
variants=[]
for shift in [-10,0,10]:
    b=copy.params.copy();b['Accts']+=shift;b['Accounts_exact_copy_constructed']-=shift
    variants.append({'Accounts_coefficient':float(b.Accts),'copy_coefficient':float(b.Accounts_exact_copy_constructed),'sum':float(b.Accts+b.Accounts_exact_copy_constructed),'max_fitted_difference_from_original':float(np.max(np.abs(xc@b-base.fittedvalues)))})
r={'original_accounts_slope':float(base.params.Accts),'original_design_columns':x.shape[1],'original_design_rank':int(np.linalg.matrix_rank(x)),'duplicate_design_columns':xc.shape[1],'duplicate_design_rank':int(np.linalg.matrix_rank(xc)),'software_pseudoinverse_accounts_coefficient':float(copy.params.Accts),'software_pseudoinverse_copy_coefficient':float(copy.params.Accounts_exact_copy_constructed),'max_fitted_difference':float(np.max(np.abs(copy.fittedvalues-base.fittedvalues))),'equivalent_coefficient_choices':variants,'combined_accounts_slope_standard_error':float(base.bse.Accts),'combined_accounts_slope_ci95':base.conf_int().loc['Accts'].tolist()}
with open('accounts_exact_copy_demo_results.json','w') as f:json.dump(r,f,indent=2)
fig,axes=plt.subplots(1,2,figsize=(11,4.7))
axes[0].scatter(base.fittedvalues,copy.fittedvalues,color='#326a9f')
lo=base.fittedvalues.min();hi=base.fittedvalues.max()
axes[0].plot([lo,hi],[lo,hi],color='gray',linestyle='--')
axes[0].set_xlabel('Fitted Sales: original eight-predictor model')
axes[0].set_ylabel('Fitted Sales: model with constructed exact copy')
axes[0].set_title('Fitted sales are unchanged (numerical precision)')
aa=np.linspace(-12,18,200)
axes[1].plot(aa,base.params.Accts-aa,color='#326a9f')
for v in variants:axes[1].scatter(v['Accounts_coefficient'],v['copy_coefficient'],color='#bc402b')
axes[1].set_xlabel('Accounts coefficient')
axes[1].set_ylabel('Constructed copy coefficient')
axes[1].set_title(f'Every point on this line gives the same fit\nCoefficients sum to {base.params.Accts:.3f}')
fig.suptitle('Constructed exact-copy experiment using all 25 actual observations',fontsize=14)
fig.tight_layout();fig.savefig('accounts_exact_copy_demo.png',dpi=180)
print(json.dumps(r,indent=2))
