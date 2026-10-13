import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(ROOT/'.mplcache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import json
import hashlib
path=ROOT/'sales_territory.csv'
before=hashlib.sha256(path.read_bytes()).hexdigest()
df=pd.read_csv(path)
base=sm.add_constant(df.drop(columns='Sales'),has_constant='add')
exact=base.copy(); exact['Accts_copy']=df.Accts
noise_sd=.01*df.Accts.std(ddof=1)
noise=np.random.default_rng(4031).normal(0,noise_sd,len(df))
near=base.copy(); near['Accts_near_copy']=df.Accts+noise
sales_delta=.01*df.Sales.std(ddof=1)
y_changed=df.Sales.copy(); y_changed.iloc[0]+=sales_delta
models={'original':sm.OLS(df.Sales,base).fit(),'exact_copy':sm.OLS(df.Sales,exact).fit(),'near_copy':sm.OLS(df.Sales,near).fit(),'near_copy_changed_sales':sm.OLS(y_changed,near).fit()}
rows=[]
for name,fit in models.items():
    X=fit.model.exog
    copy_name='Accts_copy' if name=='exact_copy' else 'Accts_near_copy'
    rows.append(dict(model=name,rank=int(np.linalg.matrix_rank(X)),columns=X.shape[1],unique=bool(np.linalg.matrix_rank(X)==X.shape[1]),Accts=float(fit.params.Accts),copy_coefficient=float(fit.params[copy_name]) if copy_name in fit.params else None))
coef=pd.DataFrame(rows); coef.to_csv(ROOT/'stress_coefficients.csv',index=False)
f0=np.asarray(models['near_copy'].predict(near)); f1=np.asarray(models['near_copy_changed_sales'].predict(near)); change=f1-f0
out=pd.DataFrame({'territory_row':np.arange(1,len(df)+1),'Sales_original':df.Sales,'Sales_constructed':y_changed,'fixed_noise':noise,'Accts_near_copy':near.Accts_near_copy,'fitted_before':f0,'fitted_after':f1,'fitted_change':change})
out.to_csv(ROOT/'stress_row_results.csv',index=False)
dcoef=models['near_copy_changed_sales'].params-models['near_copy'].params
dcoef.to_csv(ROOT/'stress_all_coefficient_changes.csv',header=['change'])
results=dict(label='Constructed collinearity stress test, not new-territory validation',seed=4031,noise_generator='numpy default_rng normal',noise_sd=float(noise_sd),Accts_sample_sd=float(df.Accts.std(ddof=1)),Sales_sample_sd=float(df.Sales.std(ddof=1)),first_sales_increase=float(sales_delta),models=rows,Accts_coefficient_change=float(dcoef.Accts),copy_coefficient_change=float(dcoef.Accts_near_copy),fitted_max_absolute_change=float(np.max(np.abs(change))),fitted_rms_change=float(np.sqrt(np.mean(change**2))),fitted_first_row_change=float(change[0]),original_csv_unchanged=before==hashlib.sha256(path.read_bytes()).hexdigest())
(ROOT/'stress_results.json').write_text(json.dumps(results,indent=2))
plt.rcParams.update({'font.size':14,'axes.labelsize':15,'axes.titlesize':17,'xtick.labelsize':13,'ytick.labelsize':13})
fig,ax=plt.subplots(figsize=(9,5.5),layout='constrained')
ax.scatter(out.territory_row,change,s=65)
ax.axhline(0,color='black',linewidth=1)
ax.set(title='Constructed near-copy stress test: fitted-value changes',xlabel='Original territory row',ylabel='Change in fitted Sales (hundreds of units)')
ax.grid(alpha=.2)
fig.savefig(ROOT/'stress_fitted_changes.png',dpi=180)
plt.close(fig)
print(json.dumps(results,indent=2))
