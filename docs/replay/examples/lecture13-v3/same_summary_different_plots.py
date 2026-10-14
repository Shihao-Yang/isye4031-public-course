import os
os.environ['MPLCONFIGDIR']='.mplcache'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import json
rng=np.random.default_rng(403116)
x=np.linspace(-2,2,30);X=np.column_stack([np.ones(len(x)),x]);Q=np.linalg.qr(X)[0]
spike=np.zeros(len(x));spike[23]=1
patterns={'Scattered residuals':rng.normal(size=len(x)),'Curved relationship':x*x,'One large residual':spike}
rows=[];data=[];fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True,sharey='row')
for j,(name,v) in enumerate(patterns.items()):
    residual=v-Q@(Q.T@v);residual*=np.sqrt(30)/np.linalg.norm(residual)
    y=5+2*x+residual;m=sm.OLS(y,X).fit()
    rows.append({'dataset':name,'n':30,'x_mean':float(x.mean()),'y_mean':float(y.mean()),'x_sample_variance':float(x.var(ddof=1)),'y_sample_variance':float(y.var(ddof=1)),'correlation':float(np.corrcoef(x,y)[0,1]),'intercept':float(m.params[0]),'slope':float(m.params[1]),'R_squared':float(m.rsquared),'SSE':float(m.ssr),'slope_SE':float(m.bse[1])})
    for i in range(30):data.append({'dataset':name,'row':i+1,'constructed_x':float(x[i]),'constructed_y':float(y[i]),'residual':float(residual[i])})
    axes[0,j].scatter(x,y,color='#326a9f');axes[0,j].plot(x,m.fittedvalues,color='#bc402b');axes[0,j].set_title(name)
    axes[1,j].scatter(x,m.resid,color='#326a9f');axes[1,j].axhline(0,color='gray',lw=.8);axes[1,j].set_xlabel('Constructed x')
axes[0,0].set_ylabel('Constructed y');axes[1,0].set_ylabel('Residual from the identical fitted line')
fig.suptitle('Three newly constructed datasets: same numerical summaries, different patterns',fontsize=14);fig.tight_layout();fig.savefig('same_summary_different_plots.png',dpi=180)
pd.DataFrame(data).to_csv('constructed_same_summary_datasets.csv',index=False);pd.DataFrame(rows).to_csv('same_summary_comparison.csv',index=False)
with open('same_summary_different_plots_results.json','w') as f:json.dump({'seed':403116,'construction':'Common x; y=5+2*x+residual. Each constructed residual vector is projected perpendicular to intercept and x, then scaled to squared length 30. All datasets constructed here; none is actual territory data.','summaries':rows},f,indent=2)
print(json.dumps(rows,indent=2))
