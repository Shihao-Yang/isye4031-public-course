import os
os.environ['MPLCONFIGDIR']='.mplcache'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,LinearRegression
from sklearn.pipeline import make_pipeline
import json

df=pd.read_csv('sales_territory.csv');x=df.drop(columns='Sales');y=df.Sales
alphas=np.unique(np.r_[0,1,10,100,np.logspace(-3,3,65)]);rows=[];coefs=[]
for a in alphas:
    def model():return make_pipeline(StandardScaler(),LinearRegression() if a==0 else Ridge(alpha=a))
    fit=model().fit(x,y);pred=fit.predict(x);coefs.append(fit[-1].coef_)
    changes=[]
    for i in range(len(df)):
        keep=np.arange(len(df))!=i
        deleted=model().fit(x.loc[keep],y.loc[keep])
        changes.append(np.sqrt(np.mean((deleted.predict(x)-pred)**2)))
    rows.append({'alpha':float(a),'training_RMSE':float(np.sqrt(np.mean((y-pred)**2))),'coefficient_L2_norm_standardized':float(np.linalg.norm(fit[-1].coef_)),'mean_deletion_RMS_fitted_change':float(np.mean(changes)),'max_deletion_RMS_fitted_change':float(np.max(changes))})
out=pd.DataFrame(rows);out.to_csv('ridge_tradeoff_path_results.csv',index=False)
pd.DataFrame(coefs,columns=x.columns).assign(alpha=alphas).to_csv('ridge_standardized_coefficient_path.csv',index=False)
selected=[]
for a in [0,1,10,100]:
    fit=make_pipeline(StandardScaler(),LinearRegression() if a==0 else Ridge(alpha=a)).fit(x,y)
    nearest=out.iloc[np.argmin(np.abs(out.alpha-a))].to_dict()
    # Grid includes 1, 10, 100 exactly.
    selected.append(nearest)
r={'model':'All eight actual predictors, standardized; intercept unpenalized; all 25 observations','objective':'sum of squared residuals + alpha * sum of squared standardized slopes','selected_results':selected,'note':'In-sample fit and one-row deletion sensitivity, not predictive validation or tuning. Each deletion refits scaling on its remaining 24 rows. No alpha is selected.'}
with open('ridge_tradeoff_analysis_results.json','w') as f:json.dump(r,f,indent=2)
fig,axes=plt.subplots(1,2,figsize=(11,5))
coefs=np.array(coefs)
for j,col in enumerate(x.columns):axes[0].plot(alphas,coefs[:,j],label=col)
axes[0].set_xscale('symlog',linthresh=.001);axes[0].set_xlabel('Ridge penalty alpha (0 = OLS)');axes[0].set_ylabel('Sales slope per predictor standard deviation\n(hundreds of units)');axes[0].set_title('Coefficients shrink as the penalty increases');axes[0].legend(fontsize=8,ncol=2)
axes[1].plot(alphas,out.training_RMSE,label='Training RMSE',color='#bc402b');axes[1].plot(alphas,out.max_deletion_RMS_fitted_change,label='Largest one-row deletion change',color='#326a9f');axes[1].plot(alphas,out.mean_deletion_RMS_fitted_change,label='Mean one-row deletion change',color='#4a8751');axes[1].set_xscale('symlog',linthresh=.001);axes[1].set_xlabel('Ridge penalty alpha (0 = OLS)');axes[1].set_ylabel('Hundreds of sales units');axes[1].set_title('Fit versus sensitivity');axes[1].legend(fontsize=8)
fig.tight_layout();fig.savefig('ridge_tradeoff_analysis.png',dpi=180)
print(json.dumps(r,indent=2))
