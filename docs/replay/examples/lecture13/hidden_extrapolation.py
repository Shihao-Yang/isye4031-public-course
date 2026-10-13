import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(ROOT/'.mplcache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import json

df=pd.read_csv(ROOT/'sales_territory.csv')
predictors=list(df.drop(columns='Sales').columns)
X=sm.add_constant(df[predictors],has_constant='add')
fit=sm.OLS(df.Sales,X).fit()
profiles=pd.DataFrame([df[predictors].mean(),df[predictors].mean()],index=['All predictor means','Min Time, max Accts'])
profiles.loc['Min Time, max Accts','Time']=df.Time.min()
profiles.loc['Min Time, max Accts','Accts']=df.Accts.max()
X0=sm.add_constant(profiles,has_constant='add')[X.columns]
# QR avoids forming the poorly scaled raw cross-product inverse.
Q,R=np.linalg.qr(X.to_numpy(),mode='reduced')
transformed=np.linalg.solve(R.T,X0.to_numpy().T)
h0=np.sum(transformed**2,axis=0)
training_h=np.sum(Q**2,axis=1)
pred=fit.get_prediction(X0).summary_frame(alpha=.05)
output=profiles.copy()
output['h0']=h0
output['fitted_Sales']=pred['mean']
output['prediction_lower_95']=pred['obs_ci_lower']
output['prediction_upper_95']=pred['obs_ci_upper']
output['prediction_width']=pred['obs_ci_upper']-pred['obs_ci_lower']
output.to_csv(ROOT/'hypothetical_profiles.csv')
pd.DataFrame({'territory_row':np.arange(1,len(df)+1),'training_leverage':training_h}).to_csv(ROOT/'training_leverages.csv',index=False)
res=dict(model='Original eight-predictor OLS model with intercept',profile_label='Constructed teaching examples, not observed new territories',n=len(df),residual_df=int(fit.df_resid),largest_training_leverage=float(training_h.max()),largest_training_leverage_row=int(np.argmax(training_h)+1),profiles=output.reset_index(names='profile').to_dict(orient='records'),all_profiles_within_separate_ranges=bool(((profiles>=df[predictors].min())&(profiles<=df[predictors].max())).all().all()),error_variance_estimate=float(fit.mse_resid),interval_assumptions='Correct linear mean also at the hypothetical profile; independent normal homoskedastic errors; new individual error independent of training errors; fixed accurately measured predictors.')
(ROOT/'hidden_extrapolation_results.json').write_text(json.dumps(res,indent=2))
plt.rcParams.update({'font.size':14,'axes.labelsize':15,'axes.titlesize':17,'xtick.labelsize':13,'ytick.labelsize':13})
fig,ax=plt.subplots(figsize=(9,6),layout='constrained')
ax.scatter(df.Time,df.Accts,s=60,alpha=.75,label='Observed territories')
for label,marker,color in [('All predictor means','*','darkorange'),('Min Time, max Accts','X','crimson')]:
    ax.scatter(profiles.loc[label,'Time'],profiles.loc[label,'Accts'],marker=marker,color=color,s=220,label='Hypothetical: '+label,zorder=3)
ax.set(title='Teaching profiles and observed Time–Accts combinations',xlabel='Time with company (months)',ylabel='Accts (supplied accounts measure)')
ax.legend(fontsize=13,loc='best')
ax.grid(alpha=.2)
fig.savefig(ROOT/'hypothetical_time_accts.png',dpi=180)
plt.close(fig)
print(json.dumps(res,indent=2))
