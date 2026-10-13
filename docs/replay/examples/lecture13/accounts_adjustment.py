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
others=[c for c in df.columns if c not in ['Sales','Accts']]
Z=sm.add_constant(df[others],has_constant='add')
a=sm.OLS(df.Accts,Z).fit()
y=sm.OLS(df.Sales,Z).fit()
full=sm.OLS(df.Sales,sm.add_constant(df.drop(columns='Sales'),has_constant='add')).fit()
r_a=a.resid
r_y=y.resid
partial=sm.OLS(r_y,r_a).fit()
tss=float(np.sum((df.Accts-df.Accts.mean())**2))
rss=float(np.sum(r_a**2))
r2=float(a.rsquared)
vif=1/(1-r2)
se_factor=np.sqrt(vif)
res=dict(other_predictors=others,n=len(df),auxiliary_r_squared=r2,one_minus_r_squared=1-r2,
         accounts_centered_sum_squares=tss,accounts_residual_sum_squares=rss,
         vif=vif,standard_error_inflation_factor=float(se_factor),
         residual_slope=float(partial.params.iloc[0]),full_model_accounts_coefficient=float(full.params.Accts),
         slope_difference=float(partial.params.iloc[0]-full.params.Accts),
         full_model_accounts_standard_error=float(full.bse.Accts),
         full_model_accounts_confidence_interval_95=full.conf_int().loc['Accts'].tolist(),
         full_model_residual_df=int(full.df_resid),
         full_model_error_variance_estimate=float(full.mse_resid),
         standard_error_from_residual_information=float(np.sqrt(full.mse_resid/rss)),
         hypothetical_orthogonal_standard_error=float(np.sqrt(full.mse_resid/tss)))
(ROOT/'accounts_adjustment_results.json').write_text(json.dumps(res,indent=2))
pd.DataFrame({'territory_row':np.arange(1,len(df)+1),'Accts':df.Accts,'Accts_fitted_from_other_seven':a.fittedvalues,'Accts_residual':r_a,'Sales_residual':r_y}).to_csv(ROOT/'accounts_residuals.csv',index=False)
(ROOT/'accounts_auxiliary_summary.txt').write_text(a.summary().as_text())
plt.rcParams.update({'font.size':14,'axes.labelsize':15,'axes.titlesize':17,'xtick.labelsize':13,'ytick.labelsize':13})
fig,ax=plt.subplots(figsize=(9,5.5),layout='constrained')
ax.scatter(np.arange(1,len(df)+1),r_a,s=65)
ax.axhline(0,color='black',linewidth=1)
ax.set(title='Accts variation left after adjustment',xlabel='Territory row in supplied CSV',ylabel='Accts residual (supplied accounts units)')
ax.grid(alpha=.2)
fig.savefig(ROOT/'accounts_remaining_variation.png',dpi=180)
plt.close(fig)
fig,ax=plt.subplots(figsize=(8,6),layout='constrained')
ax.scatter(r_a,r_y,s=65)
grid=np.linspace(r_a.min(),r_a.max(),200)
ax.plot(grid,float(partial.params.iloc[0])*grid,color='darkorange',linewidth=2,label=f'Slope = {partial.params.iloc[0]:.6f}')
ax.axhline(0,color='gray',linewidth=.8)
ax.axvline(0,color='gray',linewidth=.8)
ax.set(title='Sales versus Accts after the same adjustment',xlabel='Accts residual (supplied accounts units)',ylabel='Sales residual (hundreds of units)')
ax.legend(fontsize=13)
ax.grid(alpha=.2)
fig.savefig(ROOT/'sales_accounts_residuals.png',dpi=180)
plt.close(fig)
print(json.dumps(res,indent=2))
