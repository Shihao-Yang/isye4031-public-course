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
X=sm.add_constant(df.drop(columns='Sales'),has_constant='add')
f=sm.OLS(df.Sales,X).fit()
i=f.get_influence()
d=pd.DataFrame({'csv_row_position':np.arange(1,len(df)+1),'fitted_Sales':f.fittedvalues,'residual':f.resid,'leverage':i.hat_matrix_diag,'externally_studentized_residual':i.resid_studentized_external,'Cooks_distance':i.cooks_distance[0]})
d.to_csv(ROOT/'influence_all_rows.csv',index=False)
top_h=d.nlargest(2,'leverage').index.tolist()
top_t=int(d.externally_studentized_residual.abs().idxmax())
top_c=int(d.Cooks_distance.idxmax())
selected=list(dict.fromkeys(top_h+[top_t]))
d.loc[selected].to_csv(ROOT/'influence_selected_rows.csv',index=False)
keep=np.arange(len(df))!=top_c
s=sm.OLS(df.Sales.loc[keep],X.loc[keep]).fit()
changes=[dict(predictor=p,original=float(f.params[p]),sensitivity_excluding_largest_Cook=float(s.params[p]),change=float(s.params[p]-f.params[p])) for p in ['Time','Accts']]
pd.DataFrame(changes).to_csv(ROOT/'influence_sensitivity_coefficients.csv',index=False)
pd.DataFrame({'original_coefficient':f.params,'sensitivity_coefficient':s.params,'change':s.params-f.params}).to_csv(ROOT/'influence_all_coefficient_changes.csv')
results=dict(model='Original eight-predictor OLS with intercept',row_numbers='Positions in supplied CSV, excluding header',selected_rows=d.loc[selected].to_dict(orient='records'),largest_Cook_row=top_c+1,largest_Cook_distance=float(d.loc[top_c,'Cooks_distance']),sensitivity_label='Excluding largest-Cook row only as sensitivity check; not a data-cleaning decision',sensitivity_changes=changes)
(ROOT/'influence_results.json').write_text(json.dumps(results,indent=2))
plt.rcParams.update({'font.size':14,'axes.labelsize':15,'axes.titlesize':17,'xtick.labelsize':13,'ytick.labelsize':13})
labels=list(dict.fromkeys(selected+[top_c]))
fig,ax=plt.subplots(figsize=(9,6),layout='constrained')
ax.scatter(d.fitted_Sales,d.residual,s=65)
ax.axhline(0,color='black',linewidth=1)
for j in labels: ax.annotate(str(j+1),(d.loc[j,'fitted_Sales'],d.loc[j,'residual']),xytext=(7,7),textcoords='offset points',fontsize=14)
ax.set(title='Original full model: residuals versus fitted Sales',xlabel='Fitted Sales (hundreds of units)',ylabel='Residual Sales (hundreds of units)')
ax.grid(alpha=.2)
fig.savefig(ROOT/'residuals_vs_fitted.png',dpi=180)
plt.close(fig)
fig,ax=plt.subplots(figsize=(9,6),layout='constrained')
sc=ax.scatter(d.leverage,d.externally_studentized_residual,s=65+220*d.Cooks_distance,c=d.Cooks_distance,cmap='viridis',edgecolors='black',linewidth=.5)
for j in labels: ax.annotate(str(j+1),(d.loc[j,'leverage'],d.loc[j,'externally_studentized_residual']),xytext=(8,8),textcoords='offset points',fontsize=14)
ax.axhline(0,color='gray',linewidth=1)
ax.set(title="Original full model: leverage, residuals, and influence",xlabel='Leverage (dimensionless)',ylabel='Externally studentized residual (dimensionless)')
fig.colorbar(sc,ax=ax,label="Cook's distance (color and marker size)")
ax.grid(alpha=.2)
fig.savefig(ROOT/'leverage_residual_cook.png',dpi=180)
plt.close(fig)
print(json.dumps(results,indent=2))
