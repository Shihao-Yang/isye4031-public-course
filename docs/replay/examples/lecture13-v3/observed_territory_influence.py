import os
os.environ['MPLCONFIGDIR']='.mplcache'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import json

df=pd.read_csv('sales_territory.csv');X=sm.add_constant(df.drop(columns='Sales'));m=sm.OLS(df.Sales,X).fit();inf=m.get_influence()
rows=[]
for i in range(len(df)):
    keep=np.arange(len(df))!=i
    deleted=sm.OLS(df.Sales[keep],X[keep]).fit()
    delta=deleted.predict(X)-m.fittedvalues
    rows.append({'row':i+1,'leverage':float(inf.hat_matrix_diag[i]),'externally_studentized_residual':float(inf.resid_studentized_external[i]),'Cooks_distance':float(inf.cooks_distance[0][i]),'RMS_fitted_change_all_25':float(np.sqrt(np.mean(delta**2))),'max_absolute_fitted_change_all_25':float(np.max(np.abs(delta))),'Time_slope_without_row':float(deleted.params.Time),'Accounts_slope_without_row':float(deleted.params.Accts)})
out=pd.DataFrame(rows);out.to_csv('observed_territory_influence_values.csv',index=False)
top=out.nlargest(4,'Cooks_distance');lev=out.nlargest(4,'leverage')
selected=sorted(set(top.row)|set(lev.row));df.iloc[[i-1 for i in selected]].assign(CSV_row=selected).to_csv('unusual_territories_original_values.csv',index=False)
r={'model':'Full eight-predictor OLS sales model with intercept, all 25 rows','average_leverage':9/25,'top_leverage':lev.to_dict(orient='records'),'top_Cooks_distance':top.to_dict(orient='records'),'original_Time_slope':float(m.params.Time),'original_Accounts_slope':float(m.params.Accts),'original_residual_SD':float(np.sqrt(m.mse_resid)),'note':'One-row deletion refits are sensitivity analysis, not validation or a recommendation to delete. Fitted changes measured at all 25 original predictor vectors, including the omitted point.'}
with open('observed_territory_influence_results.json','w') as f:json.dump(r,f,indent=2)
fig,axes=plt.subplots(1,2,figsize=(11,5))
axes[0].scatter(out.leverage,out.externally_studentized_residual,s=35+out.Cooks_distance*200,color='#326a9f',alpha=.8)
for row in out.itertuples():
    if row.row in selected or abs(row.externally_studentized_residual)>2:axes[0].annotate(str(row.row),(row.leverage,row.externally_studentized_residual),xytext=(5,4),textcoords='offset points')
axes[0].axhline(0,color='gray',lw=.7);axes[0].set_xlabel('Leverage: unusual predictor combination');axes[0].set_ylabel('Externally studentized residual');axes[0].set_title('Bubble size increases with Cook’s distance')
axes[1].bar(out.row,out.RMS_fitted_change_all_25,color='#326a9f');axes[1].set_xlabel('Observation omitted (CSV row order)');axes[1].set_ylabel('RMS change in fitted Sales at all 25 points\n(hundreds of units)');axes[1].set_title('Measured influence: one-row deletion refits')
fig.suptitle('Observed territory diagnostics: original full eight-predictor model',fontsize=14)
fig.tight_layout();fig.savefig('observed_territory_influence.png',dpi=180)
print(json.dumps(r,indent=2))
