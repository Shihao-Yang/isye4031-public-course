import os
os.environ['MPLCONFIGDIR']='.mplcache'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import json

df=pd.read_csv('sales_territory.csv');X=sm.add_constant(df.drop(columns='Sales'));m=sm.OLS(df.Sales,X).fit();inf=m.get_influence();h=inf.hat_matrix_diag;e=m.resid.to_numpy()
rows=[]
for i in range(25):
    keep=np.arange(25)!=i;deleted=sm.OLS(df.Sales[keep],X[keep]).fit()
    prediction=float(deleted.predict(X.iloc[[i]]).iloc[0]);err=float(df.Sales.iloc[i]-prediction)
    change=float(m.fittedvalues.iloc[i]-prediction)
    dffits=change/(np.sqrt(deleted.mse_resid)*np.sqrt(h[i]))
    rows.append({'row':i+1,'leverage':float(h[i]),'ordinary_residual':float(e[i]),'deleted_prediction':prediction,'deleted_residual_explicit':err,'deleted_residual_formula':float(e[i]/(1-h[i])),'fitted_change_at_omitted_point':change,'DFFITS_explicit':float(dffits),'DFFITS_formula':float(inf.resid_studentized_external[i]*np.sqrt(h[i]/(1-h[i]))),'PRESS_contribution':err**2})
out=pd.DataFrame(rows);out.to_csv('press_dffits_deletion_values.csv',index=False)
r={'model':'Fixed full eight-predictor OLS model with intercept; all 25 observations used once as held-out points; no selection or tuning','PRESS':float(out.PRESS_contribution.sum()),'LOOCV_RMSE':float(np.sqrt(out.PRESS_contribution.mean())),'training_RMSE':float(np.sqrt(np.mean(e**2))),'maximum_deleted_residual_formula_error':float(np.max(np.abs(out.deleted_residual_explicit-out.deleted_residual_formula))),'maximum_DFFITS_formula_error':float(np.max(np.abs(out.DFFITS_explicit-out.DFFITS_formula))),'largest_PRESS_contributions':out.nlargest(4,'PRESS_contribution').to_dict(orient='records'),'largest_absolute_DFFITS':out.loc[out.DFFITS_explicit.abs().nlargest(4).index].to_dict(orient='records'),'qualification':'LOOCV assesses this fixed specification, not the broader model exploration in the conversation; sampling dependence and predictor timing are not resolved.'}
with open('press_dffits_deletion_results.json','w') as f:json.dump(r,f,indent=2)
fig,axes=plt.subplots(1,2,figsize=(11,4.8))
axes[0].scatter(out.ordinary_residual,out.deleted_residual_explicit)
for row in out.nlargest(4,'PRESS_contribution').itertuples():axes[0].annotate(str(row.row),(row.ordinary_residual,row.deleted_residual_explicit),xytext=(5,4),textcoords='offset points')
lims=[min(out.ordinary_residual.min(),out.deleted_residual_explicit.min()),max(out.ordinary_residual.max(),out.deleted_residual_explicit.max())];axes[0].plot(lims,lims,color='gray',linestyle='--');axes[0].set_xlabel('Ordinary residual: observed minus full-model fit');axes[0].set_ylabel('Deleted residual: observed minus held-out prediction');axes[0].set_title('PRESS sums squared deleted residuals')
axes[1].bar(out.row,out.DFFITS_explicit,color=['#326a9f' if v>=0 else '#bc402b' for v in out.DFFITS_explicit]);axes[1].axhline(0,color='gray',lw=.7);axes[1].set_xlabel('Observation omitted (CSV row order)');axes[1].set_ylabel('DFFITS (standardized fitted-value change)');axes[1].set_title('DFFITS measures influence at the omitted point')
fig.suptitle('Explicit deletion refits match the OLS shortcut formulas',fontsize=14);fig.tight_layout();fig.savefig('press_dffits_deletion_check.png',dpi=180)
print(json.dumps(r,indent=2))
