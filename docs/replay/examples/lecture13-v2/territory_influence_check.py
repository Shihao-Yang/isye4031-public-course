import os
os.environ['MPLCONFIGDIR']='.mplcache'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

df=pd.read_csv('sales_territory.csv'); x=sm.add_constant(df.drop(columns='Sales'))
full=sm.OLS(df.Sales,x).fit();inf=full.get_influence()
rows=[]
for i in range(len(df)):
 keep=np.arange(len(df))!=i
 fit=sm.OLS(df.Sales[keep],x[keep]).fit()
 reduced=sm.OLS(df.Sales[keep],x.drop(columns=['Time','Accts'])[keep]).fit()
 f,p,q=fit.compare_f_test(reduced)
 rows.append({'territory_row':i+1,'leverage':float(inf.hat_matrix_diag[i]),'externally_studentized_residual':float(inf.resid_studentized_external[i]),'cooks_distance':float(inf.cooks_distance[0][i]),'Accts_dfbeta':float(inf.dfbetas[i,x.columns.get_loc('Accts')]),'Time_dfbeta':float(inf.dfbetas[i,x.columns.get_loc('Time')]),'Accts_coefficient_without_row':float(fit.params.Accts),'Accts_p_without_row':float(fit.pvalues.Accts),'Time_coefficient_without_row':float(fit.params.Time),'joint_Time_Accts_p_without_row':float(p)})
table=pd.DataFrame(rows);table.to_csv('territory_influence_diagnostics.csv',index=False)
original_reduced=sm.OLS(df.Sales,x.drop(columns=['Time','Accts'])).fit()
out={'model':'Original additive OLS with all eight predictors and intercept; rows numbered 1 through 25 in supplied CSV','full_model':{'Accts_coefficient':float(full.params.Accts),'Accts_p':float(full.pvalues.Accts),'Time_coefficient':float(full.params.Time),'joint_Time_Accts_p':float(full.compare_f_test(original_reduced)[1])},'top_five_Cooks_distance':table.sort_values('cooks_distance',ascending=False).head(5).to_dict(orient='records'),'leave_one_out_ranges':{c:{'min':float(table[c].min()),'max':float(table[c].max())} for c in ['Accts_coefficient_without_row','Time_coefficient_without_row','joint_Time_Accts_p_without_row']},'most_influential_row_values':df.iloc[int(table.loc[table.cooks_distance.idxmax(),'territory_row'])-1].to_dict(),'duplicate_rows':int(df.duplicated().sum()),'missing_values':int(df.isna().sum().sum()),'predictor_ranges':{c:{'min':float(df[c].min()),'max':float(df[c].max())} for c in df.columns}}
with open('territory_influence_results.json','w') as f:json.dump(out,f,indent=2)
fig,ax=plt.subplots(figsize=(7,4.7));ax.scatter(table.leverage,table.externally_studentized_residual,s=30+200*table.cooks_distance,color='tab:blue',alpha=.7)
for row in table.sort_values('cooks_distance',ascending=False).head(3).itertuples():ax.annotate(str(row.territory_row),(row.leverage,row.externally_studentized_residual),xytext=(5,5),textcoords='offset points')
ax.axhline(0,color='gray',lw=.7);ax.set_xlabel('Leverage in full eight-predictor model');ax.set_ylabel('Externally studentized residual');ax.set_title('Territory influence: bubble size increases with Cook’s distance');fig.tight_layout();fig.savefig('territory_influence_plot.png',dpi=170)
print(json.dumps(out,indent=2))
