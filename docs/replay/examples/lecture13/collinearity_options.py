from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
ROOT=Path(__file__).resolve().parent
df=pd.read_csv(ROOT/'sales_territory.csv')
p=list(df.drop(columns='Sales').columns)
raw=df[p]
centered=raw-raw.mean()
scaled=centered/raw.std(ddof=1)
interaction=centered.copy()
interaction['Time_x_Accts']=centered.Time*centered.Accts
reduced=raw.drop(columns='Accts')
frames={'original':raw,'centered':centered,'centered_scaled':scaled,'centered_interaction':interaction,'without_Accts':reduced}
fits={}
vifs={}
for name,frame in frames.items():
    X=sm.add_constant(frame,has_constant='add')
    fits[name]=sm.OLS(df.Sales,X).fit()
    vifs[name]={col:float(variance_inflation_factor(X.to_numpy(),i)) for i,col in enumerate(X.columns) if col!='const'}
vtable=pd.DataFrame(vifs).reindex(p+['Time_x_Accts'])
vtable.to_csv(ROOT/'options_vifs.csv')
rows=[]
for name in ['original','without_Accts']:
    f=fits[name]
    rows.append(dict(model=name,Time_coefficient=float(f.params.Time),Time_standard_error=float(f.bse.Time),Time_p_value=float(f.pvalues.Time)))
pd.DataFrame(rows).to_csv(ROOT/'options_time_comparison.csv',index=False)
pd.DataFrame({name:f.fittedvalues for name,f in fits.items()}).to_csv(ROOT/'options_fitted_values.csv',index=False)
for name,f in fits.items():
    pd.DataFrame({'coefficient':f.params,'standard_error':f.bse,'p_value':f.pvalues}).to_csv(ROOT/f'options_{name}_coefficients.csv')
res=dict(time_comparison=rows,vifs=vifs,coordinate_checks={name:dict(max_absolute_vif_difference=float(max(abs(vifs[name][c]-vifs['original'][c]) for c in p)),max_absolute_fitted_difference=float(np.max(np.abs(fits[name].fittedvalues-fits['original'].fittedvalues)))) for name in ['centered','centered_scaled']},interaction=dict(coefficient=float(fits['centered_interaction'].params.Time_x_Accts),p_value=float(fits['centered_interaction'].pvalues.Time_x_Accts)),inference_assumptions='Correct linear conditional mean, independent homoskedastic errors, and normal errors for exact finite-sample t tests. Models fitted to identical unmodified rows.')
(ROOT/'options_results.json').write_text(json.dumps(res,indent=2))
print(json.dumps(res,indent=2))
