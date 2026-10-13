from pathlib import Path
import json
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f as f_dist
ROOT=Path(__file__).resolve().parent
df=pd.read_csv(ROOT/'sales_territory.csv')
full=sm.OLS(df.Sales,sm.add_constant(df.drop(columns='Sales'),has_constant='add')).fit()
reduced=sm.OLS(df.Sales,sm.add_constant(df.drop(columns=['Sales','Time','Accts']),has_constant='add')).fit()
q=int(reduced.df_resid-full.df_resid)
num=(reduced.ssr-full.ssr)/q
den=full.ssr/full.df_resid
F=num/den
p=f_dist.sf(F,q,full.df_resid)
res=dict(model='Original unmodified eight-predictor OLS model with intercept',reduced_predictors=list(reduced.params.index),n=len(df),SSE_reduced=float(reduced.ssr),SSE_full=float(full.ssr),SSE_difference=float(reduced.ssr-full.ssr),numerator_df=q,denominator_df=int(full.df_resid),F_numerator=float(num),F_denominator=float(den),partial_F=float(F),joint_p_value=float(p),individual_p_values={v:float(full.pvalues[v]) for v in ['Time','Accts']},software_crosscheck=list(map(float,full.compare_f_test(reduced))),assumptions='Nested fixed-design linear models, correct full conditional mean, independent normal homoskedastic errors for exact F and t inference; exploratory comparison selected after inspecting data.')
(ROOT/'joint_test_results.json').write_text(json.dumps(res,indent=2))
pd.DataFrame([{'model':'reduced_without_Time_Accts','SSE':reduced.ssr,'residual_df':reduced.df_resid},{'model':'original_full','SSE':full.ssr,'residual_df':full.df_resid}]).to_csv(ROOT/'joint_test_model_comparison.csv',index=False)
pd.DataFrame({'coefficient':full.params,'standard_error':full.bse,'p_value':full.pvalues}).to_csv(ROOT/'joint_test_full_coefficients.csv')
print(json.dumps(res,indent=2))
