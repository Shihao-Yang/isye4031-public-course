import os
os.environ['MPLCONFIGDIR']='.mplcache'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro,probplot
import json

df=pd.read_csv('sales_territory.csv');X=sm.add_constant(df.drop(columns='Sales'));m=sm.OLS(df.Sales,X).fit();inf=m.get_influence();std=inf.resid_studentized_internal
fig,axes=plt.subplots(2,2,figsize=(11,8))
ax=axes[0,0];ax.scatter(m.fittedvalues,m.resid);smooth=lowess(m.resid,m.fittedvalues,frac=.7);ax.plot(smooth[:,0],smooth[:,1],color='#bc402b');ax.axhline(0,color='gray',lw=.7);ax.set_xlabel('Fitted Sales');ax.set_ylabel('Residual Sales');ax.set_title('Residuals versus fitted: mean pattern')
ax=axes[0,1];(theory,ordered),(slope,intercept,r)=probplot(std,dist='norm');ax.scatter(theory,ordered);ax.plot(theory,intercept+slope*theory,color='gray');ax.set_xlabel('Normal theoretical quantile');ax.set_ylabel('Ordered standardized residual');ax.set_title('Normal Q–Q plot')
ax=axes[1,0];spread=np.sqrt(np.abs(std));ax.scatter(m.fittedvalues,spread);smooth=lowess(spread,m.fittedvalues,frac=.7);ax.plot(smooth[:,0],smooth[:,1],color='#bc402b');ax.set_xlabel('Fitted Sales');ax.set_ylabel('Square root of |standardized residual|');ax.set_title('Scale–location: changing error spread')
ax=axes[1,1];ax.plot(np.arange(1,26),std,'o-');ax.axhline(0,color='gray',lw=.7);ax.set_xlabel('CSV row order (not established time order)');ax.set_ylabel('Standardized residual');ax.set_title('Residuals in supplied row order')
for ax,yy in [(axes[0,0],m.resid),(axes[1,0],spread),(axes[1,1],std)]:
    for row in [7,10,12]:
        xx=row if ax is axes[1,1] else m.fittedvalues.iloc[row-1]
        ax.annotate(str(row),(xx,yy.iloc[row-1] if hasattr(yy,'iloc') else yy[row-1]),xytext=(4,4),textcoords='offset points',fontsize=9)
fig.suptitle('Error diagnostics: full eight-predictor model, all 25 observations',fontsize=14);fig.tight_layout();fig.savefig('full_model_error_diagnostics.png',dpi=180)
bp=het_breuschpagan(m.resid,X);sh=shapiro(m.resid)
r={'model':'Original eight-predictor OLS sales model with intercept; all 25 observations','residual_df':float(m.df_resid),'Shapiro_Wilk_raw_residuals':{'statistic':float(sh.statistic),'pvalue':float(sh.pvalue),'qualification':'Descriptive screen; fitted residuals are correlated and have unequal variances due to leverage.'},'Breusch_Pagan':{'LM':float(bp[0]),'LM_pvalue':float(bp[1]),'F':float(bp[2]),'F_pvalue':float(bp[3]),'qualification':'Small sample with eight predictors limits diagnostic power and asymptotic calibration.'},'largest_absolute_external_studentized_residual':float(np.max(np.abs(inf.resid_studentized_external))),'note':'Plots do not establish independence; CSV row order has no documented temporal interpretation. LOWESS curves are descriptive.'}
with open('full_model_error_diagnostics_results.json','w') as f:json.dump(r,f,indent=2)
pd.DataFrame({'row':np.arange(1,26),'fitted_Sales':m.fittedvalues,'residual':m.resid,'standardized_residual':std,'externally_studentized_residual':inf.resid_studentized_external}).to_csv('full_model_error_diagnostics_values.csv',index=False)
print(json.dumps(r,indent=2))
