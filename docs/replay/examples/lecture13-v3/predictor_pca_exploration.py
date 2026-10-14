import os
os.environ['MPLCONFIGDIR']='.mplcache'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json

df=pd.read_csv('sales_territory.csv');x=df.drop(columns='Sales');scaler=StandardScaler();z=scaler.fit_transform(x);pca=PCA();scores=pca.fit_transform(z)
# Orient each component so its largest weight is positive; signs have no statistical meaning.
weights=pca.components_.copy()
for j in range(8):
    if weights[j,np.argmax(np.abs(weights[j]))]<0:weights[j]*=-1;scores[:,j]*=-1
pd.DataFrame(weights.T,index=x.columns,columns=[f'PC{i+1}' for i in range(8)]).to_csv('predictor_pca_weights.csv')
pd.DataFrame(scores,columns=[f'PC{i+1}' for i in range(8)]).to_csv('predictor_pca_scores.csv',index=False)
base=sm.OLS(df.Sales,sm.add_constant(x)).fit();fits=[]
for k in range(1,9):
    m=sm.OLS(df.Sales,sm.add_constant(scores[:,:k])).fit()
    fits.append({'components':k,'cumulative_predictor_variance':float(pca.explained_variance_ratio_[:k].sum()),'in_sample_sales_R_squared':float(m.rsquared),'RMS_fitted_difference_from_full_model':float(np.sqrt(np.mean((m.fittedvalues-base.fittedvalues)**2)))})
pd.DataFrame(fits).to_csv('predictor_pca_in_sample_fit_comparison.csv',index=False)
r={'method':'PCA of all eight predictors standardized to mean zero and population SD one, fitted to all 25 rows; Sales excluded from PCA','variance_fractions':pca.explained_variance_ratio_.tolist(),'weights':{f'PC{i+1}':dict(zip(x.columns,weights[i].tolist())) for i in range(3)},'in_sample_component_regressions':fits,'max_absolute_off_diagonal_score_covariance':float(np.max(np.abs(np.cov(scores,rowvar=False)-np.diag(np.diag(np.cov(scores,rowvar=False)))))),'note':'Exploratory in-sample analysis, not validation. New component columns are mathematical combinations of actual predictors. Component signs are arbitrary.'}
with open('predictor_pca_exploration_results.json','w') as f:json.dump(r,f,indent=2)
fig,axes=plt.subplots(1,2,figsize=(11,5))
ratios=pca.explained_variance_ratio_
axes[0].bar(np.arange(1,9),ratios*100,label='Individual component');axes[0].plot(np.arange(1,9),np.cumsum(ratios)*100,'o-',color='#bc402b',label='Cumulative');axes[0].set_xticks(range(1,9));axes[0].set_xlabel('Principal component');axes[0].set_ylabel('Standardized predictor variance (%)');axes[0].set_title('How much predictor variation is retained?');axes[0].legend(fontsize=8)
im=axes[1].imshow(weights[:3].T,cmap='RdBu_r',vmin=-.7,vmax=.7,aspect='auto');axes[1].set_yticks(range(8),x.columns);axes[1].set_xticks(range(3),['PC1','PC2','PC3'])
for i in range(8):
    for j in range(3):axes[1].text(j,i,f'{weights[j,i]:.2f}',ha='center',va='center',fontsize=10)
axes[1].set_title('Weights on standardized predictors');fig.colorbar(im,ax=axes[1],shrink=.8,label='Component weight')
fig.suptitle('PCA of actual sales-territory predictors: all 25 observations',fontsize=14);fig.tight_layout();fig.savefig('predictor_pca_exploration.png',dpi=180)
print(json.dumps({'variance_fractions':r['variance_fractions'],'first_three_weights':r['weights'],'fits':fits},indent=2))
