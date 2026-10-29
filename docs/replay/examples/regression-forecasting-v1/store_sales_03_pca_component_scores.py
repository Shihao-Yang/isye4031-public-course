from pathlib import Path
import os
os.environ['MPLCONFIGDIR'] = str(Path.cwd() / '.mplcache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

cols = ['size_sqft', 'advertising_thousand_usd_month', 'staff_count',
        'area_household_income_thousand_usd_year', 'nearby_competitor_count']
labels = ['Store area', 'Advertising', 'Staff', 'Area income', 'Competitors']
d = pd.read_csv('store_sales_01_data.csv')
d = d.loc[d['split'].eq('training'), ['store_id'] + cols].reset_index(drop=True)
X = d[cols].to_numpy()
scaler = StandardScaler().fit(X)
z = scaler.transform(X)
pca = PCA().fit(z)
# Choose a convenient sign for PC1. The reversed sign represents the same axis.
if pca.components_[0, 0] < 0:
    pca.components_[0] *= -1
scores = pca.transform(z)
example = pd.DataFrame({'input':labels, 'raw_value':X[0], 'training_mean':scaler.mean_,
    'training_sd_ddof0':scaler.scale_, 'standardized_value':z[0],
    'pca_center':pca.mean_, 'PC1_weight':pca.components_[0],
    'PC1_contribution':(z[0]-pca.mean_)*pca.components_[0]})
example.to_csv('store_sales_03_score_calculation.csv', index=False)
weights = pd.DataFrame(pca.components_.T, index=cols, columns=[f'PC{i}' for i in range(1,6)])
weights.to_csv('store_sales_03_pca_weights.csv', index_label='input')
variance = pd.DataFrame({'component':np.arange(1,6), 'variance_fraction':pca.explained_variance_ratio_,
    'cumulative_fraction':np.cumsum(pca.explained_variance_ratio_)})
variance.to_csv('store_sales_03_explained_variance.csv', index=False)
out = pd.DataFrame(scores, columns=[f'PC{i}' for i in range(1,6)])
out.insert(0,'store_id',d.store_id)
out.to_csv('store_sales_03_training_scores.csv', index=False)
pd.DataFrame(X,columns=labels).corr().to_csv('store_sales_03_input_correlations.csv',index_label='input')
summary = {'training_stores':len(d), 'inputs':cols, 'example_store_id':int(d.store_id.iloc[0]),
    'example_PC1_score':float(scores[0,0]), 'sum_of_contributions':float(example.PC1_contribution.sum()),
    'PC1_weight_squared_sum':float(np.sum(pca.components_[0]**2)),
    'sales_used':False,'test_used':False,
    'standardization':'Training mean and population standard deviation (ddof=0)',
    'purpose':'Descriptive PCA of original five inputs. No squared term. Future predictive validation must refit scaling and PCA within each training fold.'}
Path('store_sales_03_pca_summary.json').write_text(json.dumps(summary,indent=2))
plt.rcParams.update({'font.size':12, 'axes.titlesize':14})
fig, ax = plt.subplots(figsize=(8,4.5))
positions = np.arange(1,6)
ax.bar(positions,100*pca.explained_variance_ratio_,color='#2166ac',width=.58,label='Individual component')
ax.plot(positions,100*np.cumsum(pca.explained_variance_ratio_),'o-',color='#c45118',lw=2,label='Cumulative')
for i, pct in zip(positions,100*pca.explained_variance_ratio_):
    ax.text(i,pct+2,f'{pct:.1f}%',ha='center',fontsize=11)
ax.set(xticks=positions,xticklabels=[f'PC{i}' for i in positions],ylim=(0,112),
       ylabel='Variance in standardized inputs (%)',title='PCA of the five inputs: 225 training stores')
ax.legend(loc='center right',fontsize=10)
ax.grid(axis='y',alpha=.2)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
fig.savefig('store_sales_03_pca_variance.png',dpi=180)
print(example.to_string(index=False,float_format=lambda v:f'{v:.6f}'))
print(variance.to_string(index=False))
print(json.dumps(summary,indent=2))
