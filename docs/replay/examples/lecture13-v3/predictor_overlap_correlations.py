import os
os.environ['MPLCONFIGDIR'] = '.mplcache'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

df = pd.read_csv('sales_territory.csv')
x = df.drop(columns='Sales')
corr = x.corr()
corr.to_csv('predictor_overlap_correlations.csv')
pairs = sorted([{'predictor_1':x.columns[i], 'predictor_2':x.columns[j], 'correlation':float(corr.iloc[i,j])} for i in range(len(x.columns)) for j in range(i)], key=lambda p:abs(p['correlation']), reverse=True)
fig, ax = plt.subplots(figsize=(9,7.5))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(x.columns)),x.columns,rotation=40,ha='right')
ax.set_yticks(range(len(x.columns)),x.columns)
for i in range(len(x.columns)):
    for j in range(len(x.columns)):
        value = corr.iloc[i,j]
        ax.text(j,i,f'{value:.2f}',ha='center',va='center',color='white' if abs(value)>.65 else 'black',fontsize=11)
fig.colorbar(im, ax=ax, label='Pearson correlation',shrink=.8)
ax.set_title('Overlap among predictors in the actual sales-territory data\nAll 25 observations; pairwise linear associations',pad=18)
fig.tight_layout()
fig.savefig('predictor_overlap_correlations.png',dpi=180)
results={'observations':len(df),'method':'Pearson correlations among eight predictors, all observations retained','ranked_pairs':pairs}
with open('predictor_overlap_correlations_results.json','w') as f:json.dump(results,f,indent=2)
print(json.dumps(pairs[:8],indent=2))
