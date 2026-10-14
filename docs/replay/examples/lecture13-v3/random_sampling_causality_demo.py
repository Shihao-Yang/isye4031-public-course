import os
os.environ['MPLCONFIGDIR'] = '.mplcache'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
rng = np.random.default_rng(403113)
n = 100000
potential = rng.normal(size=n)
ad_noise = rng.normal(size=n)
sales_noise = rng.normal(size=n)
# Constructed population: advertising has zero causal effect on sales.
ad = 2 * potential + ad_noise
sales = 3 * potential + sales_noise
idx = rng.choice(n, size=1000, replace=False)
assigned = rng.permutation(ad[idx])
df = pd.DataFrame({'territory_potential': potential[idx], 'observational_advertising': ad[idx], 'randomly_assigned_advertising': assigned, 'sales': sales[idx]})
df.to_csv('constructed_random_sampling_demo.csv', index=False)
def slope(x,y):
    return float(np.linalg.lstsq(np.column_stack([np.ones(len(x)),x]),y,rcond=None)[0][1])
b_obs = slope(df.observational_advertising, df.sales)
b_rand = slope(df.randomly_assigned_advertising, df.sales)
fig, axes = plt.subplots(1,2,figsize=(11,4.8),sharey=True)
for ax, col, title, b in zip(axes, ['observational_advertising','randomly_assigned_advertising'], ['Random sample; advertising follows territory potential','Same sample; advertising randomly assigned'],[b_obs,b_rand]):
    x=df[col].to_numpy(); y=df.sales.to_numpy()
    ax.scatter(x,y,s=10,alpha=.25,color='#326a9f')
    xx=np.array([x.min(),x.max()])
    ax.plot(xx,y.mean()+b*(xx-x.mean()),color='#bc402b',lw=2,label=f'Fitted slope: {b:.3f}')
    ax.set_title(title,fontsize=11)
    ax.set_xlabel('Constructed advertising (arbitrary units)')
    ax.legend(loc='upper left')
axes[0].set_ylabel('Constructed sales (arbitrary units)')
fig.suptitle('Constructed experiment: true causal effect of advertising is zero',fontsize=14)
fig.tight_layout()
fig.savefig('random_sampling_causality_demo.png',dpi=180)
results={'seed':403113,'population_size':n,'random_sample_size':1000,'data_generating_equations':{'advertising':'2 * territory_potential + independent_noise','sales':'3 * territory_potential + independent_noise'},'true_advertising_causal_effect':0,'observational_simple_regression_slope':b_obs,'random_assignment_simple_regression_slope':b_rand,'note':'Constructed illustration, not analysis or validation of the supplied sales-territory data.'}
with open('random_sampling_causality_demo_results.json','w') as f: json.dump(results,f,indent=2)
print(json.dumps(results,indent=2))
