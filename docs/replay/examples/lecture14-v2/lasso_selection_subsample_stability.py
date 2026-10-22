"""Training-only repeated-subsample sensitivity of tuned Lasso selection."""
import os
os.environ['MPLCONFIGDIR'] = '.mplcache'
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

SEED=4031
rng=np.random.default_rng(SEED)
df=pd.read_csv('constructed_sales_seed4031.csv')
split=pd.read_csv('fixed_sales_split.csv')
train=df.loc[split['split'].eq('train')].reset_index(drop=True)
names=[c for c in train if c not in ['store_id','weekly_sales']]
alphas=np.geomspace(.001,10,121)
records=[]
membership=[]
repeats=60
size=int(.8*len(train))
for rep in range(repeats):
    # No duplicates within a subsample, so inner CV cannot place duplicate stores
    # on both sides of a fold. Repeated subsamples overlap across repetitions.
    idx=np.sort(rng.choice(len(train),size=size,replace=False))
    subset=train.iloc[idx]
    fold_seed=int(rng.integers(0,2**31-1))
    pipe=Pipeline([('scale',StandardScaler()),('lasso',Lasso(max_iter=100000,tol=1e-9))])
    search=GridSearchCV(pipe,{'lasso__alpha':alphas},
                        cv=KFold(n_splits=5,shuffle=True,random_state=fold_seed),
                        scoring='neg_root_mean_squared_error',refit=True)
    search.fit(subset[names],subset.weekly_sales.to_numpy()/1000)
    coefs=search.best_estimator_.named_steps['lasso'].coef_
    records.append({'repetition':rep+1,'lambda':float(search.best_params_['lasso__alpha']),
                    'nonzero_count':int(np.sum(np.abs(coefs)>1e-8)),**dict(zip(names,coefs.tolist()))})
    membership.append({'repetition':rep+1,'fold_seed':fold_seed,'store_ids':subset.store_id.tolist()})
    if (rep+1)%20==0:
        print(f'Completed {rep+1} of {repeats} subsamples',flush=True)

fits=pd.DataFrame(records)
fits.to_csv('lasso_subsample_tuned_coefficients.csv',index=False)
direct={'digital_ad_spend':4,'promotion_index':70,'floor_area':7,'staff_hours':8,'local_demand_index':180}
noise={'store_code_index','display_color_index','survey_noise_index'}
with open('lasso_fivefold_validation_results.json') as f:
    full=json.load(f)['fits']['minimum_CV']['training_refit_coefficients_thousands_per_SD']
rows=[]
for name in names:
    coef=fits[name]
    count=int((coef.abs()>1e-8).sum())
    group='Direct equation term' if name in direct else ('Independent noise input' if name in noise else 'Correlated proxy')
    rows.append({'predictor':name,'selected_count':count,'selection_fraction':count/repeats,
                 'known_synthetic_role':group,'known_direct_coefficient_original_units':direct.get(name,0),
                 'selected_in_full_training_fit':bool(abs(full[name])>1e-8),
                 'coefficient_median_thousands_per_subsample_SD':float(coef.median())})
summary=pd.DataFrame(rows)
summary.to_csv('lasso_subsample_selection_frequencies.csv',index=False)
result={'dataset':'Original constructed sales, seed 4031','experiment_seed':SEED,
        'method':f'{repeats} overlapping subsamples of {size} stores drawn without replacement from 168 training stores. Each independently retunes minimum-mean-RMSE Lasso over the original 121 penalties with five-fold CV. Scaling occurs inside every fold. Final coefficients are refitted on that subsample.',
        'selection_threshold':1e-8,'selection_frequencies':rows,
        'selected_count_range':[int(fits.nonzero_count.min()),int(fits.nonzero_count.max())],
        'lambda_range':[float(fits['lambda'].min()),float(fits['lambda'].max())],
        'membership':membership,
        'limits':'Training-only subsampling sensitivity, not independent new datasets, validation, a formal stability-selection method, causal inference, or confidence probabilities. Changes combine subsample and CV fold variability. All reserved test rows unused.'}
with open('lasso_subsample_stability_results.json','w') as f:
    json.dump(result,f,indent=2)

palette={'Direct equation term':'#24689b','Correlated proxy':'#d68a29','Independent noise input':'#8a8a8a'}
fig,ax=plt.subplots(figsize=(11,7))
positions=np.arange(len(summary))
bars=ax.barh(positions,summary.selection_fraction*100,color=[palette[g] for g in summary.known_synthetic_role])
ax.set_yticks(positions,[s.replace('_',' ') for s in summary.predictor])
ax.invert_yaxis()
ax.set(xlim=(0,114),xlabel='Selected in tuned subsample fits (%)',
       title='Does Lasso select the same variables? Constructed sales data')
for bar,count in zip(bars,summary.selected_count):
    ax.text(bar.get_width()+1,bar.get_y()+bar.get_height()/2,f'{count}/{repeats}',va='center',fontsize=10)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=color,label=label) for label,color in palette.items()],
          loc='lower right',fontsize=9)
ax.grid(axis='x',alpha=.18)
ax.set_axisbelow(True)
fig.subplots_adjust(left=.27,right=.96,top=.90,bottom=.23)
fig.text(.06,.045,f'Training-only sensitivity: {repeats} subsamples of {size}/168 stores, seed 4031; penalty retuned within each subsample.\nColors identify roles in the known synthetic equation, not causal conclusions. Subsamples overlap.\nSelection fractions are descriptive frequencies, not probabilities of a real effect. Reserved test set unused.',fontsize=10)
fig.savefig('lasso_subsample_selection_stability.png',dpi=160)
plt.close(fig)
print(summary[['predictor','selected_count','known_synthetic_role','selected_in_full_training_fit']].to_string(index=False))
print(json.dumps({'selected_count_range':result['selected_count_range'],'lambda_range':result['lambda_range']}))
