from pathlib import Path
import os
os.environ['MPLCONFIGDIR']=str(Path('.mplcache').resolve())
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score,accuracy_score,recall_score,precision_score

roles=pd.read_csv('banknote_split_roles_v1.csv')
dr=roles.loc[roles.split_role.eq('development')].copy()
ids=dr.row_id.to_numpy(); keep=set(ids)
features=['variance','skewness','kurtosis','entropy']
dev=pd.read_csv('banknote_raw.csv',header=None,names=features+['class'],skiprows=lambda i:i+1 not in keep)
X=dev[features].to_numpy(); y=dev['class'].to_numpy()
foldtable=pd.read_csv('banknote_four_feature_cv_folds_v2.csv')
assert np.array_equal(foldtable.row_id.to_numpy(),ids)
assert np.array_equal(foldtable.group_id.to_numpy(),dr.group_id.to_numpy())
fold_ids=foldtable.validation_fold.to_numpy(); groups=dr.group_id.to_numpy()
Cs=[.01,.1,1.,10.,100.,1000.]
gammas=[.001,.01,.1,1.,10.,100.]
records=[]; predictions={}
for C in Cs:
    for gamma in gammas:
        oof=np.full(len(y),-1)
        for fold in range(5):
            train=fold_ids!=fold; valid=~train
            assert not set(groups[train]).intersection(groups[valid])
            model=make_pipeline(StandardScaler(),SVC(kernel='rbf',C=C,gamma=gamma))
            model.fit(X[train],y[train]); pred=model.predict(X[valid]); oof[valid]=pred
            records.append({'model':'RBF SVM','C':C,'gamma':gamma,'fold':fold,
                'balanced_accuracy':balanced_accuracy_score(y[valid],pred),
                'accuracy':accuracy_score(y[valid],pred),
                'class_1_recall':recall_score(y[valid],pred),
                'class_1_precision':precision_score(y[valid],pred,zero_division=0),
                'training_balanced_accuracy':balanced_accuracy_score(y[train],model.predict(X[train])),
                'solver_fit_status':int(model[-1].fit_status_)})
        predictions[(C,gamma)]=oof
folds=pd.DataFrame(records)
folds.to_csv('banknote_joint_rbf_cv_fold_results_v1.csv',index=False)
summary=folds.groupby(['C','gamma'],sort=False).agg(
    mean_balanced_accuracy=('balanced_accuracy','mean'),sd_balanced_accuracy=('balanced_accuracy','std'),
    mean_accuracy=('accuracy','mean'),mean_class_1_recall=('class_1_recall','mean'),
    mean_class_1_precision=('class_1_precision','mean'),
    mean_training_balanced_accuracy=('training_balanced_accuracy','mean')).reset_index()
summary.to_csv('banknote_joint_rbf_cv_grid_v1.csv',index=False)
# Exact ties: smaller C, then smaller gamma. This is deterministic, not a universal complexity ordering.
best=summary.sort_values(['mean_balanced_accuracy','C','gamma'],ascending=[False,True,True]).iloc[0]
bestC,bestgamma=float(best.C),float(best.gamma)
rbf=best.to_dict(); rbf['model']='RBF SVM'
linear=pd.read_csv('banknote_four_feature_cv_selected_v2.csv')
comparison=pd.concat([linear,pd.DataFrame([rbf])],ignore_index=True)
comparison.to_csv('banknote_joint_rbf_cv_comparison_v1.csv',index=False)
rbf_folds=folds.loc[folds.C.eq(bestC)&folds.gamma.eq(bestgamma)].copy()
linear_folds=pd.read_csv('banknote_four_feature_cv_fold_results_v2.csv')
chosen=[]
for _,r in linear.iterrows():
    chosen.append(linear_folds.loc[linear_folds.model.eq(r.model)&linear_folds.C.eq(r.C)])
chosen.append(rbf_folds)
pd.concat(chosen,ignore_index=True).to_csv('banknote_joint_rbf_cv_selected_folds_v1.csv',index=False)
pd.DataFrame({'row_id':ids,'group_id':groups,'validation_fold':fold_ids,'class':y,
    'selected_C':bestC,'selected_gamma':bestgamma,
    'oof_prediction':predictions[(bestC,bestgamma)]}).to_csv('banknote_joint_rbf_cv_selected_oof_v1.csv',index=False)

plt.rcParams.update({'font.size':12})
matrix=summary.pivot(index='C',columns='gamma',values='mean_balanced_accuracy').reindex(index=Cs,columns=gammas)
fig,ax=plt.subplots(figsize=(9,6.3),constrained_layout=True)
im=ax.imshow(100*matrix.to_numpy(),cmap='viridis',vmin=50,vmax=100,aspect='auto')
ax.set_xticks(range(len(gammas)),[f'{g:g}' for g in gammas])
ax.set_yticks(range(len(Cs)),[f'{c:g}' for c in Cs])
for i in range(len(Cs)):
    for j in range(len(gammas)):
        value=float(matrix.iloc[i,j])*100
        ax.text(j,i,f'{value:.2f}',ha='center',va='center',fontsize=11,color='black' if value>80 else 'white')
from matplotlib.patches import Rectangle
ax.add_patch(Rectangle((gammas.index(bestgamma)-.48,Cs.index(bestC)-.48),.96,.96,fill=False,edgecolor='#D55E00',lw=3))
ax.set(xlabel='Gamma',ylabel='C',title='RBF SVM: mean validation balanced accuracy (%)\nFour predictors; same five development folds')
fig.colorbar(im,ax=ax,label='Balanced accuracy (%)',shrink=.85)
fig.savefig('banknote_joint_rbf_cv_heatmap_v1.png',dpi=180); plt.close(fig)
protocol={'features':features,'development_rows':len(y),'C_grid':Cs,'gamma_grid':gammas,
    'folds':'Saved StratifiedGroupKFold(5, shuffle=True, random_state=4031) development folds, grouped by exact feature vector',
    'pipeline':'StandardScaler fit inside each training fold, then SVC',
    'selection':'Highest mean validation balanced accuracy; exact ties use smaller C, then smaller gamma',
    'selected_C':bestC,'selected_gamma':bestgamma,
    'number_of_exact_top_ties':int(summary.mean_balanced_accuracy.eq(best.mean_balanced_accuracy).sum()),
    'solver_problems':int(folds.solver_fit_status.sum()),
    'scope':'Model selection CV; the same folds select and compare settings, so maxima can be optimistic. Not nested CV.',
    'holdout_accessed':False}
Path('banknote_joint_rbf_cv_protocol_v1.json').write_text(json.dumps(protocol,indent=2)+'\n')
print(comparison[['model','C','gamma','mean_balanced_accuracy','sd_balanced_accuracy','mean_class_1_recall','mean_class_1_precision']].to_string(index=False))
print(json.dumps(protocol,indent=2))
print('Top grid cells:')
print(summary.sort_values(['mean_balanced_accuracy','C','gamma'],ascending=[False,True,True]).head(8).to_string(index=False))
