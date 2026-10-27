from pathlib import Path
import os
os.environ['MPLCONFIGDIR'] = str(Path('.mplcache').resolve())
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score

roles=pd.read_csv('banknote_split_roles_v1.csv')
dev_roles=roles.loc[roles.split_role.eq('development')].copy()
dev_set=set(dev_roles.row_id)
features=['variance','skewness','kurtosis','entropy']
dev=pd.read_csv('banknote_raw.csv',header=None,names=features+['class'],
    skiprows=lambda i:i+1 not in dev_set)
X=dev[features].to_numpy()
y=dev['class'].to_numpy()
groups=dev_roles.group_id.to_numpy()
folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=4031).split(X,y,groups))
fold_id=np.full(len(y),-1)
for k,(train,valid) in enumerate(folds):
    assert not set(groups[train]).intersection(groups[valid])
    fold_id[valid]=k
pd.DataFrame({'row_id':dev_roles.row_id.to_numpy(),'group_id':groups,'validation_fold':fold_id}).to_csv('banknote_four_feature_cv_folds_v2.csv',index=False)
Cs=[.001,.01,.1,1.,10.,100.,1000.]
rows=[]
oofs={}
for name in ['Linear SVM','Logistic regression']:
    for C in (Cs if name == 'Linear SVM' else Cs+[10000.,100000.]):
        oof=np.full(len(y),-1)
        for k,(train,valid) in enumerate(folds):
            estimator=SVC(kernel='linear',C=C) if name=='Linear SVM' else LogisticRegression(C=C,l1_ratio=0,solver='lbfgs',max_iter=10000)
            pipeline=make_pipeline(StandardScaler(),estimator)
            pipeline.fit(X[train],y[train])
            pred=pipeline.predict(X[valid])
            oof[valid]=pred
            status=int(pipeline[-1].fit_status_) if name=='Linear SVM' else int(pipeline[-1].n_iter_.max()>=10000)
            rows.append({'model':name,'C':C,'fold':k,'training_rows':len(train),
                'validation_rows':len(valid),'balanced_accuracy':balanced_accuracy_score(y[valid],pred),
                'accuracy':accuracy_score(y[valid],pred),'class_1_recall':recall_score(y[valid],pred),
                'class_1_precision':precision_score(y[valid],pred,zero_division=0),
                'convergence_problem':status})
        oofs[(name,C)]=oof
fold_results=pd.DataFrame(rows)
fold_results.to_csv('banknote_four_feature_cv_fold_results_v2.csv',index=False)
summary=fold_results.groupby(['model','C'],sort=False).agg(
    mean_balanced_accuracy=('balanced_accuracy','mean'),
    sd_balanced_accuracy=('balanced_accuracy','std'),
    mean_accuracy=('accuracy','mean'),mean_class_1_recall=('class_1_recall','mean'),
    mean_class_1_precision=('class_1_precision','mean')).reset_index()
summary.to_csv('banknote_four_feature_cv_grid_v2.csv',index=False)
selected=[]
pred_rows=[]
for name in ['Linear SVM','Logistic regression']:
    candidates=summary.loc[summary.model.eq(name)].sort_values(['mean_balanced_accuracy','C'],ascending=[False,True])
    best=candidates.iloc[0]
    selected.append(best.to_dict())
    C=float(best.C)
    pred_rows.append(pd.DataFrame({'row_id':dev_roles.row_id.to_numpy(),'group_id':groups,
        'validation_fold':fold_id,'model':name,'selected_C':C,'class':y,'oof_prediction':oofs[(name,C)]}))
pd.DataFrame(selected).to_csv('banknote_four_feature_cv_selected_v2.csv',index=False)
pd.concat(pred_rows,ignore_index=True).to_csv('banknote_four_feature_cv_selected_oof_v2.csv',index=False)
plt.rcParams.update({'font.size':12,'axes.titlesize':14})
fig,axes=plt.subplots(1,2,figsize=(11.5,4.8),sharey=True,constrained_layout=True)
for ax,name,color in zip(axes,['Linear SVM','Logistic regression'],['#0072B2','#D55E00']):
    s=summary.loc[summary.model.eq(name)]
    for k in range(5):
        f=fold_results.loc[fold_results.model.eq(name)&fold_results.fold.eq(k)]
        ax.plot(f.C,f.balanced_accuracy,color=color,alpha=.2,lw=1)
    ax.plot(s.C,s.mean_balanced_accuracy,'o-',color=color,lw=2,label='Mean of five validation folds')
    best=next(r for r in selected if r['model']==name)
    ax.scatter([best['C']],[best['mean_balanced_accuracy']],s=170,facecolors='none',edgecolors='black',lw=1.5,zorder=5,label='Highest mean; ties use smaller C')
    ax.set(xscale='log',xlabel='C (log scale)',title=name)
    ax.grid(alpha=.2)
axes[0].set_ylabel('Validation balanced accuracy')
axes[0].legend(loc='lower right',fontsize=9)
axes[1].text(.04,.05,'Thin lines: individual folds',transform=axes[1].transAxes,fontsize=10)
fig.suptitle('Four predictors: development cross-validation only',fontsize=16)
fig.savefig('banknote_four_feature_cv_v2.png',dpi=180)
plt.close(fig)
Path('banknote_four_feature_cv_protocol_v2.json').write_text(json.dumps({
    'features':features,'development_rows':len(y),'unique_development_groups':len(set(groups)),
    'splitter':'StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=4031)',
    'scaling':'StandardScaler fit on training rows inside every fold',
    'primary_metric':'mean validation balanced accuracy',
    'selection_rule':'Maximum mean balanced accuracy for each model; exact ties use smaller C',
    'C_grid':{'Linear SVM':Cs,'Logistic regression':Cs+[10000.,100000.]},'logistic_threshold':0.5,'svm_threshold':0,
    'class_weight':None,'positive_class':1,
    'convergence_problems':int(fold_results.convergence_problem.sum()),
    'interpretation':'Model-selection CV, not independent performance assessment; no final holdout evaluated'
},indent=2)+'\n')
print(pd.DataFrame(selected).to_string(index=False))
print('Convergence problems:',int(fold_results.convergence_problem.sum()))
print(summary[['model','C','mean_balanced_accuracy']].to_string(index=False))
