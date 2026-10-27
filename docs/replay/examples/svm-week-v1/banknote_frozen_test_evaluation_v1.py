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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,accuracy_score,recall_score,precision_score,confusion_matrix

plan=json.loads(Path('banknote_frozen_final_evaluation_plan_v1.json').read_text())
roles=pd.read_csv(plan['split_file'])
features=plan['features_in_order']
raw=pd.read_csv('banknote_raw.csv',header=None,names=features+['class'])
assert len(raw)==len(roles)
train=roles.split_role.eq('development').to_numpy()
test=roles.split_role.eq('final_holdout').to_numpy()
assert not set(roles.loc[train,'group_id']).intersection(roles.loc[test,'group_id'])
Xdev=raw.loc[train,features]; ydev=raw.loc[train,'class']
Xtest=raw.loc[test,features]; ytest=raw.loc[test,'class'].to_numpy()
models={
    'Linear SVM':make_pipeline(StandardScaler(),SVC(kernel='linear',C=100,tol=.001)),
    'Logistic regression':make_pipeline(StandardScaler(),LogisticRegression(C=1000,l1_ratio=0,solver='lbfgs',max_iter=10000)),
    'RBF SVM':make_pipeline(StandardScaler(),SVC(kernel='rbf',C=1,gamma=1,tol=.001))
}
records=[]; prediction_rows=[]; preds={}; cms={}; scales={}
for name,model in models.items():
    model.fit(Xdev,ydev)
    pred=model.predict(Xtest)
    preds[name]=pred
    cm=confusion_matrix(ytest,pred,labels=[0,1]); cms[name]=cm
    records.append({'model':name,'test_rows':len(ytest),
        'balanced_accuracy':balanced_accuracy_score(ytest,pred),
        'accuracy':accuracy_score(ytest,pred),
        'class_0_recall':recall_score(ytest,pred,pos_label=0),
        'class_1_recall':recall_score(ytest,pred,pos_label=1),
        'class_1_precision':precision_score(ytest,pred,pos_label=1,zero_division=0),
        'errors':int(np.sum(pred!=ytest)),
        'true_0_pred_0':int(cm[0,0]),'true_0_pred_1':int(cm[0,1]),
        'true_1_pred_0':int(cm[1,0]),'true_1_pred_1':int(cm[1,1])})
    prediction_rows.append(pd.DataFrame({'row_id':roles.loc[test,'row_id'].to_numpy(),
        'group_id':roles.loc[test,'group_id'].to_numpy(),'model':name,'class':ytest,
        'prediction':pred,'decision_score':model.decision_function(Xtest)}))
    scales[name]={'mean':model[0].mean_.tolist(),'scale':model[0].scale_.tolist()}
pd.DataFrame(records).to_csv('banknote_frozen_test_metrics_v1.csv',index=False)
pd.concat(prediction_rows,ignore_index=True).to_csv('banknote_frozen_test_predictions_v1.csv',index=False)
pairs=[]
for reference in ['Linear SVM','Logistic regression']:
    ref_correct=preds[reference]==ytest; rbf_correct=preds['RBF SVM']==ytest
    pairs.append({'reference':reference,'both_correct':int(np.sum(ref_correct&rbf_correct)),
        'rbf_correct_reference_wrong':int(np.sum(rbf_correct&~ref_correct)),
        'reference_correct_rbf_wrong':int(np.sum(ref_correct&~rbf_correct)),
        'both_wrong':int(np.sum(~ref_correct&~rbf_correct))})
pd.DataFrame(pairs).to_csv('banknote_frozen_test_paired_errors_v1.csv',index=False)
plt.rcParams.update({'font.size':12})
fig,axes=plt.subplots(1,3,figsize=(11,4),constrained_layout=True)
for ax,name in zip(axes,models):
    cm=cms[name]
    ax.imshow(cm,cmap='Blues',vmin=0,vmax=max(np.max(c) for c in cms.values()))
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=20,color='white' if cm[i,j]>100 else 'black')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['0','1'],yticklabels=['0','1'],
        xlabel='Predicted class',title=name)
axes[0].set_ylabel('True class')
fig.suptitle('Reserved test set: frozen four-predictor models',fontsize=16)
fig.savefig('banknote_frozen_test_confusion_matrices_v1.png',dpi=180); plt.close(fig)
result={'status':'Final test evaluated after explicit user request; no settings changed',
    'frozen_plan':'banknote_frozen_final_evaluation_plan_v1.json',
    'development_rows':int(train.sum()),'test_rows':int(test.sum()),
    'test_class_counts':{str(k):int(v) for k,v in pd.Series(ytest).value_counts().sort_index().items()},
    'test_unique_groups':int(roles.loc[test,'group_id'].nunique()),
    'primary_model':'RBF SVM','metrics':records,'paired_errors':pairs,
    'development_fitted_scalers':scales,
    'limitations':['Single historical test split; selection and interpretation must not be revised using these outcomes',
        'Exact duplicate groups isolated, but unknown acquisition/specimen dependence unresolved',
        'No probability calibration tested; no operational error costs or external-population performance established']}
Path('banknote_frozen_test_results_v1.json').write_text(json.dumps(result,indent=2)+'\n')
print(pd.DataFrame(records).to_string(index=False))
print(pd.DataFrame(pairs).to_string(index=False))
print('Test class counts:',result['test_class_counts'])
print('Unique test groups:',result['test_unique_groups'])
