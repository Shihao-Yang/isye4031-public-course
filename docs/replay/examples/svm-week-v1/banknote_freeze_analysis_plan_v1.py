from pathlib import Path
import json
import pandas as pd

comparison=pd.read_csv('banknote_joint_rbf_cv_comparison_v1.csv')
rbf=comparison.loc[comparison.model.eq('RBF SVM')].iloc[0]
assert float(rbf.C)==1.0 and float(rbf.gamma)==1.0
protocol=json.loads(Path('banknote_joint_rbf_cv_protocol_v1.json').read_text())
plan={
    'status':'Frozen before final holdout access; final evaluation not yet authorized or performed',
    'primary_model':{'name':'RBF SVM','estimator':'sklearn.svm.SVC','kernel':'rbf',
        'C':1.0,'gamma':1.0,'class_weight':None,'probability':False,'tol':0.001,
        'decision_threshold':0.0,'threshold_rule':'Use SVC.predict with its default zero-score decision boundary'},
    'features_in_order':['variance','skewness','kurtosis','entropy'],
    'preprocessing':{'transform':'StandardScaler','with_mean':True,'with_std':True,
        'fit_scope':'Fit once on all development rows only, then transform holdout with these saved parameters',
        'additional_transforms':None},
    'final_fit':'Refit each frozen pipeline on all 1029 development rows without further hyperparameter selection',
    'split_file':'banknote_split_roles_v1.csv',
    'row_and_group_policy':'Keep all raw rows; preserve existing row IDs, group IDs, and split roles; no regrouping or resplitting',
    'primary_metric':'balanced_accuracy_score: mean of class 0 recall and class 1 recall',
    'positive_class':1,'label_semantics':'Numeric class 0 and class 1 only; authentic/forged mapping unspecified',
    'secondary_metrics':['accuracy','class 1 recall','class 1 precision','confusion matrix with rows=true class and columns=predicted class; labels=[0,1]','error count'],
    'secondary_comparators':[
        {'name':'Linear SVM','preprocessing':'StandardScaler fitted only on development rows','kernel':'linear','C':100.0,'class_weight':None,'probability':False,'decision_threshold':0.0,'role':'Predeclared reference only; not used to choose a new winner on the final test'},
        {'name':'Logistic regression','preprocessing':'StandardScaler fitted only on development rows','C':1000.0,'l1_ratio':0,'solver':'lbfgs','max_iter':10000,'class_weight':None,'probability_threshold':0.5,'role':'Predeclared reference only; not used to choose a new winner on the final test'}],
    'selection_evidence':'banknote_joint_rbf_cv_comparison_v1.csv',
    'selection_rule':protocol['selection'],
    'test_policy':'Evaluate the frozen primary model and the two named secondary references once after explicit final-test request. Report all results, including unfavorable results. No tuning, threshold changes, feature selection, or resplitting based on final-test outcomes.',
    'post_test_changes':'Any revision motivated by final-test results is exploratory and requires new untouched data for an independent final assessment',
    'limits':['CV was used repeatedly for model selection and comparison; the winning CV score can be optimistic',
        'Exact duplicates are grouped, but unknown specimen or acquisition dependence is unresolved',
        'Historical benchmark results do not establish production performance',
        'No probability calibration or operational error costs have been selected'],
    'holdout_accessed':False
}
Path('banknote_frozen_final_evaluation_plan_v1.json').write_text(json.dumps(plan,indent=2)+'\n')
comparison.to_csv('banknote_frozen_development_comparison_v1.csv',index=False)
print(comparison[['model','C','gamma','mean_balanced_accuracy']].to_string(index=False))
print('Saved frozen plan. Final holdout not accessed.')
