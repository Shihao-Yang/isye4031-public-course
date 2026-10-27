from pathlib import Path
import json
import numpy as np
import pandas as pd

summary=pd.read_csv('banknote_four_feature_cv_grid_v2.csv')
folds=pd.read_csv('banknote_four_feature_cv_fold_results_v2.csv')
svm=summary.loc[summary.model.eq('Linear SVM')]
best=svm.sort_values(['mean_balanced_accuracy','C'],ascending=[False,True]).iloc[0]
largest=svm.sort_values('C').iloc[-1]
paired=folds.loc[folds.model.eq('Linear SVM') & folds.C.isin([best.C,largest.C])].pivot(index='fold',columns='C',values='balanced_accuracy')
paired.columns=[f'C_{c:g}' for c in paired.columns]
paired['largest_minus_selected']=paired[f'C_{largest.C:g}']-paired[f'C_{best.C:g}']
paired.to_csv('banknote_agent_claim_C_fold_check_v1.csv')

# Constructed score-scale examples, not banknote observations and not calibrated models.
scores=np.array([-1.2,-0.2,0.0,0.8,1.5])
example=pd.DataFrame({'constructed_score':scores,'score_times_10':10*scores,
    'positive_region':scores>0,'positive_region_after_times_10':10*scores>0,
    'naive_sigmoid_of_score':1/(1+np.exp(-scores)),
    'naive_sigmoid_of_rescaled_score':1/(1+np.exp(-10*scores))})
example.to_csv('svm_score_scale_constructed_check_v1.csv',index=False)
result={
    'svm_selected_C':float(best.C), 'svm_selected_mean_balanced_accuracy':float(best.mean_balanced_accuracy),
    'svm_largest_tested_C':float(largest.C),'svm_largest_C_mean_balanced_accuracy':float(largest.mean_balanced_accuracy),
    'largest_minus_selected_percentage_points':float(100*(largest.mean_balanced_accuracy-best.mean_balanced_accuracy)),
    'paired_folds_largest_better':int((paired.largest_minus_selected>0).sum()),
    'paired_folds_largest_equal':int((paired.largest_minus_selected==0).sum()),
    'paired_folds_largest_worse':int((paired.largest_minus_selected<0).sum()),
    'sigmoid_of_0_8':float(1/(1+np.exp(-.8))),
    'sigmoid_of_8':float(1/(1+np.exp(-8))),
    'score_example_status':'Constructed algebra examples only; sigmoid values are not calibrated SVM probabilities',
    'holdout_accessed':False
}
Path('banknote_agent_claim_checks_v1.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
print(paired.to_string())
