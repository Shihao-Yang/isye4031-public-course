from pathlib import Path
import json
import pandas as pd

audit=json.loads(Path('banknote_data_audit_v1.json').read_text())
selected=pd.read_csv('banknote_four_feature_cv_selected_v2.csv')
selected['mean_balanced_accuracy_percent']=100*selected.mean_balanced_accuracy
selected['fold_sd_percentage_points']=100*selected.sd_balanced_accuracy
selected.to_csv('banknote_linear_recap_results_v1.csv',index=False)
scores=selected.set_index('model').mean_balanced_accuracy
recap={
    'total_rows':audit['raw_rows'],
    'development_rows':audit['development_rows'],
    'reserved_holdout_rows':audit['reserved_holdout_rows'],
    'development_class_counts':audit['development_class_counts'],
    'logistic_minus_svm_balanced_accuracy_percentage_points':float(100*(scores['Logistic regression']-scores['Linear SVM'])),
    'data_figure':'banknote_development_scatter_v1.png',
    'source_results':'banknote_four_feature_cv_selected_v2.csv',
    'evaluation_status':'development model-selection cross-validation; final holdout unused'
}
Path('banknote_linear_recap_v1.json').write_text(json.dumps(recap,indent=2)+'\n')
print(selected[['model','C','mean_balanced_accuracy_percent','fold_sd_percentage_points']].to_string(index=False))
print(json.dumps(recap,indent=2))
