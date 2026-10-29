import json
from pathlib import Path
import numpy as np
import pandas as pd

# Assigned model outputs only. No observed outcomes, fitting, or evaluation.
logits = np.array([-2.0, -0.4, 0.4, 1.4])
svm_scores = np.array([-1.2, 0.1, 0.6, 1.3])
probabilities = 1 / (1 + np.exp(-logits))
example = pd.DataFrame({'hypothetical_store':list('ABCD'), 'logistic_score':logits,
    'logistic_probability_miss':probabilities,
    'logistic_label_threshold_0_5':(probabilities >= .5).astype(int),
    'logistic_label_threshold_0_3':(probabilities >= .3).astype(int),
    'svm_score':svm_scores, 'svm_label_threshold_0':(svm_scores >= 0).astype(int)})
example.to_csv('store_sales_05_hypothetical_model_outputs.csv',index=False)
Path('store_sales_05_illustration_notes.json').write_text(json.dumps({
    'purpose':'Conceptual score illustration, not a new fitted classification dataset',
    'positive_class':'Store misses its sales target',
    'logistic_link':'p = 1/(1+exp(-eta))',
    'svm_orientation':'Positive score denotes predicted miss',
    'scores':'Independently assigned hypothetical logistic and SVM outputs; not estimated from data',
    'observed_labels_available':False,'performance_estimated':False,
    'store_sales_regression_data_used':False,'reserved_test_used':False},indent=2))
print(example.to_string(index=False,float_format=lambda v:f'{v:.4f}'))
