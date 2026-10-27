from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

roles = pd.read_csv('banknote_split_roles_v1.csv')
dev_rows = set(roles.loc[roles.split_role == 'development', 'row_id'])
dev = pd.read_csv('banknote_raw.csv', header=None,
                  names=['variance','skewness','kurtosis','entropy','class'],
                  skiprows=lambda i: i+1 not in dev_rows)
X = StandardScaler().fit_transform(dev[['variance','skewness']])
y = dev['class'].to_numpy()
t = 2*y-1
records = []
for C in [0.01,0.1,1.0,10.0]:
    model = SVC(kernel='linear', C=C)
    model.fit(X,y)
    f = model.decision_function(X)
    signed_margin = t*f
    hinge = np.maximum(0,1-signed_margin)
    norm = float(np.linalg.norm(model.coef_[0]))
    records.append({'C':C,'training_errors':int(np.sum(model.predict(X)!=y)),
                    'correct_but_inside_margin':int(np.sum((signed_margin>0)&(signed_margin<1-1e-6))),
                    'total_hinge_loss':float(hinge.sum()),
                    'full_margin_width':2/norm,
                    'regularization_term':0.5*norm**2,
                    'objective':0.5*norm**2+C*float(hinge.sum())})
summary = pd.DataFrame(records)
summary.to_csv('banknote_soft_margin_tradeoff_v1.csv',index=False)
examples = pd.DataFrame({'signed_score':[1.5,1.0,0.5,0.0,-0.5,-2.0]})
examples['slack_or_hinge_loss'] = np.maximum(0,1-examples.signed_score)
examples.to_csv('soft_margin_constructed_score_examples_v1.csv',index=False)
Path('banknote_soft_margin_tradeoff_context_v1.json').write_text(json.dumps({
    'purpose':'Fixed-C two-feature development training illustration; no tuning, CV, or final test',
    'features':['variance','skewness'], 'development_rows':len(dev),
    'positive_signed_label':'class 1',
    'objective':'0.5 * ||w||^2 + C * sum(max(0, 1 - t_i * f_i))',
    'score_examples':'Constructed algebra examples, not individual banknote observations'
},indent=2)+'\n')
print(summary.to_string(index=False))
print(examples.to_string(index=False))
