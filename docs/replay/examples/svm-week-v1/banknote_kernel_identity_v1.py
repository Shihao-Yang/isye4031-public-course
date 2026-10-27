from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

roles=pd.read_csv('banknote_split_roles_v1.csv')
ids=roles.loc[roles.split_role.eq('development'),'row_id'].to_numpy()
idset=set(ids)
dev=pd.read_csv('banknote_raw.csv',header=None,
 names=['variance','skewness','kurtosis','entropy','class'],skiprows=lambda i:i+1 not in idset)
Z=StandardScaler().fit_transform(dev[['variance','skewness']])
y=dev['class'].to_numpy()
# This weighted map corresponds exactly to K(z,u)=(1+z dot u)^2.
# It differs from the independently scaled expansion in the previous illustration.
def phi(z):
    a,b=z[:,0],z[:,1]
    return np.column_stack([np.ones(len(z)),np.sqrt(2)*a,np.sqrt(2)*b,a*a,np.sqrt(2)*a*b,b*b])
P=phi(Z)
K=(1+Z@Z.T)**2
explicit_gram=P@P.T
kernel_model=SVC(kernel='poly',degree=2,gamma=1,coef0=1,C=1,tol=1e-8)
explicit_model=SVC(kernel='linear',C=1,tol=1e-8)
kernel_model.fit(Z,y)
explicit_model.fit(P,y)
score_k=kernel_model.decision_function(Z)
score_e=explicit_model.decision_function(P)
# Reconstruct the kernel decision function from only support vectors and signed dual weights.
score_sum=(1+Z@kernel_model.support_vectors_.T)**2@kernel_model.dual_coef_[0]+kernel_model.intercept_[0]
result={
 'kernel':'(1 + dot(z,u))**2; degree=2, gamma=1, coef0=1',
 'explicit_map':['1','sqrt(2)*z1','sqrt(2)*z2','z1**2','sqrt(2)*z1*z2','z2**2'],
 'example_row_ids':ids[:2].tolist(),
 'example_z':Z[0].tolist(),'example_u':Z[1].tolist(),
 'example_kernel_value':float(K[0,1]),'example_explicit_dot_product':float(explicit_gram[0,1]),
 'max_gram_identity_absolute_error':float(np.max(np.abs(K-explicit_gram))),
 'kernel_vs_explicit_max_score_difference':float(np.max(np.abs(score_k-score_e))),
 'kernel_vs_explicit_prediction_disagreements':int(np.sum(kernel_model.predict(Z)!=explicit_model.predict(P))),
 'support_vector_score_reconstruction_max_error':float(np.max(np.abs(score_sum-score_k))),
 'kernel_training_errors':int(np.sum(kernel_model.predict(Z)!=y)),
 'explicit_training_errors':int(np.sum(explicit_model.predict(P)!=y)),
 'kernel_support_vectors':len(kernel_model.support_),
 'solver_fit_status':[int(kernel_model.fit_status_),int(explicit_model.fit_status_)],
 'scope':'Two-feature development training identity check; not CV or final test',
 'distinction':'Weighted polynomial map differs from previous independently scaled quadratic map; C=1 fixed'
}
Path('banknote_kernel_identity_results_v1.json').write_text(json.dumps(result,indent=2)+'\n')
pd.DataFrame({'row_id':ids,'class':y,'kernel_score':score_k,'explicit_score':score_e,'reconstructed_kernel_score':score_sum}).to_csv('banknote_kernel_identity_scores_v1.csv',index=False)
print(json.dumps(result,indent=2))
