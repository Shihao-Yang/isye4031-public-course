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
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score

roles=pd.read_csv('banknote_split_roles_v1.csv')
dev_ids=roles.loc[roles.split_role.eq('development'),'row_id'].to_numpy()
dev_set=set(dev_ids)
features=['variance','skewness']
dev=pd.read_csv('banknote_raw.csv',header=None,
    names=features+['kurtosis','entropy','class'],skiprows=lambda i:i+1 not in dev_set)
X=dev[features].to_numpy()
y=dev['class'].to_numpy()
# The first scaling defines z. The second scales expanded columns for fitting.
models={
    'Linear features':make_pipeline(StandardScaler(),SVC(kernel='linear',C=1)),
    'Quadratic features':make_pipeline(StandardScaler(),PolynomialFeatures(degree=2,include_bias=False),StandardScaler(),SVC(kernel='linear',C=1))
}
results=[]
predictions=[]
for name,model in models.items():
    model.fit(X,y)
    pred=model.predict(X)
    results.append({'model':name,'C':1,'features_used':features,
        'training_errors':int(np.sum(pred!=y)),
        'training_balanced_accuracy':float(balanced_accuracy_score(y,pred)),
        'support_vectors':int(len(model[-1].support_)),'solver_fit_status':int(model[-1].fit_status_)})
    predictions.append(pd.DataFrame({'row_id':dev_ids,'model':name,'class':y,'score':model.decision_function(X),'training_prediction':pred}))
quad=models['Quadratic features']
expanded_scaler=quad[2]
# Undo only the expanded-column scaling to express f in z coordinates.
beta=quad[-1].coef_[0]/expanded_scaler.scale_
b0=float(quad[-1].intercept_[0]-np.dot(quad[-1].coef_[0],expanded_scaler.mean_/expanded_scaler.scale_))
term_names=['z_variance','z_skewness','z_variance_squared','z_variance_times_z_skewness','z_skewness_squared']
coeffs={'intercept':b0,**dict(zip(term_names,map(float,beta)))}
zscaler=quad[0]
Z=zscaler.transform(X)
zx,zy=np.meshgrid(np.linspace(Z[:,0].min()-.25,Z[:,0].max()+.25,400),np.linspace(Z[:,1].min()-.25,Z[:,1].max()+.25,400))
zgrid=np.c_[zx.ravel(),zy.ravel()]
xgrid=zscaler.inverse_transform(zgrid)
plt.rcParams.update({'font.size':12,'axes.titlesize':14})
fig,axes=plt.subplots(1,2,figsize=(12,5.6),sharex=True,sharey=True,constrained_layout=True)
for ax,(name,model),result in zip(axes,models.items(),results):
    score=model.decision_function(xgrid).reshape(zx.shape)
    ax.contourf(zx,zy,score,levels=[-1e9,0,1e9],colors=['#0072B2','#D55E00'],alpha=.09)
    for c,color,marker in [(0,'#0072B2','o'),(1,'#D55E00','^')]:
        m=y==c
        ax.scatter(Z[m,0],Z[m,1],c=color,marker=marker,s=17,alpha=.5,linewidths=0,label=f'Class {c}')
    ax.contour(zx,zy,score,levels=[0],colors=['black'],linewidths=2)
    ax.set(xlabel='Standardized variance',title=f'{name}\n{result["training_errors"]} training errors',
           xlim=(zx.min(),zx.max()),ylim=(zy.min(),zy.max()))
    ax.set_aspect('equal',adjustable='box')
    ax.grid(alpha=.15)
axes[0].set_ylabel('Standardized skewness')
axes[0].legend(loc='lower left',fontsize=10)
fig.suptitle('Linear in coefficients, curved in the original predictors\nTwo-feature development training illustration; fixed C = 1',fontsize=16)
fig.savefig('banknote_quadratic_boundary_v1.png',dpi=180)
plt.close(fig)
pd.DataFrame(results).to_csv('banknote_quadratic_boundary_results_v1.csv',index=False)
pd.concat(predictions,ignore_index=True).to_csv('banknote_quadratic_boundary_predictions_v1.csv',index=False)
Path('banknote_quadratic_boundary_coefficients_v1.json').write_text(json.dumps({
    'coefficients_in_standardized_original_predictors':coeffs,
    'original_predictor_means':zscaler.mean_.tolist(),
    'original_predictor_scales':zscaler.scale_.tolist(),
    'expanded_features':term_names,
    'purpose':'Two-feature development training geometry, not four-feature CV or final-test results',
    'pipeline':'Scale original predictors; expand degree 2 without constant; scale expanded columns; linear SVC(C=1)',
    'holdout_accessed':False},indent=2)+'\n')
print(pd.DataFrame(results).to_string(index=False))
print(json.dumps(coeffs,indent=2))
# Check that reported coefficients reconstruct the fitted scores.
reconstructed=b0+quad[1].transform(Z)@beta
print('Coefficient reconstruction maximum difference:',float(np.max(np.abs(reconstructed-quad.decision_function(X)))))
