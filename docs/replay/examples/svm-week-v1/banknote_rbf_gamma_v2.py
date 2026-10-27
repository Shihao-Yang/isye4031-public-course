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
from sklearn.metrics import balanced_accuracy_score

roles=pd.read_csv('banknote_split_roles_v1.csv')
dr=roles.loc[roles.split_role.eq('development')].copy()
ids=dr.row_id.to_numpy(); keep=set(ids)
features=['variance','skewness','kurtosis','entropy']
dev=pd.read_csv('banknote_raw.csv',header=None,names=features+['class'],skiprows=lambda i:i+1 not in keep)
y=dev['class'].to_numpy(); X=dev[features].to_numpy()
foldtable=pd.read_csv('banknote_four_feature_cv_folds_v2.csv')
assert np.array_equal(foldtable.row_id.to_numpy(),ids)
fold_ids=foldtable.validation_fold.to_numpy()
assert np.array_equal(foldtable.group_id.to_numpy(),dr.group_id.to_numpy())
gammas=[.01,.1,1.,10.,100.]
C=1.
rows=[]
for gamma in gammas:
    for fold in range(5):
        train=fold_ids!=fold; valid=~train
        assert not set(dr.group_id.to_numpy()[train]).intersection(dr.group_id.to_numpy()[valid])
        model=make_pipeline(StandardScaler(),SVC(kernel='rbf',C=C,gamma=gamma))
        model.fit(X[train],y[train])
        rows.append({'gamma':gamma,'C':C,'fold':fold,
            'training_balanced_accuracy':balanced_accuracy_score(y[train],model.predict(X[train])),
            'validation_balanced_accuracy':balanced_accuracy_score(y[valid],model.predict(X[valid])),
            'support_vectors':len(model[-1].support_),'solver_fit_status':int(model[-1].fit_status_)})
f=pd.DataFrame(rows)
f.to_csv('banknote_rbf_gamma_four_feature_folds_v2.csv',index=False)
s=f.groupby('gamma',sort=False).agg(mean_training_balanced_accuracy=('training_balanced_accuracy','mean'),
    mean_validation_balanced_accuracy=('validation_balanced_accuracy','mean'),
    sd_validation_balanced_accuracy=('validation_balanced_accuracy','std')).reset_index()
s.to_csv('banknote_rbf_gamma_four_feature_summary_v2.csv',index=False)

# Separate two-feature geometry: never a slice of a four-feature fitted model.
scaler=StandardScaler(); Z=scaler.fit_transform(dev[features[:2]])
xx,yy=np.meshgrid(np.linspace(Z[:,0].min()-.25,Z[:,0].max()+.25,350),np.linspace(Z[:,1].min()-.25,Z[:,1].max()+.25,350))
grid=np.c_[xx.ravel(),yy.ravel()]
plt.rcParams.update({'font.size':12,'axes.titlesize':13})
fig,axes=plt.subplots(1,3,figsize=(14,6.5),sharex=True,sharey=True)
geom=[]
for ax,gamma in zip(axes,[.01,1.,100.]):
    model=SVC(kernel='rbf',C=C,gamma=gamma).fit(Z,y)
    scores=model.decision_function(grid).reshape(xx.shape)
    ax.contourf(xx,yy,scores,levels=[-1e9,0,1e9],colors=['#0072B2','#D55E00'],alpha=.12)
    ax.contour(xx,yy,scores,levels=[0],colors='black',linewidths=1.6)
    for c,color,marker in [(0,'#0072B2','o'),(1,'#D55E00','^')]:
        m=y==c
        ax.scatter(Z[m,0],Z[m,1],c=color,marker=marker,s=12,alpha=.45,linewidths=0,label=f'Class {c}')
    error=int(np.sum(model.predict(Z)!=y))
    geom.append({'gamma':gamma,'C':C,'training_errors':error,'support_vectors':len(model.support_)})
    ax.set(title=f'Gamma = {gamma:g}\n{error} training errors',xlabel='Standardized variance',
        xlim=(xx.min(),xx.max()),ylim=(yy.min(),yy.max()))
    ax.set_aspect('equal',adjustable='box'); ax.grid(alpha=.15)
axes[0].set_ylabel('Standardized skewness'); axes[0].legend(loc='lower left',fontsize=10)
fig.suptitle('RBF boundaries using only variance and skewness\nDevelopment training illustrations; fixed C = 1',fontsize=16)
fig.subplots_adjust(top=.77,bottom=.13,left=.06,right=.99,wspace=.12)
fig.savefig('banknote_rbf_gamma_boundaries_v2.png',dpi=170); plt.close(fig)
pd.DataFrame(geom).to_csv('banknote_rbf_gamma_two_feature_geometry_v2.csv',index=False)

fig,axes=plt.subplots(1,2,figsize=(11.7,4.8),constrained_layout=True)
d=np.linspace(0,3,300)
for gamma,color in zip([.01,1.,100.],['#0072B2','#009E73','#D55E00']):
    axes[0].plot(d,np.exp(-gamma*d*d),label=f'Gamma = {gamma:g}',color=color,lw=2)
axes[0].set(xlabel='Distance between standardized observations',ylabel='RBF similarity K(z, u)',title='Larger gamma makes similarity more local')
axes[0].legend(); axes[0].grid(alpha=.2)
for k in range(5):
    part=f.loc[f.fold.eq(k)]
    axes[1].plot(part.gamma,part.validation_balanced_accuracy,color='#0072B2',alpha=.2,lw=1)
axes[1].plot(s.gamma,s.mean_training_balanced_accuracy,'s--',color='#777777',label='Mean training')
axes[1].plot(s.gamma,s.mean_validation_balanced_accuracy,'o-',color='#0072B2',lw=2,label='Mean validation')
axes[1].set(xscale='log',xlabel='Gamma (log scale)',ylabel='Balanced accuracy',title='Four predictors: five-fold development CV',ylim=(.45,1.02))
axes[1].legend(loc='lower left'); axes[1].grid(alpha=.2)
fig.suptitle('RBF similarity and prediction at fixed C = 1',fontsize=16)
fig.savefig('banknote_rbf_gamma_similarity_cv_v2.png',dpi=170); plt.close(fig)
Path('banknote_rbf_gamma_protocol_v2.json').write_text(json.dumps({
 'C':C,'gammas':gammas,'four_feature_validation':'Same saved StratifiedGroupKFold(5, shuffle=True, random_state=4031) development folds; StandardScaler fit inside training folds',
 'geometry':'Separate two-feature models fitted to standardized variance and skewness on all development rows',
 'kernel':'exp(-gamma * squared_Euclidean_distance)',
 'radius_at_similarity_exp_minus_1':{str(g):float(1/np.sqrt(g)) for g in [.01,1.,100.]},
 'scope':'One gamma sweep at fixed C; model-selection evidence, not complete joint tuning or final-test performance',
 'convergence_problems':int(f.solver_fit_status.sum()),'holdout_accessed':False
},indent=2)+'\n')
print(s.to_string(index=False)); print(pd.DataFrame(geom).to_string(index=False))
