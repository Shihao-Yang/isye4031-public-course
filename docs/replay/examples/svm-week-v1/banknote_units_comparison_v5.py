from pathlib import Path
import os
os.environ['MPLCONFIGDIR'] = str(Path('.mplcache').resolve())
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

roles = pd.read_csv('banknote_split_roles_v1.csv')
dev_ids = roles.loc[roles.split_role.eq('development'),'row_id'].to_numpy()
dev_set = set(dev_ids)
dev = pd.read_csv('banknote_raw.csv',header=None,
    names=['variance','skewness','kurtosis','entropy','class'],
    skiprows=lambda i:i+1 not in dev_set)
X = dev[['variance','skewness']].to_numpy()
y = dev['class'].to_numpy()
multiplier = np.array([0.1,1.])
# Use tight convergence tolerance; the rescaled raw problem is poorly conditioned.
models = {}
records = []
for standardized in [False,True]:
    for changed in [False,True]:
        key=(standardized,changed)
        estimator=SVC(kernel='linear',C=1.0,tol=1e-7,max_iter=5000000)
        model=make_pipeline(StandardScaler(),estimator) if standardized else estimator
        train=X*multiplier if changed else X
        model.fit(train,y)
        svc=model[-1] if standardized else model
        scores=model.decision_function(train)
        pred=model.predict(train)
        models[key]=model
        if standardized:
            scaler=model[0]
            coef=svc.coef_[0]/scaler.scale_
            intercept=float(svc.intercept_[0]-np.dot(svc.coef_[0],scaler.mean_/scaler.scale_))
        else:
            coef=svc.coef_[0].copy()
            intercept=float(svc.intercept_[0])
        # Express both fits in the original displayed feature coordinates.
        coef=coef*(multiplier if changed else 1)
        records.append({'standardized':standardized,'variance_multiplied_by_0.1':changed,
            'training_errors':int(np.sum(pred!=y)), 'support_vectors':len(svc.support_),
            'boundary_coefficient_original_variance':float(coef[0]),
            'boundary_coefficient_original_skewness':float(coef[1]),
            'boundary_intercept':intercept,'solver_fit_status':int(svc.fit_status_),
            'solver_iterations':int(svc.n_iter_[0])})
comparisons=[]
for standardized in [False,True]:
    original=models[(standardized,False)]
    changed=models[(standardized,True)]
    comparisons.append({'standardized':standardized,
        'changed_development_predictions':int(np.sum(original.predict(X)!=changed.predict(X*multiplier))),
        'max_absolute_score_difference':float(np.max(np.abs(original.decision_function(X)-changed.decision_function(X*multiplier))))})

plt.rcParams.update({'font.size':12,'axes.titlesize':14})
fig,axes=plt.subplots(1,2,figsize=(12,5.6),sharex=True,sharey=True,constrained_layout=True)
xlim=(X[:,0].min()-.6,X[:,0].max()+.6)
ylim=(X[:,1].min()-1,X[:,1].max()+1)
gx,gy=np.meshgrid(np.linspace(*xlim,300),np.linspace(*ylim,300))
grid=np.c_[gx.ravel(),gy.ravel()]
for ax,standardized in zip(axes,[False,True]):
    for c,color,marker in [(0,'#0072B2','o'),(1,'#D55E00','^')]:
        mask=y==c
        ax.scatter(X[mask,0],X[mask,1],s=15,c=color,marker=marker,alpha=.3,linewidths=0)
    for changed,color,style in [(False,'#111111','-'),(True,'#AA3377','--')]:
        scores=models[(standardized,changed)].decision_function(grid*(multiplier if changed else 1)).reshape(gx.shape)
        ax.contour(gx,gy,scores,levels=[0],colors=[color],linestyles=[style],linewidths=2.4)
    ax.set(xlim=xlim,ylim=ylim,xlabel='Variance (original numerical values)',
        title='Standardize before fitting' if standardized else 'Fit without standardization')
    ax.grid(alpha=.15)
axes[0].set_ylabel('Skewness (original numerical values)')
from matplotlib.lines import Line2D
handles=[Line2D([],[],color='#111111',lw=2.4,label='Original units'),
         Line2D([],[],color='#AA3377',ls='--',lw=2.4,label='Variance multiplied by 0.1')]
axes[0].legend(handles=handles,loc='lower left',fontsize=10)
axes[1].legend(handles=handles,loc='lower left',fontsize=10)
fig.suptitle('Changing units at fixed C = 1\nBoundaries shown in the same original coordinates',fontsize=16)
fig.savefig('banknote_units_comparison_v5.png',dpi=180)
plt.close(fig)
pd.DataFrame(records).to_csv('banknote_units_model_results_v5.csv',index=False)
pd.DataFrame(comparisons).to_csv('banknote_units_prediction_comparison_v5.csv',index=False)
Path('banknote_units_context_v5.json').write_text(json.dumps({
    'purpose':'Two-feature development training illustration; final holdout unused',
    'unit_change':'variance multiplied by 0.1; no physical unit claimed',
    'features':['variance','skewness'],'C':1.0,
    'standardization':'Each model learns feature means and standard deviations from development rows',
    'solver_tol':1e-7,'solver_max_iter':5000000
},indent=2)+'\n')
print(pd.DataFrame(records).to_string(index=False))
print(pd.DataFrame(comparisons).to_string(index=False))
