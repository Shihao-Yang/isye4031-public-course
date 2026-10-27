from pathlib import Path
import os
os.environ['MPLCONFIGDIR'] = str(Path('.mplcache').resolve())
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

roles = pd.read_csv('banknote_split_roles_v1.csv')
dev_ids = roles.loc[roles.split_role.eq('development'), 'row_id'].to_numpy()
dev_set = set(dev_ids)
dev = pd.read_csv('banknote_raw.csv', header=None,
    names=['variance','skewness','kurtosis','entropy','class'],
    skiprows=lambda i: i+1 not in dev_set)
scaler = StandardScaler()
X = scaler.fit_transform(dev[['variance','skewness']])
y = dev['class'].to_numpy()
t = 2*y-1
Cs = [0.01,0.1,1.0,10.0]
xlim = (X[:,0].min()-.25,X[:,0].max()+.25)
ylim = (X[:,1].min()-.25,X[:,1].max()+.25)
gx,gy = np.meshgrid(np.linspace(*xlim,350),np.linspace(*ylim,350))
plt.rcParams.update({'font.size':11,'axes.titlesize':13,'axes.labelsize':12})
fig, axes = plt.subplots(2,2,figsize=(12,11),sharex=True,sharey=True)
records, points = [], []
for ax,C in zip(axes.flat,Cs):
    model = SVC(kernel='linear',C=C)
    model.fit(X,y)
    w,b = model.coef_[0],float(model.intercept_[0])
    norm = float(np.linalg.norm(w))
    scores = model.decision_function(X)
    signed = t*scores
    alpha = np.zeros(len(y))
    alpha[model.support_] = np.abs(model.dual_coef_[0])
    svmask = np.zeros(len(y),dtype=bool)
    svmask[model.support_] = True
    at_cap = svmask & np.isclose(alpha,C,rtol=1e-5,atol=1e-8)
    free = svmask & ~at_cap
    rec = {'C':C,'support_vectors':int(svmask.sum()),
           'free_support_vectors':int(free.sum()),'support_vectors_at_C':int(at_cap.sum()),
           'training_errors':int(np.sum(model.predict(X)!=y)),
           'full_margin_width':2/norm,'w_variance':float(w[0]),
           'w_skewness':float(w[1]),'intercept':b}
    records.append(rec)
    points.append(pd.DataFrame({'C':C,'row_id':dev_ids,'class':y,
        'standardized_variance':X[:,0],'standardized_skewness':X[:,1],
        'score':scores,'signed_score':signed,'support_vector':svmask,
        'dual_weight_alpha':alpha,'alpha_at_C':at_cap}))
    zz = w[0]*gx+w[1]*gy+b
    ax.contourf(gx,gy,zz,levels=[-1,1],colors=['#777777'],alpha=.10)
    for c,color,marker in [(0,'#0072B2','o'),(1,'#D55E00','^')]:
        m = y==c
        ax.scatter(X[m,0],X[m,1],s=16,c=color,marker=marker,alpha=.5,linewidths=0,zorder=2)
    ax.scatter(X[svmask,0],X[svmask,1],s=46,facecolors='none',edgecolors='#222222',linewidths=.65,zorder=3)
    ax.contour(gx,gy,zz,levels=[-1,0,1],colors=['#222222']*3,
               linestyles=['--','-','--'],linewidths=[1.3,2,1.3],zorder=4)
    ax.set_title(f'C = {C:g}: {svmask.sum()} support vectors\nMargin width {2/norm:.3f}; {rec["training_errors"]} training errors')
    ax.set(xlim=xlim,ylim=ylim)
    ax.set_aspect('equal',adjustable='box')
    ax.grid(alpha=.15)
for ax in axes[1,:]: ax.set_xlabel('Standardized variance')
for ax in axes[:,0]: ax.set_ylabel('Standardized skewness')
handles = [Line2D([],[],ls='',marker='o',color='#0072B2',label='Class 0'),
           Line2D([],[],ls='',marker='^',color='#D55E00',label='Class 1'),
           Line2D([],[],ls='',marker='o',mfc='none',mec='#222222',ms=9,label='Support vector (ring)'),
           Line2D([],[],color='#222222',lw=2,label='Decision boundary'),
           Line2D([],[],color='#222222',ls='--',label='Margin boundaries')]
fig.suptitle('How C changes a two-feature linear SVM\nSame development data and scaling in every panel',fontsize=17,y=.985)
fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.005),fontsize=11)
fig.tight_layout(rect=[0,.075,1,.94],h_pad=2,w_pad=1.5)
fig.savefig('banknote_C_support_vectors_v1.png',dpi=180)
plt.close(fig)
pd.DataFrame(records).to_csv('banknote_C_support_vectors_summary_v1.csv',index=False)
pd.concat(points,ignore_index=True).to_csv('banknote_C_support_vectors_points_v1.csv',index=False)
Path('banknote_C_support_vectors_context_v1.json').write_text(json.dumps({
    'purpose':'Fixed-C two-feature development training illustration only; no CV or final test',
    'features':['variance','skewness'], 'standardization_mean':scaler.mean_.tolist(),
    'standardization_scale':scaler.scale_.tolist(),
    'support_vector_definition':'SVC support_ indices; dual weights from abs(dual_coef_)',
    'alpha_cap_tolerance':{'rtol':1e-5,'atol':1e-8},
    'margin_equations':'w dot z + b = -1 and +1'
},indent=2)+'\n')
print(pd.DataFrame(records).to_string(index=False))
