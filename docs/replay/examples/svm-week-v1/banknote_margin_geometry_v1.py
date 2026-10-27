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

# Load only development rows, using the saved roles without inspecting heldout values.
roles = pd.read_csv('banknote_split_roles_v1.csv')
dev_rows = set(roles.loc[roles.split_role == 'development', 'row_id'])
cols = ['variance','skewness','kurtosis','entropy','class']
dev = pd.read_csv('banknote_raw.csv', header=None, names=cols,
                  skiprows=lambda i: i+1 not in dev_rows)
features = ['variance','skewness']
scaler = StandardScaler()
X = scaler.fit_transform(dev[features])
y = dev['class'].to_numpy()
# A fixed C for geometry, with no tuning or performance evaluation.
model = SVC(kernel='linear', C=1.0)
model.fit(X, y)
w = model.coef_[0]
b = float(model.intercept_[0])
norm = float(np.linalg.norm(w))
scores = model.decision_function(X)
signed_margins = (2*y-1)*scores
result = {
    'purpose': 'two-feature development training illustration, not model selection or final testing',
    'features': features, 'C': 1.0,
    'standardization_mean': scaler.mean_.tolist(),
    'standardization_scale': scaler.scale_.tolist(),
    'w': w.tolist(), 'b': b, 'norm_w': norm,
    'distance_from_boundary_to_each_margin': 1/norm,
    'full_margin_width': 2/norm,
    'development_training_rows': len(y),
    'training_misclassified_rows': int(np.sum(model.predict(X) != y)),
    'rows_strictly_inside_margin_or_on_wrong_side': int(np.sum(signed_margins < 1-1e-6)),
    'support_vectors': int(len(model.support_)),
}
Path('banknote_margin_geometry_results_v1.json').write_text(json.dumps(result, indent=2)+'\n')
pd.DataFrame({'row_id': sorted(dev_rows), 'standardized_variance': X[:,0],
              'standardized_skewness': X[:,1], 'class': y,
              'score': scores, 'signed_functional_margin': signed_margins,
              'support_vector': np.isin(np.arange(len(y)), model.support_)}).to_csv('banknote_margin_geometry_points_v1.csv', index=False)
plt.rcParams.update({'font.size':12, 'axes.labelsize':13})
fig, ax = plt.subplots(figsize=(9,6.4), constrained_layout=True)
for c,color,marker in [(0,'#0072B2','o'),(1,'#D55E00','^')]:
    mask = y == c
    ax.scatter(X[mask,0],X[mask,1],s=23,alpha=.48,c=color,marker=marker,label=f'Class {c}',linewidths=0)
xlim = (float(X[:,0].min()-.3),float(X[:,0].max()+.3))
ylim = (float(X[:,1].min()-.3),float(X[:,1].max()+.3))
xx,yy = np.meshgrid(np.linspace(*xlim,300),np.linspace(*ylim,300))
zz = w[0]*xx+w[1]*yy+b
ax.contourf(xx,yy,zz,levels=[-1,1],colors=['#777777'],alpha=.10)
ax.contour(xx,yy,zz,levels=[-1,0,1],colors=['#333333']*3,linestyles=['--','-','--'],linewidths=[1.6,2.3,1.6])
# Draw a segment normal to the boundary, at a location away from its densest region.
anchor_x = .25
anchor = np.array([anchor_x,(-b-w[0]*anchor_x)/w[1]])
normal_offset = w/(norm**2)
p1,p2 = anchor-normal_offset,anchor+normal_offset
ax.annotate('',xy=p1,xytext=p2,arrowprops={'arrowstyle':'<->','color':'#6A3D9A','lw':2.5})
ax.annotate(f'Full margin = 2 / ||w|| = {2/norm:.2f}',xy=anchor,xytext=(.8,1.7),
            arrowprops={'arrowstyle':'-','color':'#6A3D9A'},color='#6A3D9A',fontsize=12,
            bbox={'facecolor':'white','alpha':.9,'edgecolor':'none'})
from matplotlib.lines import Line2D
handles,labels=ax.get_legend_handles_labels()
handles += [Line2D([0],[0],color='#333333',lw=2.3),Line2D([0],[0],color='#333333',ls='--',lw=1.6)]
labels += ['Decision line: f(z) = 0','Margin lines: f(z) = -1 and +1']
ax.legend(handles,labels,loc='lower left',fontsize=10,framealpha=.95)
ax.set(xlim=xlim,ylim=ylim,xlabel='Variance (standard deviations from development mean)',
       ylabel='Skewness (standard deviations from development mean)',
       title='A line and its soft margin in two banknote features\nFixed C = 1; development training illustration')
ax.set_aspect('equal',adjustable='box')
ax.grid(alpha=.15)
fig.savefig('banknote_margin_geometry_v1.png',dpi=180)
plt.close(fig)
print(json.dumps(result,indent=2))
