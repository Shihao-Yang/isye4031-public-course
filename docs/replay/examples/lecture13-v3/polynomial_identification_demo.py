import os
os.environ['MPLCONFIGDIR']='.mplcache'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
rng=np.random.default_rng(403115)
n=30
x=np.repeat([0.,1.],15);noise=rng.normal(0,.15,n);y=2+3*x+noise
xn=x+rng.normal(0,.001,n)
X=np.column_stack([np.ones(n),x,x*x]);N=np.column_stack([np.ones(n),xn,xn*xn])
b=np.linalg.lstsq(X,y,rcond=None)[0];bn=np.linalg.lstsq(N,y,rcond=None)[0]
s2=float(np.sum((y-N@bn)**2)/(n-3));se=np.sqrt(np.diag(s2*np.linalg.inv(N.T@N)))
variants=np.array([[b[0],b[1]+v,b[2]-v] for v in [-10,0,10]])
grid=np.linspace(0,1,200);G=np.column_stack([np.ones(len(grid)),grid,grid*grid])
pd.DataFrame({'constructed_x_two_levels':x,'constructed_x_near_two_levels':xn,'constructed_y':y}).to_csv('constructed_polynomial_identification_data.csv',index=False)
r={'seed':403115,'construction':'30 constructed observations at x=0 or 1; y=2+3*x+Normal(0,0.15). Near-two-level x adds Normal(0,0.001); y unchanged. Neither dataset is sales-territory data.','exact_design_rank':int(np.linalg.matrix_rank(X)),'exact_design_columns':3,'exact_pseudoinverse_coefficients':b.tolist(),'equivalent_coefficients_intercept_linear_quadratic':variants.tolist(),'max_fitted_difference_equivalent_choices':float(np.max(np.abs(X@variants.T-(X@b)[:,None]))),'near_design_rank':int(np.linalg.matrix_rank(N)),'near_coefficients':bn.tolist(),'near_classical_standard_errors':se.tolist()}
with open('polynomial_identification_demo_results.json','w') as f:json.dump(r,f,indent=2)
fig,axes=plt.subplots(1,2,figsize=(11,4.8))
axes[0].scatter(x,y,color='black',s=20,alpha=.6,label='Constructed observations')
for v in variants:axes[0].plot(grid,G@v,label=f'Linear {v[1]:.2f}, quadratic {v[2]:.2f}')
axes[0].set_xlabel('Constructed x');axes[0].set_ylabel('Constructed y');axes[0].set_title('Exact two levels: same fit, different curves');axes[0].legend(fontsize=8)
axes[1].scatter(xn,y,color='black',s=20,alpha=.6,label='Constructed near-two-level observations');axes[1].plot(grid,G@bn,color='#bc402b',label='Fitted quadratic');axes[1].set_xlabel('Constructed x');axes[1].set_ylabel('Constructed y');axes[1].set_title('Tiny x perturbations: unique but unstable fit');axes[1].legend(fontsize=8)
fig.suptitle('Constructed polynomial identification experiment',fontsize=14);fig.tight_layout();fig.savefig('polynomial_identification_demo.png',dpi=180)
print(json.dumps(r,indent=2))
