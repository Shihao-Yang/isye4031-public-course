"""Plot saved stability results with a separate legend area."""
import os
os.environ['MPLCONFIGDIR']='.mplcache'
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

summary=pd.read_csv('lasso_subsample_selection_frequencies.csv')
palette={'Direct equation term':'#24689b','Correlated proxy':'#d68a29','Independent noise input':'#8a8a8a'}
fig,ax=plt.subplots(figsize=(11,7.8))
bars=ax.barh(np.arange(len(summary)),summary.selection_fraction*100,
             color=[palette[g] for g in summary.known_synthetic_role])
ax.set_yticks(np.arange(len(summary)),[s.replace('_',' ') for s in summary.predictor])
ax.invert_yaxis()
ax.set(xlim=(0,112),xlabel='Selected in tuned subsample fits (%)',
       title='Does Lasso select the same variables? Constructed sales data')
for bar,count in zip(bars,summary.selected_count):
    ax.text(bar.get_width()+1,bar.get_y()+bar.get_height()/2,f'{count}/60',va='center',fontsize=10)
ax.grid(axis='x',alpha=.18)
ax.set_axisbelow(True)
fig.subplots_adjust(left=.27,right=.96,top=.91,bottom=.30)
fig.legend(handles=[Patch(facecolor=c,label=g) for g,c in palette.items()],
           loc='lower center',bbox_to_anchor=(.5,.18),ncol=3,frameon=False,fontsize=10)
fig.text(.06,.05,'Training-only sensitivity: 60 subsamples of 134/168 stores, seed 4031; penalty retuned within each subsample.\nColors identify roles in the known synthetic equation, not causal conclusions. Subsamples overlap.\nSelection fractions are descriptive frequencies, not probabilities of a real effect. Reserved test set unused.',fontsize=10)
fig.savefig('lasso_selection_stability_clear_legend.png',dpi=160)
