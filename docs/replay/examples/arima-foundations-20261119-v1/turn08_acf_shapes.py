import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

base=Path(__file__).resolve().parent
lags=np.arange(1,21)
cutoff=np.zeros(20)
cutoff[:2]=[.5,.2]
curves={'fast_positive':.4**lags,'slow_positive':.95**lags,'alternating':(-.65)**lags}
result={'type':'Constructed theoretical ACF illustrations, not estimates from sales','random_simulation':False,'seed':None,'lags':lags.tolist(),'cutoff':cutoff.tolist(),'decays':{k:v.tolist() for k,v in curves.items()},'definitions':{'cutoff':'exactly zero at every lag after lag 2','fast_positive':'0.4 raised to lag','slow_positive':'0.95 raised to lag','alternating':'negative 0.65 raised to lag'}}
(base/'turn08_acf_shapes_results.json').write_text(json.dumps(result,indent=2)+'\n')
plt.rcParams.update({'font.size':16,'axes.titlesize':20,'axes.labelsize':17})
fig,axes=plt.subplots(1,2,figsize=(15,6.5),sharey=True)
marker,stems,baseline=axes[0].stem(lags,cutoff,basefmt=' ')
plt.setp(marker,color='#176a9a',markersize=6)
plt.setp(stems,color='#176a9a',linewidth=2)
axes[0].set_title('Cutoff: zero after lag 2',pad=16)
axes[0].annotate('Exactly zero from lag 3 onward',xy=(10,0),xytext=(5,.43),arrowprops={'arrowstyle':'->','color':'#444444'},fontsize=15)
axes[1].plot(lags,curves['fast_positive'],'o-',lw=2,color='#176a9a',label='Fast decay',markersize=4)
axes[1].plot(lags,curves['slow_positive'],'o-',lw=2,color='#b45c20',label='Slow decay',markersize=4)
axes[1].plot(lags,curves['alternating'],'o-',lw=1.7,color='#7d4b99',label='Alternating decay',markersize=4)
axes[1].set_title('Die down: approach zero gradually',pad=16)
axes[1].legend(frameon=False,loc='upper right',fontsize=14)
for ax in axes:
    ax.axhline(0,color='#333333',lw=1)
    ax.set_xlabel('Lag')
    ax.set_xlim(.5,20.5)
    ax.set_xticks([1,2,5,10,15,20])
    ax.set_ylim(-.8,1.05)
    ax.grid(axis='y',alpha=.2)
    ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Theoretical autocorrelation')
fig.suptitle('Constructed illustrations, not paper-towel data',fontsize=20,y=1.01)
fig.tight_layout(pad=2,w_pad=2)
fig.savefig(base/'turn08_acf_shapes.png',dpi=160,bbox_inches='tight')
plt.close(fig)
print('Saved constructed illustrations and numerical values; no random simulation used.')
