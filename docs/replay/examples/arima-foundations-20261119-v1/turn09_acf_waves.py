import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

base=Path(__file__).resolve().parent
lags=np.arange(1,25)
damping=.88
wave=damping**lags*np.cos(np.pi*lags/4)
alternating=(-damping)**lags
result={'type':'Constructed theoretical ACF illustrations, not sales estimates','random_simulation':False,'seed':None,'lags':lags.tolist(),'damping_factor':damping,'wave_formula':'0.88**k * cos(pi*k/4)','alternating_formula':'(-0.88)**k','wave':wave.tolist(),'alternating':alternating.tolist(),'envelope':(damping**lags).tolist()}
(base/'turn09_acf_waves_results.json').write_text(json.dumps(result,indent=2)+'\n')
plt.rcParams.update({'font.size':16,'axes.titlesize':20,'axes.labelsize':17})
fig,axes=plt.subplots(1,2,figsize=(15,6.3),sharex=True,sharey=True)
for ax,values,title,color in [(axes[0],wave,'Damped wave: signs change in groups','#176a9a'),(axes[1],alternating,'Alternating: sign changes at every lag','#7d4b99')]:
    marker,stems,baseline=ax.stem(lags,values,basefmt=' ')
    plt.setp(marker,color=color,markersize=5)
    plt.setp(stems,color=color,linewidth=1.9)
    ax.plot(lags,values,color=color,lw=1,alpha=.55)
    ax.plot(lags,damping**lags,'--',color='#888888',lw=1.3)
    ax.plot(lags,-damping**lags,'--',color='#888888',lw=1.3)
    ax.axhline(0,color='#333333',lw=1)
    ax.set_title(title,pad=15)
    ax.set_xlabel('Lag')
    ax.set_xticks([1,4,8,12,16,20,24])
    ax.set_xlim(.4,24.6)
    ax.set_ylim(-1,1)
    ax.grid(axis='y',alpha=.2)
    ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Theoretical autocorrelation')
fig.suptitle('Constructed illustrations; gray curves show shrinking envelopes',fontsize=19,y=1.01)
fig.tight_layout(pad=2,w_pad=2)
fig.savefig(base/'turn09_acf_waves.png',dpi=160,bbox_inches='tight')
plt.close(fig)
print(json.dumps({key:result[key] for key in ['type','wave_formula','alternating_formula','random_simulation']},indent=2))
