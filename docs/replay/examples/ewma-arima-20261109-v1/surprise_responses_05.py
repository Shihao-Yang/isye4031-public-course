import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

h=np.arange(9)
shock=10.0
phi=0.6
theta=0.6
alpha=0.2
ar=shock*phi**h
ma=np.zeros(len(h)); ma[0]=shock; ma[1]=theta*shock
pulse=alpha*shock*(1-alpha)**h
forecast=np.full(len(h),alpha*shock)
results={'constructed_parameters':{'shock':shock,'AR_phi':phi,'MA_theta_plus_convention':theta,'EWMA_alpha':alpha},
         'weeks_since_shock':h.tolist(),'AR_innovation_response':ar.tolist(),
         'MA_innovation_response':ma.tolist(),'EWMA_level_response_to_isolated_observation_pulse':pulse.tolist(),
         'EWMA_forecast_after_shock_no_further_innovations':forecast.tolist()}
with open('surprise_responses_05_results.json','w') as f:
    json.dump(results,f,indent=2)
fig,axes=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
for ax, vals, title in zip(axes,[ar,ma,pulse],['AR(1): innovation of 10','MA(1): innovation of 10','EWMA: observed surprise of 10']):
    ax.plot(h,vals,'o-',color='#21618c',ms=4,label='Later observations fixed at baseline' if ax is axes[2] else None)
    ax.axhline(0,color='gray',lw=.7)
    ax.set(title=title,xlabel='Weeks since surprise',ylim=(-.5,11),xticks=[0,2,4,6,8])
    ax.grid(axis='y',alpha=.2)
    ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Change from baseline (constructed units)')
axes[2].plot(h,forecast,'--',color='#b45f06',label='Forecast path: no further surprises')
axes[2].legend(fontsize=8,loc='upper right')
fig.savefig('surprise_responses_05.png',dpi=170)
print(json.dumps(results,indent=2))
