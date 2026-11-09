import csv
import json
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('paper_towels.csv',newline='') as f:
    data=list(csv.reader(f))
y=np.array([float(r[1]) for r in data[1:]])
w=np.diff(y)
def errors(theta):
    prev=0.; out=[]
    for z in w:
        prev=z-theta*prev
        out.append(prev)
    return np.array(out)
fit=minimize_scalar(lambda theta:float(np.sum(errors(theta)**2)),bounds=(-.999,.999),method='bounded',options={'xatol':1e-13})
theta=float(fit.x); e=errors(theta)
# Residual variance estimate: 119 conditional errors minus one fitted coefficient.
sigma2=float(e@e/(len(e)-1))
h=np.arange(1,7)
forecast=float(y[-1]+theta*e[-1])
variance=sigma2*(1+(h-1)*(1+theta)**2)
se=np.sqrt(variance)
z=float(norm.ppf(.975))
mean=np.full(6,100000+10000*forecast)
lo=100000+10000*(forecast-z*se)
hi=100000+10000*(forecast+z*se)
rows=[{'week':int(120+k),'horizon':int(k),'point_rolls':float(m),'lower95_rolls':float(l),'upper95_rolls':float(u),'forecast_SE_recorded_units':float(s)} for k,m,l,u,s in zip(h,mean,lo,hi,se)]
result={'theta_hat_plus_convention':theta,'last_error':float(e[-1]),'sigma2_estimate_SSE_over_118':sigma2,
        'last_observed_rolls':float(100000+10000*y[-1]),'forecast_change_week121':float(theta*e[-1]),
        'assumptions':'Independent Gaussian innovations with constant variance; fitted theta and last innovation treated as known; parameter and model uncertainty omitted.',
        'forecasts':rows}
with open('paper_towels_forecast_08_results.json','w') as f:
    json.dump(result,f,indent=2)
with open('paper_towels_forecast_08.csv','w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
fig,ax=plt.subplots(figsize=(10,4.3),constrained_layout=True)
ax.plot(np.arange(105,121),100000+10000*y[104:],color='#21618c',marker='o',ms=3,label='Observed sales')
ax.fill_between(120+h,lo,hi,color='#b45f06',alpha=.18,label='95% prediction interval (plug-in)')
ax.plot(np.r_[120,120+h],np.r_[100000+10000*y[-1],mean],'o--',color='#b45f06',ms=4,label='Future point forecast')
ax.axvline(120.5,color='gray',ls=':',lw=1)
ax.set(xlabel='Week index',ylabel='Paper-towel sales (rolls)',title='No-drift ARIMA(0,1,1): six-week forecasts',xticks=[105,110,115,120,121,122,123,124,125,126])
ax.ticklabel_format(axis='y',style='plain',useOffset=False)
ax.legend(fontsize=9,loc='lower left')
ax.grid(axis='y',alpha=.2)
ax.spines[['top','right']].set_visible(False)
fig.savefig('paper_towels_forecast_08.png',dpi=170)
print(json.dumps(result,indent=2))
