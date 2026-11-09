import csv
import json
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('paper_towels.csv',newline='') as f:
    rows=list(csv.reader(f))
y=np.array([float(r[1]) for r in rows[1:]])
w=np.diff(y)
# Conditional least squares: condition on y_1 and set e_1 = 0.
# Plus-sign convention: w_t = e_t + theta * e_{t-1}.
def innovations(theta):
    previous=0.0
    errors=[]
    forecasts=[]
    for value in w:
        forecast=theta*previous
        error=value-forecast
        forecasts.append(forecast)
        errors.append(error)
        previous=error
    return np.array(errors),np.array(forecasts)
def objective(theta):
    errors,_=innovations(theta)
    return float(errors@errors)
fit=minimize_scalar(objective,bounds=(-.999,.999),method='bounded',options={'xatol':1e-13})
theta=float(fit.x)
e,f=innovations(theta)
checks=[]
for i in range(5):
    checks.append({'week':i+2,'previous_level':float(y[i]),'actual_level':float(y[i+1]),
                   'actual_change':float(w[i]),'previous_error':float(0 if i==0 else e[i-1]),
                   'forecast_change':float(f[i]),'forecast_level':float(y[i]+f[i]),
                   'error_actual_minus_forecast':float(e[i])})
results={'method':'conditional least squares; fixed y1, e1=0; -0.999 < theta < 0.999',
         'optimizer_success':bool(fit.success),'theta_hat_plus_convention':theta,
         'SSE':float(e@e),'number_errors':len(e),'in_sample_RMSE':float(np.sqrt(np.mean(e**2))),
         'SSE_theta_zero':objective(0),'first_updates':checks,
         'last_fitted_error':float(e[-1]),'forecast_week121_change':float(theta*e[-1]),
         'forecast_week121_level':float(y[-1]+theta*e[-1]),
         'forecast_week121_rolls':float(100000+10000*(y[-1]+theta*e[-1]))}
with open('paper_towels_fit_06_results.json','w') as out:
    json.dump(results,out,indent=2)
with open('paper_towels_fit_06_errors.csv','w',newline='') as out:
    writer=csv.writer(out)
    writer.writerow(['week','observed_change','fitted_one_step_change','fitted_error'])
    writer.writerows(zip(range(2,len(y)+1),w,f,e))
grid=np.linspace(-.95,.95,401)
fig,ax=plt.subplots(figsize=(8,3.6),constrained_layout=True)
ax.plot(grid,[objective(v) for v in grid],color='#21618c')
ax.plot(theta,objective(theta),'o',color='#b45f06',label=f'Minimum: theta = {theta:.4f}')
ax.set(xlabel='MA coefficient theta (plus-sign convention)',ylabel='Sum of squared one-step errors',title='No-drift ARIMA(0,1,1): conditional least squares')
ax.legend()
ax.grid(alpha=.2)
ax.spines[['top','right']].set_visible(False)
fig.savefig('paper_towels_fit_06.png',dpi=170)
print(json.dumps(results,indent=2))
