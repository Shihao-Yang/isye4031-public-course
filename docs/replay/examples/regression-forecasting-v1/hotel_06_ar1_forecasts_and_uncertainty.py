from pathlib import Path
import os
os.environ['MPLCONFIGDIR']=str(Path.cwd()/'.mplcache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from scipy.optimize import minimize_scalar

d=pd.read_csv('hotel_monthly.csv').query('t<=156').copy()
def design(frame):
    return np.column_stack([np.ones(len(frame)),frame.t.to_numpy()/12,
                            frame.month.to_numpy()[:,None]==np.arange(2,13)])
def fit_ar(x,z):
    def transformed(rho):
        xx=np.vstack([np.sqrt(1-rho*rho)*x[0],x[1:]-rho*x[:-1]])
        zz=np.r_[np.sqrt(1-rho*rho)*z[0],z[1:]-rho*z[:-1]]
        b=np.linalg.lstsq(xx,zz,rcond=None)[0]
        sse=np.sum((zz-xx@b)**2)
        return xx,b,sse
    def objective(rho):
        _,_,sse=transformed(rho)
        return len(z)*np.log(sse/len(z))-np.log1p(-rho*rho)
    opt=minimize_scalar(objective,bounds=(-.98,.98),method='bounded',options={'xatol':1e-10})
    assert opt.success
    rho=float(opt.x)
    xx,b,sse=transformed(rho)
    sigma2=sse/(len(z)-x.shape[1])
    cov=sigma2*np.linalg.inv(xx.T@xx)
    return rho,b,sigma2,cov
rows=[]; details=[]; example={}
for origin in [96,108,120,132,144]:
    tr=d.loc[d.t<=origin]; va=d.loc[(d.t>origin)&(d.t<=origin+12)]
    x=design(tr); future=design(va); z=np.log(tr.rooms_per_day.to_numpy())
    b0=np.linalg.lstsq(x,z,rcond=None)[0]; e0=z-x@b0
    sig0=np.sum(e0**2)/(len(z)-x.shape[1]); cov0=sig0*np.linalg.inv(x.T@x)
    mu0=future@b0; smear=np.mean(np.exp(e0))
    pred0=np.exp(mu0)*smear
    variance0=sig0+np.einsum('ij,jk,ik->i',future,cov0,future)
    low0=np.exp(mu0-1.96*np.sqrt(variance0)); high0=np.exp(mu0+1.96*np.sqrt(variance0))
    rho,b,sig,cov=fit_ar(x,z)
    last=float(z[-1]-x[-1]@b)
    h=np.arange(1,13)
    correction=rho**h*last
    mu=future@b+correction
    innovation_var=sig*np.array([np.sum(rho**(2*np.arange(k))) for k in h])
    sensitivity=future-rho**h[:,None]*x[-1]
    variance=innovation_var+np.einsum('ij,jk,ik->i',sensitivity,cov,sensitivity)
    pred=np.exp(mu+.5*innovation_var)
    low=np.exp(mu-1.96*np.sqrt(variance)); high=np.exp(mu+1.96*np.sqrt(variance))
    actual=va.rooms_per_day.to_numpy()
    for j in range(12):
        rows.append({'origin':origin,'horizon':int(h[j]),'t':int(va.t.iloc[j]),'actual':float(actual[j]),
                     'iid_forecast':float(pred0[j]),'iid_lower95':float(low0[j]),'iid_upper95':float(high0[j]),
                     'ar1_forecast':float(pred[j]),'ar1_lower95':float(low[j]),'ar1_upper95':float(high[j]),
                     'ar1_log_correction':float(correction[j]),'ar1_innovation_log_variance':float(innovation_var[j])})
    details.append({'origin':origin,'rho':rho,'last_log_residual':last,'innovation_variance':float(sig)})
    if origin==120:
        residual=z-x@b
        innovation=residual[1:]-rho*residual[:-1]
        def acf(a,k):
            a=a-a.mean(); return float(np.dot(a[k:],a[:-k])/np.dot(a,a))
        example={'origin':origin,'rho':rho,'latest_log_residual':last,
                 'innovation_variance':float(sig),'ar1_error_acf_lag1':acf(residual,1),
                 'innovation_acf_lag1':acf(innovation,1),'innovation_acf_lag12':acf(innovation,12)}
r=pd.DataFrame(rows)
r.to_csv('hotel_06_ar1_validation_forecasts_and_intervals.csv',index=False)
pd.DataFrame(details).to_csv('hotel_06_ar1_origin_estimates.csv',index=False)
metrics=[]
for name,col in [('Log seasonal, independent errors','iid_forecast'),('Log seasonal, AR(1) errors','ar1_forecast')]:
    for label,subset in [('all five origins',r),('origin 120',r.loc[r.origin.eq(120)])]:
        err=subset.actual-subset[col]
        metrics.append({'method':name,'evaluation':label,'rmse':float(np.sqrt(np.mean(err**2))),
                        'mae':float(np.mean(np.abs(err)))})
pd.DataFrame(metrics).to_csv('hotel_06_ar1_validation_metrics.csv',index=False)
Path('hotel_06_ar1_method_and_example.json').write_text(json.dumps({'model':'log(y_t)=x_t beta+u_t; u_t=rho*u_(t-1)+epsilon_t',
 'fitting':'Stationary Gaussian profile likelihood for rho and GLS for regression coefficients',
 'point_forecast':'exp(x_future beta+rho^h*u_T+innovation_variance_h/2)',
 'interval':'Approximate 95% prediction interval on log scale, exponentiated. Includes innovation variance and beta covariance conditional on estimated rho; omits rho-estimation uncertainty.',
 'innovation_assumption':'Independent Gaussian innovations with constant variance; remaining dependence can invalidate intervals',
 'validation':'Same origins 96,108,120,132,144; each forecasts a full year without updates',
 'reserved_final_used':False,'example':example},indent=2))
a=r.loc[r.origin.eq(120)]
plt.rcParams.update({'font.size':12,'axes.titlesize':14})
fig,ax=plt.subplots(figsize=(10,4.8))
ax.fill_between(a.horizon,a.ar1_lower95,a.ar1_upper95,color='#c45118',alpha=.18,label='AR(1): approximate 95% prediction interval')
ax.plot(a.horizon,a.iid_lower95,':',color='#2166ac',lw=1.4,label='Independent-error 95% interval bounds')
ax.plot(a.horizon,a.iid_upper95,':',color='#2166ac',lw=1.4)
ax.plot(a.horizon,a.iid_forecast,'s--',color='#2166ac',ms=4,label='Independent-error forecast')
ax.plot(a.horizon,a.ar1_forecast,'o-',color='#c45118',ms=4,label='AR(1) forecast')
ax.plot(a.horizon,a.actual,'x-',color='#333333',ms=5,label='Observed later')
ax.set(xticks=range(1,13),xlabel='Months ahead from origin 120',ylabel='Average rooms occupied per day',
       title='AR(1) changes both the forecast and conditional uncertainty')
ax.grid(alpha=.2); ax.spines[['top','right']].set_visible(False)
ax.legend(fontsize=8.5,ncol=2,loc='upper left')
fig.tight_layout(); fig.savefig('hotel_06_ar1_forecasts_and_prediction_intervals.png',dpi=180)
print(pd.DataFrame(metrics).to_string(index=False))
print(json.dumps(example,indent=2))
print(a.loc[a.horizon.isin([1,8,12])].to_string(index=False))
