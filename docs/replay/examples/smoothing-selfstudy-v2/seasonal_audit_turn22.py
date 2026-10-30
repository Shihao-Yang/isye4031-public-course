"""Demonstrate deliberate forecasting mistakes; no random seed needed."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
with (root/'cola_fitted_values_turn20.csv').open() as f:
    rs=[r for r in csv.DictReader(f) if r['model']=='Multiplicative']
l=float(rs[23]['level']);b=float(rs[23]['trend'])
s=np.array([float(r['seasonal']) for r in rs[12:24]])
h=np.arange(1,13)
correct=(l+h*b)*s
shifted=(l+h*b)*np.roll(s,-1)
test=np.array([float(r['actual']) for r in rs[24:]])
pred=np.array([float(r['fitted']) for r in rs[24:]])
errors=test-pred
# With the old seasonal multiplier, L_t*S_old=.2*y_t+.8*forecast_t.
# It is an after-observation reconstruction, not an available forecast.
leaked=pred+.2*errors
bad_errors=test-leaked
print('L24',l,'B24',b,'Jan S',s[0],'Feb S',s[1])
print('January correct',correct[0],'wrong season',shifted[0],'actual',test[0])
print('honest Jan forecast',pred[0],'leaked reconstruction',leaked[0])
print('MSD proper',np.mean(errors**2),'leaked',np.mean(bad_errors**2),'factor',np.mean(bad_errors**2)/np.mean(errors**2))
print('proper cases',correct[0]*100,'wrong cases',correct[0])
with (root/'seasonal_audit_turn22.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['month_of_year','correct_origin24_forecast_hundreds','shifted_season_forecast_hundreds','actual_third_year','rolling_forecast_hundreds','leaked_reconstruction_hundreds'])
    w.writerows(zip(h,correct,shifted,test,pred,leaked))
fig,axes=plt.subplots(1,3,figsize=(13,4.5))
axes[0].plot(h,correct,'o-',color='#176b91',markersize=3,label='Correct seasonal index')
axes[0].plot(h,shifted,'s--',color='#b85416',markersize=3,label='Seasonal index shifted by one')
axes[0].set(title='1. Wrong season moves the peak',xlabel='Forecast month',ylabel='Sales (hundreds of cases)',xticks=[1,4,8,9,12])
axes[0].legend(fontsize=7.5)
values=[np.mean(errors**2),np.mean(bad_errors**2)]
axes[1].bar(['Forecast before\nobservation','Reconstruction after\nobservation'],values,color=['#176b91','#b85416'])
for i,v in enumerate(values):axes[1].text(i,v+45,f'{v:.1f}',ha='center',fontsize=9)
axes[1].set(title='2. Leakage makes errors look smaller',ylabel='Third-year MSD',ylim=(0,2900))
values=[correct[0]*100,correct[0]]
axes[2].bar(['Correct conversion','Unit label mistake'],values,color=['#176b91','#b85416'])
for i,v in enumerate(values):axes[2].text(i,v+650,f'{v:,.1f}',ha='center',fontsize=9)
axes[2].set(title='3. Missing ×100 changes the scale',ylabel='Claimed January sales (cases)',ylim=(0,41000))
for ax in axes:ax.grid(axis='y',alpha=.2)
fig.tight_layout(pad=1.8,w_pad=2)
fig.savefig(root/'seasonal_audit_turn22.png',dpi=170)
plt.close(fig)
