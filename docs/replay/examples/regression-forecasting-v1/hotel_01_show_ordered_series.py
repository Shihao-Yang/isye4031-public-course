from pathlib import Path
import os
os.environ['MPLCONFIGDIR'] = str(Path.cwd() / '.mplcache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

d = pd.read_csv('hotel_monthly.csv')
assert d['t'].tolist() == list(range(1,169))
assert np.array_equal(d['month'].to_numpy(), (d['t'].to_numpy()-1)%12+1)
assert not d.isna().any().any()
roles = pd.DataFrame({'t':d.t,'month':d.month,
    'role':np.where(d.t<=156,'development','reserved_final_evaluation')})
roles.to_csv('hotel_01_fixed_time_split.csv',index=False)
dev = d.loc[d.t<=156].copy()
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
preview = dev.head(12).copy()
preview['month_name'] = [month_names[m-1] for m in preview.month]
preview.to_csv('hotel_01_first_year_preview.csv',index=False)
summary = {'source':'FTR, fourth edition, Bowerman, O\u2019Connell and Koehler, Table 6.4, printed p. 297',
    'origin':'Textbook observations, transcribed unchanged',
    'response':'Monthly average rooms occupied per day across four hotels',
    'observations':len(d),'sequential_years':len(d)//12,
    'calendar_origin':'t=1 is January of year 1; actual calendar years unspecified',
    'development_start_t':int(dev.t.min()),'development_end_t':int(dev.t.max()),
    'development_months':len(dev),'reserved_months':int((roles.role=='reserved_final_evaluation').sum()),
    'reserved_t_range':[157,168],
    'reserved_outcomes_displayed':False,'forecast_evaluation_performed':False,
    'validation_rule':'Use earlier expanding training windows followed by 12-month validation blocks; no random folds.'}
Path('hotel_01_series_context.json').write_text(json.dumps(summary,indent=2))
plt.rcParams.update({'font.size':12,'axes.titlesize':14,'axes.labelsize':12})
fig,ax = plt.subplots(figsize=(10,4.8))
ax.plot(dev.t,dev.rooms_per_day,color='#2166ac',lw=1.8,label='Development observations (months 1–156)')
ax.axvspan(156.5,168.5,color='#999999',alpha=.22,label='Reserved final 12 months (values hidden)')
ax.axvline(156.5,color='#666666',ls='--',lw=1)
ax.set(xlim=(.5,168.5),xlabel='Sequential year (tick marks show January)',
       ylabel='Average rooms occupied per day',title='Hotel occupancy: monthly observations across four hotels')
ax.set_xticks(np.arange(1,169,12),[str(y) for y in range(1,15)])
ax.grid(alpha=.2)
ax.spines[['top','right']].set_visible(False)
ax.legend(loc='upper left',fontsize=10)
fig.tight_layout()
fig.savefig('hotel_01_monthly_series_and_holdout.png',dpi=180)
print(preview.to_string(index=False))
print(json.dumps(summary,indent=2))
