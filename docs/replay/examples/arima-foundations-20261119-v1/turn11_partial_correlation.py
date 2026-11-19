import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

base=Path(__file__).resolve().parent
raw=np.genfromtxt(base/'paper_towels.csv',delimiter=',',names=True)
y=raw['sales_above_100000_in_10000_rolls']
x=np.diff(y)
# A row is (change two weeks earlier, intervening change, current change).
earlier,middle,current=x[:-2],x[1:-1],x[2:]
X=np.column_stack([np.ones(len(middle)),middle])
b_earlier=np.linalg.lstsq(X,earlier,rcond=None)[0]
b_current=np.linalg.lstsq(X,current,rcond=None)[0]
e_earlier=earlier-X@b_earlier
e_current=current-X@b_current
paired_corr=float(np.corrcoef(earlier,current)[0,1])
partial=float(np.corrcoef(e_earlier,e_current)[0,1])
c=x-x.mean()
r1=float(c[1:]@c[:-1]/(c@c))
r2=float(c[2:]@c[:-2]/(c@c))
pacf_yw=(r2-r1*r1)/(1-r1*r1)
result={'series':'First differences in file units','triplet_count':len(middle),'ordinary_lag2_correlation_matched_rows':paired_corr,'partial_correlation_via_residuals':partial,'earlier_on_middle_intercept_slope':b_earlier.tolist(),'current_on_middle_intercept_slope':b_current.tolist(),'acf_lag1':r1,'acf_lag2':r2,'pacf_lag2_yule_walker':pacf_yw,'note':'Residual correlation and ACF-based PACF differ slightly because of finite-sample conventions.'}
(base/'turn11_partial_correlation_results.json').write_text(json.dumps(result,indent=2)+'\n')
np.savetxt(base/'turn11_partial_correlation_rows.csv',np.column_stack([np.arange(4,121),earlier,middle,current,e_earlier,e_current]),delimiter=',',header='current_week,change_two_weeks_earlier,intervening_change,current_change,earlier_residual,current_residual',comments='',fmt='%.10f')
plt.rcParams.update({'font.size':16,'axes.titlesize':19,'axes.labelsize':16})
fig,axes=plt.subplots(1,2,figsize=(15,6.4))
for ax,a,b,title,xlabel,ylabel in [
(axes[0],earlier,current,f'Before adjustment: correlation = {paired_corr:.3f}','Change two weeks earlier','Current change'),
(axes[1],e_earlier,e_current,f'After adjustment: correlation = {partial:.3f}','Earlier change left unexplained\nby intervening change','Current change left unexplained\nby intervening change')]:
    ax.scatter(a,b,s=32,alpha=.7,color='#176a9a')
    slope,intercept=np.polyfit(a,b,1)
    xx=np.array([-4,4])
    ax.plot(xx,intercept+slope*xx,color='#b45c20',lw=2)
    ax.axhline(0,color='#777777',lw=.8)
    ax.axvline(0,color='#777777',lw=.8)
    ax.set(title=title,xlabel=xlabel,ylabel=ylabel,xlim=(-4,4),ylim=(-4,4))
    ax.set_aspect('equal',adjustable='box')
    ax.grid(alpha=.15)
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('First differences; all axes in units of 10,000 rolls',fontsize=19,y=1.01)
fig.tight_layout(pad=2,w_pad=3)
fig.savefig(base/'turn11_partial_correlation.png',dpi=160,bbox_inches='tight')
plt.close(fig)
print(json.dumps(result,indent=2))
