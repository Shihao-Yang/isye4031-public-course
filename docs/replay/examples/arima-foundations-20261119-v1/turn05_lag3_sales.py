import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

base = Path(__file__).resolve().parent
a = np.genfromtxt(base / 'paper_towels.csv', delimiter=',', names=True)
y = a['sales_above_100000_in_10000_rolls']
n = len(y)
k = 3
mean = y.mean()
x, z = y[:-k], y[k:]
products = (x - mean) * (z - mean)
numerator = products.sum()
denominator = ((y - mean)**2).sum()
r = numerator / denominator
rows = np.column_stack([np.arange(1,n-k+1), np.arange(k+1,n+1), x, z, x-mean, z-mean, products])
np.savetxt(base / 'turn05_lag3_pairs.csv', rows, delimiter=',', header='earlier_week,later_week,earlier_sales,later_sales,earlier_centered,later_centered,centered_product', comments='', fmt=['%d','%d','%.4f','%.4f','%.10f','%.10f','%.10f'])
result = {'series': 'Original sales in file units', 'n': n, 'lag_weeks': k, 'pair_count': len(x), 'overall_mean': float(mean), 'sum_centered_pair_products': float(numerator), 'sum_squared_deviations_all_sales': float(denominator), 'sample_autocorrelation': float(r), 'ordinary_pearson_correlation_of_paired_columns': float(np.corrcoef(x,z)[0,1]), 'first_five_pairs': rows[:5].tolist(), 'first_product': float(products[0])}
(base / 'turn05_lag3_results.json').write_text(json.dumps(result, indent=2)+'\n')
plt.rcParams.update({'font.size':16,'axes.titlesize':20,'axes.labelsize':17})
fig, ax = plt.subplots(figsize=(8.5,7))
ax.scatter(x,z,s=38,color='#176a9a',alpha=.75)
ax.set_xlabel('Earlier sales: week t − 3\n(file units)')
ax.set_ylabel('Later sales: week t\n(file units)')
ax.set_title('Sales three weeks apart: 117 pairs',pad=18)
lo,hi=min(y)-1,max(y)+1
ax.set_xlim(lo,hi)
ax.set_ylim(lo,hi)
ax.set_aspect('equal',adjustable='box')
ax.grid(alpha=.22)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout(pad=1.8)
fig.savefig(base / 'turn05_lag3_sales.png',dpi=160)
plt.close(fig)
print(json.dumps(result, indent=2))
