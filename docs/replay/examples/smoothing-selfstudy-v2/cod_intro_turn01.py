"""Plot supplied cod catch observations; no random sampling (seed: not used)."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
with (root / 'cod.csv').open() as f:
    rows = list(csv.DictReader(f))
t = np.array([int(r['t']) for r in rows])
y = np.array([float(r['y']) for r in rows])
mean = y.mean()
print(f'n={len(y)}, total={y.sum():.0f}, mean={mean:.6f}, min={y.min():.0f}, max={y.max():.0f}')
print(f'First-year mean={y[:12].mean():.6f}; second-year mean={y[12:].mean():.6f}')
with (root / 'cod_summary_turn01.csv').open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['n', 'total_tons', 'mean_tons', 'min_tons', 'max_tons'])
    w.writerow([len(y), y.sum(), mean, y.min(), y.max()])
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.plot(t, y, 'o-', color='#176b91', lw=1.7, label='Observed monthly catch')
ax.axhline(mean, color='#b75b18', ls='--', lw=1.6,
           label=f'Historical average: {mean:.1f} tons')
ax.axvline(12.5, color='gray', lw=1, alpha=.5)
ax.set(xlabel='Month index (1 = January, first year)', ylabel='Cod catch (tons)',
       title='Bay City Seafood: 24 months of cod catch', xlim=(.5, 24.5), ylim=(250, 430))
ax.set_xticks([1, 3, 6, 9, 12, 13, 15, 18, 21, 24])
ax.grid(axis='y', alpha=.22)
ax.legend(loc='upper right', frameon=True)
fig.tight_layout(pad=1.8)
fig.savefig(root / 'cod_data_turn01.png', dpi=170)
plt.close(fig)
