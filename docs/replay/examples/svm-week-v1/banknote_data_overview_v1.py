from pathlib import Path
import os
os.environ['MPLCONFIGDIR'] = str(Path('.mplcache').resolve())
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold

names = ['variance', 'skewness', 'kurtosis', 'entropy', 'class']
raw = pd.read_csv('banknote_raw.csv', header=None, names=names)
features = names[:4]
groups = raw.groupby(features, sort=False, dropna=False).ngroup().to_numpy()
group_counts = pd.Series(groups).value_counts()
conflicts = raw.assign(group_id=groups).groupby('group_id')['class'].nunique()
audit = {
    'raw_rows': int(len(raw)),
    'predictors': features,
    'missing_values': int(raw.isna().sum().sum()),
    'unique_feature_vectors': int(len(group_counts)),
    'repeated_feature_groups': int((group_counts > 1).sum()),
    'rows_in_repeated_groups': int(group_counts[group_counts > 1].sum()),
    'extra_duplicate_rows': int((group_counts - 1).sum()),
    'conflicting_label_groups': int((conflicts > 1).sum()),
}
splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=4031)
dev_idx, holdout_idx = next(splitter.split(raw[features], raw['class'], groups))
roles = np.full(len(raw), 'development', dtype=object)
roles[holdout_idx] = 'final_holdout'
pd.DataFrame({'row_id': np.arange(1, len(raw)+1), 'group_id': groups, 'split_role': roles}).to_csv('banknote_split_roles_v1.csv', index=False)
audit['development_rows'] = int(len(dev_idx))
audit['reserved_holdout_rows'] = int(len(holdout_idx))
# All subsequent summaries and figures use development rows only.
dev = raw.iloc[dev_idx].copy()
dev.insert(0, 'row_id', dev_idx + 1)
del raw, holdout_idx

audit['development_class_counts'] = {str(k): int(v) for k,v in dev['class'].value_counts().sort_index().items()}
dev.head(6).to_csv('banknote_development_preview_v1.csv', index=False)
dev[features].describe().to_csv('banknote_development_summary_v1.csv')
Path('banknote_data_audit_v1.json').write_text(json.dumps(audit, indent=2)+'\n')
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), constrained_layout=True)
for ax, (x, y) in zip(axes, [('variance', 'skewness'), ('kurtosis', 'entropy')]):
    for c, color, marker in [(0, '#0072B2', 'o'), (1, '#D55E00', '^')]:
        subset = dev.loc[dev['class'] == c]
        ax.scatter(subset[x], subset[y], s=21, alpha=.55, color=color, marker=marker, label=f'Class {c}', linewidths=0)
    ax.set(xlabel=x.capitalize(), ylabel=y.capitalize())
    ax.grid(alpha=.18)
axes[0].legend(frameon=True)
fig.suptitle(f'Banknote image summaries: {len(dev):,} development observations', fontsize=16)
fig.savefig('banknote_development_scatter_v1.png', dpi=170)
plt.close(fig)
print(json.dumps(audit, indent=2))
print(dev.head(6).to_string(index=False))
