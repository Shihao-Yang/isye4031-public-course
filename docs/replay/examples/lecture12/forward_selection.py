"""Forward selection with conditional partial F entry tests, alpha=0.10."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f

ROOT = Path(__file__).resolve().parent
data = pd.read_csv(ROOT / 'sales_territory.csv')
assert np.isfinite(data.to_numpy()).all()
predictors = [c for c in data.columns if c != 'Sales']

def fit(names):
    return sm.OLS(data['Sales'], sm.add_constant(data[names], has_constant='add')).fit()

selected, tests, path = [], [], []
while True:
    reduced = fit(selected)
    candidates = []
    for name in predictors:
        if name in selected:
            continue
        full = fit(selected + [name])
        statistic, pvalue, df = full.compare_f_test(reduced)
        row = dict(step=len(selected)+1, candidate=name, F=float(statistic),
                   p=float(pvalue), reduced_SSE=float(reduced.ssr),
                   full_SSE=float(full.ssr), denominator_df=int(full.df_resid))
        candidates.append(row)
    if not candidates:
        break
    best = min(candidates, key=lambda row: row['p'])
    enter = best['p'] < 0.10
    for row in candidates:
        row['entered'] = enter and row['candidate'] == best['candidate']
    tests.extend(candidates)
    path.append(dict(best, decision='enter' if enter else 'stop'))
    if not enter:
        break
    selected.append(best['candidate'])

pd.DataFrame(tests).to_csv(ROOT / 'forward_candidate_tests.csv', index=False)
pd.DataFrame(path).to_csv(ROOT / 'forward_path.csv', index=False)
reduced, full = fit(['Accts']), fit(['Accts', 'Adver'])
partial_F = ((reduced.ssr-full.ssr)/1)/(full.ssr/full.df_resid)
assert np.isclose(partial_F, full.tvalues['Adver']**2)
backward = ['Time', 'MktPoten', 'Adver', 'MktShare', 'Change']
result = dict(forward_predictors=selected, backward_predictors=backward,
              nested=(set(selected) <= set(backward) or set(backward) <= set(selected)),
              advertising_after_accounts=dict(n=len(data), reduced_SSE=float(reduced.ssr),
                  full_SSE=float(full.ssr), numerator_df=1,
                  denominator_df=int(full.df_resid), F=float(partial_F),
                  p=float(f.sf(partial_F, 1, full.df_resid))))
(ROOT / 'forward_results.json').write_text(json.dumps(result, indent=2)+'\n')
print(pd.DataFrame(path)[['step', 'candidate', 'F', 'p', 'decision']].to_string(index=False))
print(json.dumps(result, indent=2))
