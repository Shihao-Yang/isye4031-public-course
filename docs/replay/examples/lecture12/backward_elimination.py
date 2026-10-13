from pathlib import Path
import json
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / 'sales_territory.csv')
active = ['Time', 'MktPoten', 'Adver', 'MktShare', 'Change', 'Accts', 'WkLoad', 'Rating']
threshold = 0.10
history = []
step = 0
while True:
    fit = sm.OLS(df['Sales'], sm.add_constant(df[active], has_constant='add')).fit()
    p = fit.pvalues.drop('const')
    worst = p.idxmax()
    remove = worst if p[worst] > threshold else None
    row = {'step': step, 'formula': 'Sales ~ ' + ' + '.join(active),
           'removed_next': remove, 'p_values': fit.pvalues.to_dict(),
           'r_squared': fit.rsquared, 'adjusted_r_squared': fit.rsquared_adj,
           'residual_df': int(fit.df_resid)}
    history.append(row)
    if remove is None:
        break
    active.remove(remove)
    step += 1
(ROOT / 'backward_elimination_history.json').write_text(json.dumps(history, indent=2))
rows = [{'step': h['step'], 'removed_next': h['removed_next'], **h['p_values']} for h in history]
pd.DataFrame(rows).to_csv(ROOT / 'backward_elimination_pvalues.csv', index=False)
ci = fit.conf_int()
pd.DataFrame({'estimate': fit.params, 'standard_error': fit.bse,
              'p_value': fit.pvalues, 'ci95_lower': ci[0],
              'ci95_upper': ci[1]}).to_csv(ROOT / 'backward_final_coefficients.csv', index_label='term')
print(pd.DataFrame(rows).to_string(index=False))
print('Final coefficients:')
print(fit.params.to_string())
print('Final fit:', history[-1])
