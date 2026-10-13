from pathlib import Path
import os
import json

ROOT = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(ROOT / '.mplcache')
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import statsmodels.api as sm

predictors = ['Time', 'MktPoten', 'Adver', 'MktShare', 'Change', 'Accts', 'WkLoad', 'Rating']
df = pd.read_csv(ROOT / 'sales_territory.csv')
assert list(df.columns) == ['Sales'] + predictors
numeric = df.apply(pd.to_numeric, errors='raise')
checks = {
    'rows': len(df), 'columns': len(df.columns),
    'missing_by_column': df.isna().sum().to_dict(),
    'nonfinite_values': int((~np.isfinite(numeric.to_numpy())).sum()),
    'duplicate_rows': int(df.duplicated().sum()),
    'constant_columns': [c for c in df if df[c].nunique() <= 1],
    'dtypes': df.dtypes.astype(str).to_dict(),
}
assert checks['nonfinite_values'] == 0
X = sm.add_constant(numeric[predictors], has_constant='add')
checks['design_rank'] = int(np.linalg.matrix_rank(X))
checks['design_columns'] = X.shape[1]
numeric.describe().T.to_csv(ROOT / 'data_summary.csv')
(ROOT / 'data_checks.json').write_text(json.dumps(checks, indent=2))
print(json.dumps(checks, indent=2))
print(numeric.agg(['min', 'max']).T)
model = sm.OLS(numeric['Sales'], X).fit()
ci = model.conf_int()
coef = pd.DataFrame({'estimate': model.params, 'standard_error': model.bse,
                     't': model.tvalues, 'p_value': model.pvalues,
                     'ci95_lower': ci[0], 'ci95_upper': ci[1]})
coef.to_csv(ROOT / 'full_regression_coefficients.csv', index_label='term')
metrics = {'formula': 'Sales ~ ' + ' + '.join(predictors), 'n': int(model.nobs),
           'residual_df': int(model.df_resid), 'r_squared': model.rsquared,
           'adjusted_r_squared': model.rsquared_adj,
           'residual_standard_error': float(np.sqrt(model.mse_resid)),
           'F': float(model.fvalue), 'F_p_value': float(model.f_pvalue)}
(ROOT / 'full_regression_metrics.json').write_text(json.dumps(metrics, indent=2))
print(coef.to_string())
print(json.dumps(metrics, indent=2))
