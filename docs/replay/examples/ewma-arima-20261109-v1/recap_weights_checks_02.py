import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

alpha = 0.2
lags = list(range(16))
weights = [alpha*(1-alpha)**k for k in lags]
with open('cod.csv', newline='') as f:
    y = [float(r['y']) for r in csv.DictReader(f)]
# Retrospective illustration: fix alpha and initialization before validation.
level = y[0]
for value in y[1:18]:
    level = alpha*value + (1-alpha)*level
predictions = []
errors = []
for value in y[18:]:
    predictions.append(level)
    errors.append(value-level)
    level = alpha*value + (1-alpha)*level
results = {
    'constructed_alpha': alpha,
    'lag_weights_0_through_15': weights,
    'weight_on_most_recent_5': sum(weights[:5]),
    'validation_months': list(range(19,25)),
    'rolling_one_step_forecasts': predictions,
    'rolling_one_step_errors_actual_minus_forecast': errors,
    'MAE_tons': sum(abs(e) for e in errors)/len(errors),
    'RMSE_tons': (sum(e*e for e in errors)/len(errors))**0.5,
    'mean_error_tons': sum(errors)/len(errors),
    'naive_MAE_tons': sum(abs(y[i]-y[i-1]) for i in range(18,24))/6,
    'naive_RMSE_tons': (sum((y[i]-y[i-1])**2 for i in range(18,24))/6)**0.5,
}
with open('recap_weights_checks_02_results.json','w') as f:
    json.dump(results,f,indent=2)
fig, ax = plt.subplots(figsize=(9,3.6),constrained_layout=True)
ax.bar(lags,weights,color='#21618c',width=.7)
ax.set(xlabel='Lag k: 0 = newest observation',ylabel='Weight in updated level',
       title='EWMA weights fade geometrically (illustration: alpha = 0.20)')
ax.set_xticks(lags)
ax.grid(axis='y',alpha=.2)
ax.spines[['top','right']].set_visible(False)
fig.savefig('recap_weights_checks_02.png',dpi=170)
print(json.dumps(results,indent=2))
