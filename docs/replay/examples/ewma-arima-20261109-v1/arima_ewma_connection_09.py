import json
from pathlib import Path

fit=json.loads(Path('paper_towels_forecast_08_results.json').read_text())
theta=fit['theta_hat_plus_convention']
e=fit['last_error']
y120=(fit['last_observed_rolls']-100000)/10000
f120=y120-e
alpha=1+theta
via_ma=y120+theta*e
via_update=f120+alpha*e
result={
    'plus_sign_MA_theta_hat':theta,
    'equivalent_forecast_update_alpha':alpha,
    'week120_actual_recorded_units':y120,
    'week120_fitted_one_step_forecast_recorded_units':f120,
    'week120_fitted_error':e,
    'forecast_update_amount':alpha*e,
    'week121_forecast_via_MA_equation':via_ma,
    'week121_forecast_via_EWMA_form':via_update,
    'ordinary_EWMA_example':{'alpha':0.2,'equivalent_plus_sign_theta':0.2-1},
    'general_ARIMA_example_order':[2,1,1],
}
assert abs(via_ma-via_update)<1e-12
Path('arima_ewma_connection_09_results.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
