# Example data
Numerical series from Bowerman, O’Connell and Koehler, Forecasting, Time Series, and Regression, fourth edition.
- cod.csv: Table 6.1, printed p282, monthly cod catch in tons, 24 months, month1 January. Bay City Seafood example. No obvious trend or seasonal pattern. Historical example, not future validation.
- thermostat.csv: Table8.1 p361, weekly thermostat units sold,52weeks. Table read DOWN columns then left to right. Original credit Brown(1962), p431, suggested by Abraham and Ledolter(1983). Week53=330 is a subsequent teaching observation.
- bikes.csv: Table6.8 p318, quarterly TRK-50 mountain bikes sold at a Swiss shop,16quarters. Quarter1 begins each year.
- tiger.csv: Table8.2 p377, quarterly Tiger Sports Drink sales in THOUSANDS of cases,32quarters. Quarter1 begins each year.
- cola.csv: Table7.1 p329, monthly Tasty Cola sales in HUNDREDS of cases,36months. Month1 January.
All calendar years unspecified. These are supplied historical teaching data, not new untouched tests. Numeric data have been transcribed from source tables; original pages and prose are not included.

## Turn 04: smoothing weights and response illustration
`cod_weights_turn04.csv`, `cod_step_turn04.csv`, and `cod_weights_response_turn04.png` are deterministic calculations from the simple exponential smoothing recursion in `cod_weights_turn04.py`. The step from an initial level of 350 tons to repeated observations of 400 tons is a constructed illustration, not observed cod catch or proprietary software output. No random sampling or seed is used.

## Turn 05: fitting alpha for cod
`cod_fit_alpha_turn05.py` uses `cod.csv` (source above) to minimize the sum of squared errors across months 1–24, with L0 fixed at the months 1–12 mean. Outputs are `cod_fit_updates_turn05.csv`, `cod_alpha_curve_turn05.csv`, `cod_forecast_turn05.csv`, and `cod_alpha_fit_turn05.png`. These are deterministic recreations under the stated SES formula, not proprietary software output. The fitted errors are not untouched future evaluation; both initialization and alpha use historical observations. No random sampling or seed is used.

## Turn 06: prediction intervals and month-25 update
The month-25 catch of 384 tons is supplied in the instructor's question; it is not in the original `cod.csv`. `cod_intervals_turn06.py` uses the turn-05 fit and an explicitly assumed additive, independent normal innovation model. Innovation variance is approximated by SSE/23; initialization and fitted-parameter uncertainty are excluded. Alpha and variance stay fixed after month 25. `cod_intervals_turn06.csv`, `cod_update_turn06.csv`, and `cod_intervals_update_turn06.png` are formula-based calculations, not proprietary software output or evidence of future interval coverage. No random sampling or seed is used.

Source consistency note: bikes.csv is Table6.8 p318. bikes_ch8.csv is the DIFFERENT numerical series visibly printed in Figure8.9 p370 for the Chapter8 smoothing example. They agree in quarters1–8 but differ in quarters9–16. Use bikes_ch8.csv to reproduce that worked example; preserve bikes.csv for the earlier time plot. Neither series is silently corrected to the other.
