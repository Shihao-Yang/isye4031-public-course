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

## Turn 07: accuracy criteria and historical holdout
`cod_accuracy_turn07.py` uses the original 24 catches in `cod.csv`, excluding the instructor-supplied month 25. Full-series fitting compares separately optimized MAD, MSD, and MAPE with L0 fixed to the first-year mean. A chronological illustration selects alpha by first-year MSD, then scores rolling one-step forecasts on months 13–24 with alpha fixed and levels updated after each actual. Fixed-alpha comparators are also reported. Because these historical data have already been inspected, this split illustrates temporal separation but is not a new untouched test. Formula-based outputs: `cod_fitting_metrics_turn07.csv`, `cod_holdout_metrics_turn07.csv`, `cod_holdout_forecasts_turn07.csv`, and `cod_accuracy_turn07.png`. No random sampling or seed is used.

Source consistency note: bikes.csv is Table6.8 p318. bikes_ch8.csv is the DIFFERENT numerical series visibly printed in Figure8.9 p370 for the Chapter8 smoothing example. They agree in quarters1–8 but differ in quarters9–16. Use bikes_ch8.csv to reproduce that worked example; preserve bikes.csv for the earlier time plot. Neither series is silently corrected to the other.

## Turn 08: thermostat Holt illustration
`thermostat_holt_turn08.py` uses `thermostat.csv` (Table 8.1, source above). Least squares on weeks 1–26 initializes the time-zero level as the intercept and the initial trend as the slope. Holt updates then run from week 1, with alpha=0.20 and gamma=0.10. The early predictions use future information through initialization and are fitted values. Outputs: `thermostat_holt_init_turn08.csv`, `thermostat_holt_updates_turn08.csv`, and `thermostat_holt_turn08.png`. These are deterministic formula-based recreations, not proprietary software output or untouched future evaluation. No random sampling or seed is used.

## Turn 09: verify thermostat transcription and fit Holt parameters
Week 24 is 206 in `thermostat.csv` and in the turn-08 calculation CSV. The turn-08 displayed chat table mistakenly showed 188; the calculations used the correct 206, including the first-26 sum of 5139. `thermostat_fit_turn09.py` verifies every turn-08 sales input and original prediction. It fixes the same OLS L0 and B0, runs from time zero, and minimizes all-52 one-step SSE over alpha,gamma in [0,1], using a deterministic grid and multiple local starts. Outputs: `thermostat_fit_summary_turn09.csv`, `thermostat_fit_updates_turn09.csv`, `thermostat_forecasts_turn09.csv`, `thermostat_fit_turn09.png`. These are formula-based historical fits, not proprietary software results or new future validation. No random sampling or seed is used.
