# Title: Lecture 18: Time-series regression and forecasting

**Instructor**: This replay shows a completed Codex CLI conversation. Human prompts were simulated for teaching. The assistant chose and executed the analyses; its wording is preserved with portable file links. Instructor notes and discussion pauses were added separately. Playback displays saved computations.

**Instructor**: By the end, you should be able to do the following independently: Define a forecast origin and horizon; interpret trend and seasonal coefficients; explain log back-transformation; evaluate forecasts in time order; diagnose residual dependence and explain an AR(1) correction. Pause at each question before revealing the response.

**Human**: Now remind us of the hotel forecasting question and show us the data. What changes when observations arrive in time order?

**Instructor**: At the end of month 156, which observations are available for forecasting the next year? Explain why a random split could answer a different question.

<!-- checkpoint: 1. Forecasts in time order -->
**Assistant**: The hotel question is: **Using past monthly occupancy, how well can we forecast the next 12 months?**

We have **168 consecutive monthly observations**, covering 14 sequential years. Each value is the **monthly average number of rooms occupied per day across four hotels**. These are textbook observations, not constructed data and not monthly room-nights.

Month \(t=1\) is January of year 1. The actual calendar years are unspecified. Here is the beginning:

| \(t\) | Month, year 1 | Average rooms occupied per day |
|---|---|---:|
| 1 | January | 501 |
| 2 | February | 488 |
| 3 | March | 504 |
| 4 | April | 578 |
| 5 | May | 545 |
| 6 | June | 632 |

![Monthly hotel occupancy in time order, with the final year reserved](/examples/regression-forecasting-v1/hotel_01_monthly_series_and_holdout.png)

The development series shows an upward movement and a recurring annual pattern. Those are features a forecasting model should investigate. The final year is shaded, with its values hidden to preserve the evaluation.

**Time order changes what counts as an honest prediction.** For the independent-store problem, randomly assigned validation folds represented predicting other stores from the same population. Here, we must predict later months using information available earlier.

That changes three things:

- **The split follows time.** We develop with months \(1\)–\(156\) and reserve months \(157\)–\(168\) for final evaluation.
- **Validation imitates forecasting.** Fit an earlier block, predict the following 12 months, then move the forecast origin forward and expand the training history. Random folds would let future observations influence predictions of the past.
- **Errors may be connected over time.** An unexpectedly busy month could be followed by another unexpectedly busy month. Trend and seasonal terms alone may not remove that dependence.

The forecast horizon also matters: predicting the next month and predicting a full year ahead are different tasks. Any preprocessing, model selection, or use of lagged occupancy must respect what is known at each forecast origin.

The data come from Bowerman, O’Connell, and Koehler, *FTR*, fourth edition, Table 6.4, printed page 297.

[Data](/examples/regression-forecasting-v1/hotel_monthly.csv) · [Plotting code](/examples/regression-forecasting-v1/hotel_01_show_ordered_series.py) · [Fixed time split](/examples/regression-forecasting-v1/hotel_01_fixed_time_split.csv)

**Instructor**: The forecast origin is the last observed time, and the horizon counts periods ahead. A trend is a sustained movement in level. Seasonality repeats at a known period; a cycle need not have a fixed period. Irregular variation is what remains unexplained. Here the observations are monthly averages of daily occupancy, not monthly room-nights. The sequential year labels do not assert actual calendar years.

**Human**: Can a constant, linear, or quadratic trend describe these hotel data? Show us what the trend leaves unexplained.

**Instructor**: Would adding a quadratic trend remove a pattern that repeats every August? Predict what the residual plot should show if seasonality remains.

<!-- checkpoint: 2. Trend and residual structure -->
**Assistant**: **A linear trend captures the long-run increase, but none of these trend-only models describes the recurring annual pattern. Adding a quadratic term contributes almost nothing.**

Using only development months \(t=1,\ldots,156\), the fitted trends are:

\[
\begin{aligned}
\text{Constant:}\quad \hat y_t&=710.54,\\
\text{Linear:}\quad \hat y_t&=554.58+1.987t,\\
\text{Quadratic:}\quad \hat y_t&=555.87+1.938t+0.000313t^2.
\end{aligned}
\]

The linear fit increases by about **1.99 rooms occupied per day for each additional month**. This describes the underlying trend, not the change expected between every pair of consecutive months.

To compare forecasting performance, we trained through months 96, 108, 120, 132, and 144, forecasting the following 12 months each time.

| Trend | Development fit RMSE | 12-month validation RMSE |
|---|---:|---:|
| Constant | 136.79 | 176.96 |
| Linear | 103.47 | 118.42 |
| Quadratic | 103.47 | 118.46 |

Errors are in **rooms occupied per day**. The linear trend improves substantially over a constant level. The quadratic curve nearly overlaps the line and gives no validation improvement.

![Trend fits and the seasonal patterns remaining in their residuals](/examples/regression-forecasting-v1/hotel_02_trends_and_unexplained_patterns.png)

A residual is \(e_t=y_t-\hat y_t\). If trend captured the systematic structure adequately, we would want the residuals to have little predictable pattern. Here, they leave clear structure:

- **Seasonality:** After the quadratic fit, August residuals average about **+210 rooms/day**, while November residuals average about **−112**. The model repeatedly underpredicts summer occupancy and overpredicts some quieter months.
- **Dependence over time:** The residual autocorrelation at lag 12 is about **0.91**. Similar deviations recur a year apart. Adjacent residuals also move together.
- **Large remaining errors:** Even with trend removed, development RMSE remains about **103 rooms/day**.

The autocorrelation does not yet establish that we need a separate autoregressive error model. An omitted seasonal pattern can itself create strong residual dependence.

**The linear trend is a useful starting point, but it is incomplete.** The residuals point to month-of-year effects that a trend alone cannot represent. We have not evaluated the reserved final year.

[Code](/examples/regression-forecasting-v1/hotel_02_compare_trends_and_residuals.py) · [Comparison results](/examples/regression-forecasting-v1/hotel_02_trend_comparison.csv) · [Monthly residual means](/examples/regression-forecasting-v1/hotel_02_monthly_residual_means.csv)

**Instructor**: A polynomial trend can represent smooth long-run curvature. A stable month-of-year pattern needs a seasonal representation. Residual dependence can come from an omitted mean pattern, so investigate trend and seasonality before attributing it entirely to an autoregressive error. The plotted lag-12 ACF uses lagged residual products divided by the full residual sum of squares; a Pearson correlation of truncated lagged vectors is a different finite-sample calculation.

**Human**: How can we model the recurring months? Show whether a log transformation helps, and explain the seasonal coefficients.

**Instructor**: Why use eleven monthly indicators with an intercept? If seasonal swings grow with the level, what would a log transformation change?

<!-- checkpoint: 3. Seasonal indicators and logs -->
**Assistant**: **Month indicators capture the recurring annual pattern, and fitting on the log scale improves validation forecasts further.**

We add 11 indicators to the linear trend, using January as the reference month:

\[
y_t=\beta_0+\beta_1t+
\sum_{m=2}^{12}\gamma_m I(\text{month}_t=m)+e_t.
\]

Each indicator is 1 in its named month and 0 otherwise. January needs no indicator because its baseline is already represented by the intercept. Including all 12 indicators alongside an intercept would create redundancy.

We compared this model with its log version:

\[
\log y_t=\beta_0+\beta_1t+
\sum_{m=2}^{12}\gamma_m I(\text{month}_t=m)+e_t.
\]

Using the same five expanding-window, 12-month validation blocks:

| Model | Validation RMSE, rooms/day |
|---|---:|
| Linear trend only | 118.42 |
| Trend + month indicators, level scale | 27.73 |
| Trend + month indicators, log scale | **20.11** |

These errors are all measured on the **original occupancy scale**. For the log model, I exponentiated forecasts and applied a smearing correction estimated only from each training block. This accounts for the fact that exponentiating a predicted log value alone does not generally estimate mean occupancy.

![Seasonal fits and residuals from the log model](/examples/regression-forecasting-v1/hotel_03_seasonal_fits_and_log_residuals.png)

**The seasonal coefficients have different meanings on the two scales.**

| Month | Level model: difference from January | Log coefficient | Log model: percentage difference |
|---|---:|---:|---:|
| February | −42.0 rooms/day | −0.0678 | −6.6% |
| July | +253.8 rooms/day | 0.3330 | +39.5% |
| August | +276.0 rooms/day | 0.3561 | +42.8% |
| November | −45.4 rooms/day | −0.0695 | −6.7% |

These comparisons adjust for the fitted time trend. For example, the log model’s August seasonal multiplier is

\[
e^{0.3561}\approx1.428,
\]

so August has a fitted seasonal level about **42.8% above January**, holding the trend term fixed. For actual months at different times, the trend difference also enters the prediction.

The level model assumes fixed seasonal differences in rooms/day. The log model assumes fixed seasonal ratios, allowing seasonal swings to grow as occupancy grows.

**The log seasonal model is the stronger candidate so far**, but its residuals still show temporal structure. Better seasonal fit does not establish independent errors. The final 12 months remain reserved.

[Code](/examples/regression-forecasting-v1/hotel_03_seasonal_indicators_and_logs.py) · [Validation results](/examples/regression-forecasting-v1/hotel_03_seasonal_validation_results.csv) · [Seasonal coefficients](/examples/regression-forecasting-v1/hotel_03_seasonal_coefficients.csv)

**Instructor**: The recorded model uses January as its reference; the legacy example used December. Either reference gives the same fitted values when the full seasonal model is reparameterized consistently. A log coefficient is a difference in log occupancy, so exp(gamma) is a seasonal ratio. The raw power y^lambda tends to 1 as lambda tends to zero; the normalized expression (y^lambda-1)/lambda tends to log(y). A common smearing factor assumes the residual distribution is sufficiently stable across the predictor values; investigate changing spread and dependence before treating it as a guaranteed mean correction.

**Human**: Walk us through one forecast from an earlier origin. How does this model compare with simply repeating last year?

**Instructor**: Use these recorded coefficients to make an independent calculation: intercept 6.233132, monthly trend 0.00290252 and August effect 0.355756. From origin 120, what is the horizon for month 128, and what is its predicted log occupancy? The training smearing factor is 1.000169. Explain how to return to rooms per day before revealing the answer.

<!-- checkpoint: 4. A forecast and its benchmark -->
**Assistant**: **The log seasonal model forecasts better overall than repeating last year, but it does not win for every month.**

Imagine standing just after **month 120**, December of year 10. We fit the model using only months 1–120 and issue all 12 forecasts for the following year. We do not update those forecasts as new observations arrive.

Consider **August of year 11**, month \(128\), eight months ahead. The fitted log forecast is

\[
\begin{aligned}
\widehat{\log y_{128}}
&=\text{intercept}+\text{trend}+\text{August effect}\\
&=6.233132+(0.00290252)(128)+0.355756\\
&\approx6.960411.
\end{aligned}
\]

January is the seasonal reference. August’s coefficient adds \(0.355756\) on the log scale.

Returning to rooms/day, including the correction estimated from training residuals,

\[
\hat y_{128}
=e^{6.960411}\times1.000169
\approx\boxed{1054.24}.
\]

**Repeating last year**, also called the seasonal-naive forecast, simply uses

\[
\hat y_{128}=y_{116}=\boxed{1040}.
\]

The occupancy observed later was **1038 rooms/day**. For this particular month, repeating last year wins: its error is only 2 rooms/day, compared with 16.24 for the model.

![Forecasts made after month 120 compared with observations that arrived later](/examples/regression-forecasting-v1/hotel_04_origin_120_forecast_comparison.png)

One month cannot settle the comparison. Over the complete forecast year and across our earlier validation origins:

| Evaluation period | Log seasonal RMSE | Repeat-last-year RMSE |
|---|---:|---:|
| Months 121–132 | **24.69** | 28.88 |
| All five 12-month validation blocks | **20.11** | 30.27 |

Errors are in rooms/day. The log seasonal model has lower RMSE in each of the five validation blocks. However, for months 121–132, repeating last year has a slightly lower mean absolute error: **19.42 versus 19.61**. The model’s advantage there comes from reducing larger errors.

The methods use different information. Repeating last year preserves each month’s most recent observed value. The regression pools information across years and projects its estimated growth forward.

**The log seasonal model earns its improvement against a meaningful baseline, while the baseline remains competitive for individual months.** These are earlier-origin validation results; months 157–168 remain reserved.

[Code](/examples/regression-forecasting-v1/hotel_04_earlier_origin_and_seasonal_naive.py) · [Comparison results](/examples/regression-forecasting-v1/hotel_04_seasonal_naive_comparison.csv) · [Forecast calculation](/examples/regression-forecasting-v1/hotel_04_august_forecast_calculation.json)

**Instructor**: The forecast horizon is eight months, but the trend input remains the absolute month index 128 used in fitting. A forecast error uses an outcome observed after the forecast was issued. Fitted residuals use observations that already influenced the model. Neither one favorable month nor a small training residual settles a forecasting comparison.

**Human**: Consider this constructed claim: “The seasonal model predicts well, so its errors must be independent.” Does the residual evidence support that?

**Instructor**: Constructed claim: “Good forecast accuracy proves independent errors.” Identify what evidence would address this claim and what evidence would not.

<!-- checkpoint: 5. Checking the independence claim -->
**Assistant**: **No. Good prediction does not imply independent errors, and the residual evidence shows remaining dependence.**

For the log seasonal model, the residual is

\[
e_t=\log(y_t)-
\left[\hat\beta_0+\hat\beta_1t+
\sum_{m=2}^{12}\hat\gamma_m I(\text{month}_t=m)\right].
\]

It measures what remains after accounting for the fitted growth trend and recurring months. Using only development months 1–156:

| Diagnostic | Result |
|---|---:|
| Residual autocorrelation, lag 1 | **0.374** |
| Residual autocorrelation, lag 2 | **0.232** |
| Residual autocorrelation, lag 12 | **0.307** |
| Longest run of residuals with the same sign | **8 months** |

![Time-ordered residuals, adjacent residual pairs, and residual autocorrelation](/examples/regression-forecasting-v1/hotel_05_log_seasonal_residual_dependence.png)

The left panel shows runs above and below zero. The middle panel shows that a positive residual often follows another positive residual, and a negative residual often follows another negative residual. The right panel shows positive association both between nearby months and between months a year apart.

The shaded band, approximately \(\pm0.157\), is a rough white-noise reference. It is a visual guide, not an exact significance test for fitted regression residuals.

**Forecast accuracy and error independence answer different questions.** Validation RMSE measures the size of forecast errors. Independence asks whether knowing one error provides information about another. A model can capture most of the trend and seasonality, predict reasonably well, and still leave persistent deviations.

That persistence matters. It suggests that recent departures from the fitted seasonal curve might help forecast subsequent months. It also means standard errors and prediction intervals calculated under independent-error assumptions can be misleading. The intervals need not have the right width or joint behavior across a forecast year.

Residuals are estimates of unobserved errors, so these plots cannot prove a particular dependence mechanism. The lag-12 pattern could also reflect seasonal structure that the fixed month effects have not fully captured.

**The supported conclusion is that independent errors are an inadequate working assumption for this fit.** An AR(1) error correction is a reasonable candidate to investigate, but these diagnostics alone do not establish that it will capture all the remaining structure or improve forecasts. The final year remains reserved.

[Code](/examples/regression-forecasting-v1/hotel_05_check_residual_independence.py) · [Diagnostic results](/examples/regression-forecasting-v1/hotel_05_residual_diagnostic_results.json) · [Residual autocorrelations](/examples/regression-forecasting-v1/hotel_05_log_seasonal_residual_acf.csv)

**Instructor**: Runs of same-sign residuals and an ACF are diagnostic evidence. Durbin-Watson is not an automatic proof of independence when it is near two. A formal table-based test has lower and upper bounds and can be inconclusive. For a two-sided test, non-rejection requires both d and 4-d to exceed the appropriate upper bound; one condition alone is insufficient. The executed diagnostic file gives d=1.2383, computed as the sum of squared successive residual differences divided by the residual sum of squares. This is consistent with the positive lag-one association, but no table-based significance decision is claimed here.

**Human**: Would an AR(1) error correction help? Show how it changes a forecast and its uncertainty.

**Instructor**: If the most recent log residual is positive and 0 < rho < 1, predict the sign of the next forecast correction and what happens as the horizon grows. Does correcting dependence guarantee better point forecasts?

<!-- checkpoint: 6. AR(1) forecast correction; optional derivation -->
**Assistant**: **AR(1) removes much of the immediate residual dependence, but it does not improve our 12-month validation forecasts here.**

We retain the log trend and month indicators, then model their errors as

\[
\log y_t=x_t^\top\beta+u_t,\qquad
u_t=\rho u_{t-1}+\varepsilon_t.
\]

The new term lets a recent departure from the seasonal curve persist into the future. Regression coefficients and error persistence are estimated together.

At our earlier origin \(T=120\), the estimates are

\[
\hat\rho=0.371,\qquad \hat u_{120}=0.01736.
\]

Occupancy ended above the fitted seasonal curve. The \(h\)-month-ahead log forecast therefore includes

\[
\hat\rho^h\hat u_{120}.
\]

For the next month, this correction is \(0.00644\). It quickly decays toward zero.

| Forecast from month 120 | Independent errors | AR(1) errors |
|---|---:|---:|
| Next-month forecast | 723.8 | **729.4** |
| Approximate 95% prediction interval | 694.9–753.6 | **702.3–757.4** |
| Eight-month-ahead forecast | 1054.2 | 1055.0 |

Values are rooms occupied per day. Differences also reflect re-estimated regression coefficients and the conversion from logs to mean occupancy.

![Forecasts and approximate prediction intervals issued at month 120](/examples/regression-forecasting-v1/hotel_06_ar1_forecasts_and_prediction_intervals.png)

**Uncertainty now depends on the forecast horizon.** Conditional on the latest error and known parameters, future log-error variance is

\[
\operatorname{Var}(u_{T+h}\mid u_T)
=\sigma_\varepsilon^2\sum_{j=0}^{h-1}\rho^{2j}.
\]

One month ahead, only one new innovation enters. Farther ahead, additional innovations accumulate. Thus the AR(1) interval can be narrower initially; accounting for dependence does not automatically widen every interval. Seasonal occupancy levels also affect interval widths after exponentiation.

The displayed intervals include approximate regression-coefficient uncertainty, but omit uncertainty in the estimated \(\rho\) and assume independent Gaussian innovations. They are model-based approximations.

Across the same five earlier 12-month validation blocks:

| Model | RMSE |
|---|---:|
| Log seasonal, independent errors | **20.11** |
| Log seasonal with AR(1) | 20.41 |

At origin 120, lag-one autocorrelation falls from about **0.37 in fitted errors to −0.04 in innovations**. Some annual dependence remains.

**AR(1) helps describe short-term dependence, but these validation results do not support adopting it for better annual point forecasts.** Improved diagnostics and improved forecasting are separate requirements. The final year remains reserved.

[Code](/examples/regression-forecasting-v1/hotel_06_ar1_forecasts_and_uncertainty.py) · [Validation results](/examples/regression-forecasting-v1/hotel_06_ar1_validation_metrics.csv) · [Forecasts and intervals](/examples/regression-forecasting-v1/hotel_06_ar1_validation_forecasts_and_intervals.csv)

**Instructor**: The innovation is the new shock after accounting for the previous error. Stationary AR(1) needs absolute rho below one. A simple residual regression estimate, a lag-one sample correlation and a joint Gaussian estimate need not agree exactly. Here the code jointly estimates the regression and AR parameter using a stationary Gaussian model. The geometric variance sum is an optional derivation; keep the forecast correction and interpretation in the core route. The plotted intervals include coefficient uncertainty conditional on the estimated rho, but omit rho-estimation uncertainty and can still be miscalibrated when innovations remain dependent.

**Human**: Freeze the simpler log seasonal model and the repeat-last-year benchmark, then forecast the reserved year. What would you recommend, and what remains unresolved?

**Instructor**: Write the chosen forecasting rule and the benchmark before revealing the reserved year. Which result would support using the model, and what could one year never establish about interval coverage?

<!-- checkpoint: 7. Reserved-year evaluation and next decision -->
**Assistant**: **I recommend the simpler log seasonal model for forecasting a full year, while retaining repeat-last-year as a benchmark.** It improves aggregate accuracy on the reserved year, though it is less accurate in most individual months.

We froze both methods at month 156, fitted the log model using months 1–156, and saved all forecasts for months 157–168 before evaluating their outcomes. Neither method was updated during the forecast year.

| Reserved-year measure | Log seasonal model | Repeat last year |
|---|---:|---:|
| RMSE, rooms/day | **23.78** | 34.61 |
| Mean absolute error, rooms/day | **21.87** | 25.58 |
| Mean forecast minus actual, rooms/day | +14.07 | −23.92 |
| Largest absolute error, rooms/day | **39.47** | 64.00 |

The log model reduces RMSE by **31.3%** against the benchmark. Its reserved-year RMSE is somewhat higher than its earlier validation RMSE of 20.11 rooms/day.

![Frozen forecasts compared with occupancy in the reserved year](/examples/regression-forecasting-v1/hotel_07_reserved_year_forecast_comparison.png)

The comparison has an important qualification: the log model has the smaller absolute error in only **5 of 12 months**. Repeating last year is often close, but makes several larger mistakes. The regression’s overall advantage comes from limiting those larger errors.

The log model also **overpredicts by about 14 rooms/day on average**, while repeating last year underpredicts by about 24. That difference is worth monitoring, but one year does not justify changing the trend after seeing these results.

Several questions remain unresolved:

- **Uncertainty:** Earlier residuals showed dependence. Independent-error prediction intervals remain questionable, and AR(1) did not improve annual validation forecasts.
- **Stability:** The model assumes continued exponential growth and fixed seasonal ratios. This evaluation cannot establish that those patterns will persist.
- **Strength of evidence:** We have one final year, with potentially dependent errors. Its 12 months are not 12 independent replications.
- **Operational information:** Hotel capacity, bookings, prices, and special events are absent. We cannot assess occupancy rates or capacity constraints.

The evidence supports the log seasonal model as our preferred annual point-forecast method for this series. It does not establish universal superiority. The reserved year has now been evaluated; any revisions need fresh future observations for another untouched assessment.

[Code](/examples/regression-forecasting-v1/hotel_07_frozen_reserved_year_evaluation.py) · [Frozen forecasts](/examples/regression-forecasting-v1/hotel_07_frozen_reserved_year_forecasts.csv) · [Evaluation results](/examples/regression-forecasting-v1/hotel_07_reserved_year_metrics.csv)

**Instructor**: Point forecasts and calibrated uncertainty are separate deliverables. Remaining annual dependence motivates a later seasonal error model and additional forecast origins. Exponential smoothing and ARIMA are subsequent syllabus topics. This lesson has introduced time-series regression and an AR(1) error bridge; it does not complete the forecasting unit. End with a recommendation that names the forecast horizon, an observed comparison and one unresolved assumption.

**Instructor**: [Data attribution and reproduction](/examples/regression-forecasting-v1/DATA.md) · [Hotel data](/examples/regression-forecasting-v1/hotel_monthly.csv) · [Environment](/examples/regression-forecasting-v1/requirements.txt) · [Script order](/examples/regression-forecasting-v1/reproduction-order.txt). Each lesson uses its own explicitly reserved evaluation data.
