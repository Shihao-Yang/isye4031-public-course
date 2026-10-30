# Title: Lecture 19: Exponential smoothing and changing trends

**Instructor**: Lectures 19 and 20 devote two full lessons to smoothing. Stationary models and ARIMA follow in a later lesson. Work through this lesson in short sessions, writing your own answer before revealing each response.

**Instructor**: Learning goals. Calculate and interpret level and trend updates, explain the smoothing weights, fit smoothing constants, distinguish fitted errors from forecast errors, and update point forecasts and prediction intervals.

**Instructor**: This page replays an actual completed Codex CLI conversation. Human prompts were simulated for teaching; the assistant executed the calculations and produced the figures. Its responses are preserved, with only portable file links changed. Instructor notes and discussion pauses were added separately. Historical software reference values are labeled and are not represented as newly executed software.

**Human**: Remind us of the cod-catch forecasting problem and show us the data. What kind of forecast seems reasonable?

**Instructor**: Before revealing the plot, describe what evidence would make a trend or seasonal forecast preferable to a constant forecast. What are the observation frequency and units?

<!-- checkpoint: 1. Cod catch and a level forecast -->
**Assistant**: The **Bay City Seafood problem** is to forecast monthly cod catch, measured in tons, using the previous **24 months**. Month 1 is January; the calendar years are unspecified.

| Month | First year | Second year |
|---|---:|---:|
| January | 362 | 276 |
| February | 381 | 334 |
| March | 317 | 394 |
| April | 297 | 334 |
| May | 399 | 384 |
| June | 402 | 314 |
| July | 375 | 344 |
| August | 349 | 337 |
| September | 386 | 345 |
| October | 328 | 362 |
| November | 389 | 314 |
| December | 343 | 365 |

![Monthly cod catch fluctuates around a roughly constant level, with no clear trend or repeating seasonal pattern.](/examples/smoothing-selfstudy-v2/cod_data_turn01.png)

A **roughly level forecast** seems reasonable. The catches fluctuate between 276 and 402 tons, but there is no convincing sustained trend or repeating seasonal pattern. We have only two annual cycles, so evidence about seasonality is limited.

A simple starting point is the historical average. Let \(y_t\) denote the catch in month \(t\). Then

\[
\bar y=\frac{\sum_{t=1}^{24}y_t}{24}
=\frac{8431}{24}
=351.29\text{ tons}.
\]

Thus, a baseline forecast for the next month is **about 351 tons**. A flat forecast means our predicted average stays constant; actual catches will still fluctuate.

**Simple exponential smoothing** is a natural next method: it estimates the current level while giving recent catches more weight. We would need future evaluation to determine whether it improves on the historical-average baseline.

Source: Bowerman, O’Connell, and Koehler, *Forecasting, Time Series, and Regression*, fourth edition, Table 6.1, p. 282; attribution in [DATA.md](/examples/smoothing-selfstudy-v2/turn01_DATA.md). [Data](/examples/smoothing-selfstudy-v2/cod.csv) · [Plot and calculation script](/examples/smoothing-selfstudy-v2/cod_intro_turn01.py) · [Summary results](/examples/smoothing-selfstudy-v2/cod_summary_turn01.csv).

**Instructor**: The series has 24 monthly catches in tons. A time plot supports an initial model choice but cannot establish that a pattern will persist. The original cod-catch example anchors the first part of this lesson.

**Human**: How does simple exponential smoothing update the cod forecast? Start with the first-year average and alpha = 0.10, and walk us through the calculation.

**Instructor**: Starting from 360.6667 tons, the next catch is 276. Predict the direction and size of the update with alpha = 0.10.

<!-- checkpoint: 2. A forecast after the first year -->
**Assistant**: Simple exponential smoothing updates the estimated **level** after each new catch. With \(\alpha=0.10\), we move the forecast **10% of the way toward the latest observation**.

Let \(y_t\) be the actual catch in month \(t\), and \(L_t\) the estimated level after observing it. Then

\[
L_t=\alpha y_t+(1-\alpha)L_{t-1}
    =L_{t-1}+\alpha(y_t-L_{t-1}).
\]

The forecast for the next month is \(\hat y_{t+1\mid t}=L_t\). The notation means “forecast month \(t+1\), using information through month \(t\).”

**Start with the first-year average.** Months 1–12 total 4,328 tons, so

\[
L_{12}=\frac{4328}{12}=360.6667.
\]

Before observing month 13, our forecast is therefore **360.67 tons**.

**Month 13: the actual catch is 276 tons.**

\[
\begin{aligned}
\text{Forecast error}
&=y_{13}-L_{12}\\
&=276-360.6667=-84.6667,\\[2pt]
L_{13}
&=360.6667+0.10(-84.6667)\\
&=\boxed{352.20}.
\end{aligned}
\]

We overpredicted by 84.67 tons, so we lower the next forecast by 8.47 tons. The forecast for month 14 becomes **352.20 tons**.

**Month 14: the actual catch is 334 tons.**

\[
L_{14}=352.20+0.10(334-352.20)
      =352.20-1.82
      =\boxed{350.38}.
\]

**Month 15: the actual catch rises to 394 tons.**

\[
L_{15}=350.38+0.10(394-350.38)
      =350.38+4.362
      =\boxed{354.742}.
\]

Now the error is positive, so the next forecast rises.

Continuing the same calculation gives the following results. Calculations retain full precision; displayed values are rounded.

| Month \(t\) | Actual \(y_t\) | Forecast before observing \(y_t\) | Updated level \(L_t\): next month’s forecast |
|---|---:|---:|---:|
| 13 | 276 | 360.67 | 352.20 |
| 14 | 334 | 352.20 | 350.38 |
| 15 | 394 | 350.38 | 354.74 |
| 16 | 334 | 354.74 | 352.67 |
| 17 | 384 | 352.67 | 355.80 |
| 18 | 314 | 355.80 | 351.62 |
| 19 | 344 | 351.62 | 350.86 |
| 20 | 337 | 350.86 | 349.47 |
| 21 | 345 | 349.47 | 349.03 |
| 22 | 362 | 349.03 | 350.32 |
| 23 | 314 | 350.32 | 346.69 |
| 24 | 365 | 346.69 | **348.52** |

![Cod catches and simple exponential smoothing forecasts. With alpha equal to 0.10, the forecast responds gradually to changes in catch.](/examples/smoothing-selfstudy-v2/cod_ses_turn02.png)

The small \(\alpha\) makes the forecast respond gradually: each update gives **10% weight to the latest catch and 90% to the previous level**. After month 24, the month-25 forecast is **348.52 tons**. Without new observations, simple exponential smoothing uses that same level for all later forecast horizons.

These are calculations under the stated smoothing formula. The second-year forecasts use only earlier observations, but this historical illustration does not establish performance on new future data.

[Calculation script](/examples/smoothing-selfstudy-v2/cod_ses_turn02.py) · [Full update results](/examples/smoothing-selfstudy-v2/cod_ses_updates_turn02.csv) · [Data source attribution](/examples/smoothing-selfstudy-v2/turn02_DATA.md)

**Instructor**: The first calculation begins after month 12, so that first-year average is available when the forecast is issued. The next checkpoint recreates the source’s different time-zero fitting convention. Both are retained to show why the initialization must be stated.

**Human**: What changes if we use that average as the initial level at time zero and run through all 24 months? Show the full calculation and explain what those early errors mean.

**Instructor**: Can a number computed from months 1–12 have been available before month 1? Explain how this affects the interpretation of the first twelve fitted errors.

<!-- checkpoint: 3. The complete smoothing table -->
**Assistant**: The update formula stays the same, but **we now apply it starting in month 1**. This produces a slightly different final forecast, and the errors for months 1–12 become **fitted errors that use information from the future through initialization**.

Set the initial level to the first-year average:

\[
L_0=\frac{4328}{12}=360.6667,\qquad \alpha=0.10.
\]

For each month \(t\), calculate:

\[
\begin{aligned}
\hat y_{t\mid t-1}&=L_{t-1} &&\text{value predicted before the update},\\
e_t&=y_t-L_{t-1} &&\text{actual minus predicted},\\
L_t&=L_{t-1}+0.10e_t &&\text{updated level}.
\end{aligned}
\]

Here \(y_t\) is the observed catch, and \(L_t\) becomes the prediction for the next month.

For month 1:

\[
e_1=362-360.6667=1.3333,
\qquad
L_1=360.6667+0.10(1.3333)=360.80.
\]

For month 2:

\[
e_2=381-360.80=20.20,
\qquad
L_2=360.80+0.10(20.20)=362.82.
\]

For month 3:

\[
e_3=317-362.82=-45.82,
\qquad
L_3=362.82+0.10(-45.82)=358.238.
\]

Thus, a positive error raises the level; a negative error lowers it. Each adjustment is 10% of the error.

The full calculation follows. Values shown are rounded to two decimals, but the recursion uses full precision.

| Month | Catch \(y_t\) | Prior level \(L_{t-1}\) | Error \(e_t\) | Adjustment \(0.10e_t\) | Updated level \(L_t\) |
|---|---:|---:|---:|---:|---:|
| 1 | 362 | 360.67 | +1.33 | +0.13 | 360.80 |
| 2 | 381 | 360.80 | +20.20 | +2.02 | 362.82 |
| 3 | 317 | 362.82 | −45.82 | −4.58 | 358.24 |
| 4 | 297 | 358.24 | −61.24 | −6.12 | 352.11 |
| 5 | 399 | 352.11 | +46.89 | +4.69 | 356.80 |
| 6 | 402 | 356.80 | +45.20 | +4.52 | 361.32 |
| 7 | 375 | 361.32 | +13.68 | +1.37 | 362.69 |
| 8 | 349 | 362.69 | −13.69 | −1.37 | 361.32 |
| 9 | 386 | 361.32 | +24.68 | +2.47 | 363.79 |
| 10 | 328 | 363.79 | −35.79 | −3.58 | 360.21 |
| 11 | 389 | 360.21 | +28.79 | +2.88 | 363.09 |
| 12 | 343 | 363.09 | −20.09 | −2.01 | 361.08 |
| 13 | 276 | 361.08 | −85.08 | −8.51 | 352.57 |
| 14 | 334 | 352.57 | −18.57 | −1.86 | 350.72 |
| 15 | 394 | 350.72 | +43.28 | +4.33 | 355.04 |
| 16 | 334 | 355.04 | −21.04 | −2.10 | 352.94 |
| 17 | 384 | 352.94 | +31.06 | +3.11 | 356.05 |
| 18 | 314 | 356.05 | −42.05 | −4.20 | 351.84 |
| 19 | 344 | 351.84 | −7.84 | −0.78 | 351.06 |
| 20 | 337 | 351.06 | −14.06 | −1.41 | 349.65 |
| 21 | 345 | 349.65 | −4.65 | −0.47 | 349.19 |
| 22 | 362 | 349.19 | +12.81 | +1.28 | 350.47 |
| 23 | 314 | 350.47 | −36.47 | −3.65 | 346.82 |
| 24 | 365 | 346.82 | +18.18 | +1.82 | **348.64** |

![Simple exponential smoothing through all 24 months. The shaded first year supplies the initial average, so its errors are fitted errors.](/examples/smoothing-selfstudy-v2/cod_ses_turn03.png)

**What do the early errors mean?** Imagine standing just before month 1. We could not yet know \(L_0=360.6667\), because its calculation requires catches from months 1 through 12. Assigning that average to “time zero” does not make it available at time zero.

For example, the month-1 error of \(+1.33\) tons measures how closely the retrospectively initialized model matches month 1. It is **not an error from a forecast we could actually have issued then**. The same issue applies throughout months 1–12. These errors help describe the fitted smoothing calculation, but they cannot establish future forecasting accuracy.

From month 13 onward, the initialization uses only past observations. With \(\alpha=0.10\) fixed in advance, those errors can be interpreted as sequential one-step forecast errors in this historical exercise.

Previously, we assigned the average directly to \(L_{12}=360.6667\). Here, processing the first year gives \(L_{12}=361.0803\), which changes the starting point for month 13. The difference shrinks with each update because each carries forward only 90% of the previous difference.

The resulting month-25 forecast is **348.64 tons**, compared with **348.52 tons** under the previous initialization.

[Full-precision calculations](/examples/smoothing-selfstudy-v2/cod_ses_updates_turn03.csv) · [Python script](/examples/smoothing-selfstudy-v2/cod_ses_turn03.py) · [Source attribution](/examples/smoothing-selfstudy-v2/turn03_DATA.md)

**Instructor**: Check the first two updates yourself: 360.8000 and 362.8200. Initializing from the training sample is legitimate fitting. Its early fitted errors must not be described as forecasts issued before those observations existed.

**Human**: Why is it called exponential smoothing? Show how the weights and the response to a change depend on alpha.

**Instructor**: Sketch the weights for a small and a large alpha. Which forecast reacts more to a one-month spike? Which retains more influence from its starting level?

<!-- checkpoint: 4. Weights and responsiveness -->
**Assistant**: It is called **exponential smoothing because the weights on older observations decrease exponentially with their age**. The parameter \(\alpha\) controls both how quickly those weights decrease and how quickly the forecast responds to change.

Start with the level update:

\[
L_t=\alpha y_t+(1-\alpha)L_{t-1}.
\]

Here \(y_t\) is the latest catch, \(L_t\) is the updated level and next month’s forecast, and \(0<\alpha<1\).

Substitute the previous update into this equation:

\[
\begin{aligned}
L_t
&=\alpha y_t+(1-\alpha)\big[\alpha y_{t-1}+(1-\alpha)L_{t-2}\big]\\
&=\alpha y_t+\alpha(1-\alpha)y_{t-1}+(1-\alpha)^2L_{t-2}.
\end{aligned}
\]

Continuing back to the initial level gives

\[
\boxed{
L_t=\sum_{j=0}^{t-1}\alpha(1-\alpha)^j y_{t-j}
       +(1-\alpha)^tL_0.
}
\]

Thus, an observation \(j\) months old has direct weight

\[
w_j=\alpha(1-\alpha)^j.
\]

Each additional month of age multiplies its weight by \(1-\alpha\). This is geometric decay in discrete time, equivalently exponential decay because \((1-\alpha)^j=e^{j\log(1-\alpha)}\).

| Age of observation | \(\alpha=0.10\) | \(\alpha=0.30\) | \(\alpha=0.70\) |
|---|---:|---:|---:|
| Latest, \(j=0\) | 0.10000 | 0.30000 | 0.70000 |
| 1 month old | 0.09000 | 0.21000 | 0.21000 |
| 2 months old | 0.08100 | 0.14700 | 0.06300 |
| 3 months old | 0.07290 | 0.10290 | 0.01890 |
| 4 months old | 0.06561 | 0.07203 | 0.00567 |

The observation weights plus the initial-level weight sum to one:

\[
\sum_{j=0}^{t-1}\alpha(1-\alpha)^j+(1-\alpha)^t=1.
\]

For our cod initialization, remember that \(L_0\) itself contains the first-year observations. The formula shows their direct weights and the remaining weight on that initial average; it does not remove the early information leakage we discussed.

![Exponential smoothing weights and response to a sustained change. Larger alpha concentrates weight on recent observations and produces a faster response.](/examples/smoothing-selfstudy-v2/cod_weights_response_turn04.png)

To see the response to change, consider a **constructed example**: the level starts at 350 tons, then every new catch is 400 tons. These are illustrative values, not additional cod observations.

With \(\alpha=0.10\), the first two updates are

\[
L_1=350+0.10(400-350)=355,
\]

\[
L_2=355+0.10(400-355)=359.5.
\]

Each update closes 10% of the remaining gap. More generally, after \(k\) observations at the new value,

\[
\boxed{L_k=400-50(1-\alpha)^k.}
\]

The initial gap is 50 tons, and the fraction still remaining is \((1-\alpha)^k\).

| Observations since change | \(\alpha=0.10\) | \(\alpha=0.30\) | \(\alpha=0.70\) |
|---|---:|---:|---:|
| 0 | 350.00 | 350.00 | 350.00 |
| 1 | 355.00 | 365.00 | 385.00 |
| 2 | 359.50 | 375.50 | 395.50 |
| 5 | 370.48 | 391.60 | 399.88 |
| 10 | 382.57 | 398.59 | 400.00 |

**Small \(\alpha\)** spreads weight over more history, creating a smoother forecast that adapts slowly. **Large \(\alpha\)** concentrates weight on recent catches, adapting quickly while also reacting more strongly to temporary fluctuations.

A single unusual catch also has a fading effect: a change of \(d\) tons in one observation changes the updated level by \(\alpha d\), then by \(\alpha(1-\alpha)d\) after the next update, assuming subsequent observations stay the same.

[Python calculations](/examples/smoothing-selfstudy-v2/cod_weights_turn04.py) · [Weight results](/examples/smoothing-selfstudy-v2/cod_weights_turn04.csv) · [Change-response results](/examples/smoothing-selfstudy-v2/cod_step_turn04.csv) · [Attribution and illustration details](/examples/smoothing-selfstudy-v2/turn04_DATA.md)

**Instructor**: The finite-sample expansion includes the initial-state weight. At alpha = 0, the level never updates; at alpha = 1, the next forecast equals the last observation. The step-change illustration is constructed and is not additional cod data.

**Human**: How should we choose alpha for the cod data? Show the fitting criterion and the resulting forecast, keeping the time-zero initialization.

**Instructor**: What must be recomputed for every candidate alpha? Would a smaller sum of squared fitted errors establish better performance next year?

<!-- checkpoint: 5. Choosing alpha -->
**Assistant**: We can choose \(\alpha\) by **minimizing the sum of squared fitted errors**, keeping the time-zero level fixed at

\[
L_0=\frac{4328}{12}=360.6667\text{ tons}.
\]

For each candidate \(\alpha\), run through all 24 months:

\[
e_t(\alpha)=y_t-L_{t-1}(\alpha),
\qquad
L_t(\alpha)=L_{t-1}(\alpha)+\alpha e_t(\alpha).
\]

Here \(y_t\) is the observed catch, \(e_t\) is actual minus predicted, and \(L_t\) is the updated level. Our fitting criterion is

\[
\boxed{
\hat\alpha=\underset{0\leq\alpha\leq1}{\arg\min}
\underbrace{\sum_{t=1}^{24}
\left[y_t-L_{t-1}(\alpha)\right]^2}_{\mathrm{SSE}(\alpha)}.
}
\]

Every candidate requires its own complete sequence of updates, because changing \(\alpha\) changes all subsequent levels. Minimizing \(\mathrm{SSE}/24\), the mean squared error, gives the same answer.

The numerical minimum is

\[
\boxed{\hat\alpha\approx0.03435.}
\]

| \(\alpha\) | Sum of squared errors, tons² | Fitted RMSE, tons |
|---|---:|---:|
| 0 | 28,424.33 | 34.41 |
| **0.03435** | **28,089.14** | **34.21** |
| 0.10 | 28,735.11 | 34.60 |
| 0.30 | 33,155.54 | 37.17 |
| 0.50 | 38,364.24 | 39.98 |
| 1.00 | 55,906.78 | 48.26 |

Here \(\mathrm{RMSE}=\sqrt{\mathrm{SSE}/24}\), which expresses the fitted error size in tons.

![Historical squared-error criterion for alpha and the resulting fitted cod series and flat future forecast.](/examples/smoothing-selfstudy-v2/cod_alpha_fit_turn05.png)

The chosen update is approximately

\[
L_t=0.03435y_t+0.96565L_{t-1}.
\]

It moves the level only **3.44% of the way toward each new catch**. For example, the first update is

\[
L_1
=360.6667+0.0343532(362-360.6667)
=360.7125.
\]

After all 24 updates,

\[
L_{24}=354.5438.
\]

Therefore, the resulting forecast is

\[
\boxed{\hat y_{25\mid24}=354.54\text{ tons}.}
\]

With no trend or seasonality in this model, all later point forecasts made at month 24 are also **354.54 tons**.

The small fitted \(\alpha\) favors a stable level for these data. However, the criterion is fairly flat near its minimum: \(\alpha=0\), which never updates the initial level, has nearly the same fitted RMSE. The data do not support treating \(0.03435\) as a precisely determined ideal value.

**This is historical fitting, not an estimate of future accuracy.** The initial level uses months 1–12, and the choice of \(\alpha\) uses all 24 observations. Consequently, even the second-year errors now participate in model selection. To assess forecasting performance, we would choose \(\alpha\) using only training data and evaluate subsequent forecasts on observations excluded from that choice.

These results recreate the stated SES formula in Python. [Script](/examples/smoothing-selfstudy-v2/cod_fit_alpha_turn05.py) · [All 24 updates](/examples/smoothing-selfstudy-v2/cod_fit_updates_turn05.csv) · [Criterion values](/examples/smoothing-selfstudy-v2/cod_alpha_curve_turn05.csv) · [Forecasts](/examples/smoothing-selfstudy-v2/cod_forecast_turn05.csv) · [Source attribution](/examples/smoothing-selfstudy-v2/turn05_DATA.md)

**Instructor**: The fitted alpha is approximately 0.0343532. Keeping full precision reproduces the source’s final level of 354.5438. Using the displayed rounded alpha of 0.034 throughout gives a slightly different final level. The objective curve is shallow near its minimum.

**Human**: How uncertain are those forecasts, and what changes when month 25 brings a catch of 384 tons? Show the intervals and the updated forecast visually.

**Instructor**: A flat point forecast is shown for every horizon. Should its uncertainty also remain flat? Predict how a catch of 384 changes the next forecast when alpha is about 0.034.

<!-- checkpoint: 6. Intervals and a new observation -->
**Assistant**: The month-25 point forecast was **354.54 tons**, but individual catches can vary substantially around it. Under an additive normal error model, its **approximate 95% prediction interval is 286.05 to 423.04 tons**.

To calculate uncertainty, we need an error model in addition to the smoothing formula. Assume

\[
y_t=L_{t-1}+\varepsilon_t,
\qquad
L_t=L_{t-1}+\alpha\varepsilon_t,
\]

where the innovations \(\varepsilon_t\) are independent normal errors with mean zero and constant variance \(\sigma^2\).

Using our historical fit, estimate the innovation variance by

\[
s^2=\frac{\mathrm{SSE}}{24-1}
=\frac{28089.14}{23}
=1221.27,
\qquad
s=34.95\text{ tons}.
\]

The divisor 23 gives an approximate adjustment for fitting one parameter, \(\alpha\), while treating the initial level as fixed. This differs from the descriptive fitted RMSE of 34.21 tons, which used divisor 24.

For a forecast \(h\) months ahead, this model gives

\[
\operatorname{Var}(y_{24+h}-\hat y_{24+h\mid24})
=\sigma^2\left[1+(h-1)\alpha^2\right].
\]

The first term represents the future month’s own error. The remaining terms account for intervening errors that will change the level. Thus, an approximate 95% prediction interval is

\[
\boxed{
L_{24}\ \pm\ 1.96s\sqrt{1+(h-1)\alpha^2}.
}
\]

With \(\alpha=0.0343532\), the intervals widen slowly:

| Forecast made after month 24 | Horizon \(h\) | Point forecast | Approximate 95% interval |
|---|---:|---:|---:|
| Month 25 | 1 | 354.54 | [286.05, 423.04] |
| Month 26 | 2 | 354.54 | [286.01, 423.08] |
| Month 30 | 6 | 354.54 | [285.85, 423.24] |
| Month 36 | 12 | 354.54 | [285.61, 423.48] |

These are **prediction intervals for individual future catches**, not confidence intervals for the underlying level.

When month 25 brings **384 tons**, its forecast error is

\[
e_{25}=384-354.5438=29.4562\text{ tons}.
\]

The level updates to

\[
\begin{aligned}
L_{25}
&=L_{24}+\alpha e_{25}\\
&=354.5438+0.0343532(29.4562)\\
&=354.5438+1.0119\\
&=\boxed{355.5558\text{ tons}.}
\end{aligned}
\]

The catch is inside the previous prediction interval. Because \(\alpha\) is small, this above-forecast catch raises the next forecast by only **1.01 tons**.

![Prediction intervals before and after observing a month-25 catch of 384 tons. The updated forecast rises slightly, while substantial uncertainty remains.](/examples/smoothing-selfstudy-v2/cod_intervals_update_turn06.png)

Keeping \(\alpha\) and the variance estimate fixed, the updated forecasts are:

| Forecast made after month 25 | Horizon \(h\) | Point forecast | Approximate 95% interval |
|---|---:|---:|---:|
| Month 26 | 1 | **355.56** | **[287.06, 424.05]** |
| Month 27 | 2 | 355.56 | [287.02, 424.09] |
| Month 31 | 6 | 355.56 | [286.86, 424.25] |
| Month 37 | 12 | 355.56 | [286.62, 424.49] |

At a fixed horizon, the interval width stays the same because we have held the error variance fixed. For a fixed target month, observing month 25 also shortens the remaining horizon and slightly reduces uncertainty.

These intervals are conditional approximations. They omit uncertainty from estimating \(\alpha\) and the initial level, and rely on the assumed independent normal errors. Our historical fitting exercise has not established their actual future coverage.

[Python calculations](/examples/smoothing-selfstudy-v2/cod_intervals_turn06.py) · [Interval results](/examples/smoothing-selfstudy-v2/cod_intervals_turn06.csv) · [Month-25 update](/examples/smoothing-selfstudy-v2/cod_update_turn06.csv) · [Sources and assumptions](/examples/smoothing-selfstudy-v2/turn06_DATA.md)

**Instructor**: The third-horizon interval from origin 24 is [285.9689, 423.1188] tons. The original hand example uses rounded alpha = 0.034 with level 354.5438 and obtains 355.5453 after the new catch. The executed calculation retains the fitted alpha and obtains 355.5558. Both differences are rounding conventions. The divisor T−1 and normal intervals are the source’s plug-in approximation; initialization, parameter and model-selection uncertainty remain outside it. Optional depth: derive the horizon variance from the innovation equations.

**Human**: How do MAD, MSD, and MAPE judge forecast accuracy? Use the cod example to show why the smallest fitting error may not identify the best forecasting procedure.

**Instructor**: An analyst reports the alpha with the lowest full-sample MSE as the best forecast. What information is missing? Explain one situation where MAPE would be unsuitable.

<!-- checkpoint: 7. Forecast accuracy -->
**Assistant**: **MAD, MSD, and MAPE summarize different costs of forecast errors. Smaller values indicate greater accuracy on the observations being scored, but a small fitting error does not guarantee accurate future forecasts.**

Let \(e_t=y_t-\hat y_t\), where \(y_t\) is actual catch and \(\hat y_t\) is its prediction. For \(n\) scored months:

| Measure | Formula | Interpretation |
|---|---|---|
| **MAD**, mean absolute deviation | \(\displaystyle \frac1n\sum_t|e_t|\) | Average absolute error, in tons. |
| **MSD**, mean squared deviation | \(\displaystyle \frac1n\sum_t e_t^2\) | Average squared error, in tons². Gives large errors extra influence. |
| **MAPE**, mean absolute percentage error | \(\displaystyle \frac{100}{n}\sum_t\frac{|e_t|}{|y_t|}\) | Average absolute error as a percentage of actual catch. |

MSD is also commonly called MSE. Its square root is RMSE. MAPE is undefined when an actual value is zero and can be dominated by small actual values.

For a worked example, our \(\alpha=0.10\) calculation predicted **361.08 tons for month 13**, whose actual catch was **276 tons**:

\[
e_{13}=276-361.0803=-85.0803.
\]

That month contributes

\[
\begin{aligned}
|e_{13}|&=85.08\text{ tons},\\
e_{13}^2&\approx7238.66\text{ tons}^2,\\
100\frac{|e_{13}|}{276}&\approx30.83\%.
\end{aligned}
\]

Each measure averages its corresponding contributions over the scored months.

Even on the same historical data, the criteria can select different smoothing parameters. Keeping \(L_0=360.6667\) and fitting all 24 months gives:

| Criterion minimized | Selected \(\alpha\) | MAD, tons | MSD, tons² | MAPE |
|---|---:|---:|---:|---:|
| MAD | 0.01785 | **28.667** | 1173.19 | 8.551% |
| MSD | 0.03435 | 28.677 | **1170.38** | 8.537% |
| MAPE | 0.06239 | 28.731 | 1176.42 | **8.530%** |

MSD penalizes large misses strongly. MAPE gives a given error in tons more influence when actual catch is low. These differences explain why there is no single “smallest error” without specifying the criterion.

To see why fitting and forecasting differ, consider a chronological comparison:

1. Use months 1–12 to calculate \(L_0\) and select \(\alpha\) by minimizing first-year MSD.
2. Freeze \(\alpha\).
3. Predict month 13, observe its catch, update the level, and continue through month 24.

The first-year fit selects **\(\alpha=0\)**, which keeps the forecast at the initial average. Compare this procedure with a fixed \(\alpha=0.10\):

| Procedure | First-year fitting MSD | Second-year forecast MSD |
|---|---:|---:|
| First-year-selected \(\alpha=0\) | **1059.89** | 1308.81 |
| Fixed \(\alpha=0.10\) | 1166.44 | **1228.15** |

**The ranking reverses.** The parameter with the smaller training error gives the larger error on the following year.

![Second-year cod forecasts and their mean squared errors. The parameter selected for its first-year fit is outperformed in the second year by fixed alpha equal to 0.10.](/examples/smoothing-selfstudy-v2/cod_accuracy_turn07.png)

The second-year results also show that the choice of metric matters:

| Procedure | MAD, tons | MSD, tons² | MAPE |
|---|---:|---:|---:|
| First-year-selected \(\alpha=0\) | 29.139 | 1308.81 | 9.072% |
| Fixed \(\alpha=0.10\) | **27.925** | **1228.15** | 8.590% |
| Fixed \(\alpha=0.30\) | 27.926 | 1381.72 | **8.483%** |
| Last catch, \(\alpha=1\) | 43.833 | 2388.33 | 13.102% |

Here \(\alpha=0.10\) has the smallest MSD, while \(\alpha=0.30\) has the smallest MAPE.

We should choose an error measure that reflects the practical cost of mistakes, then evaluate the **entire forecasting procedure** on later observations excluded from parameter selection. Choosing \(\alpha\) on all 24 months and reporting its error on those same months measures historical fit.

This split illustrates the distinction using already-inspected historical data. It does not establish that \(\alpha=0.10\) will be best in future months.

[Python calculations](/examples/smoothing-selfstudy-v2/cod_accuracy_turn07.py) · [Fitting results](/examples/smoothing-selfstudy-v2/cod_fitting_metrics_turn07.csv) · [Second-year accuracy](/examples/smoothing-selfstudy-v2/cod_holdout_metrics_turn07.csv) · [Individual forecasts](/examples/smoothing-selfstudy-v2/cod_holdout_forecasts_turn07.csv) · [Source attribution](/examples/smoothing-selfstudy-v2/turn07_DATA.md)

**Instructor**: MAD here means mean absolute forecast error, also called MAE. MSD is MSE. The historical ranking reversal illustrates the distinction between fitting and forecasting; these already-inspected data are not a new untouched test. The source explicitly skips adaptive tracking-signal smoothing, and this lesson does too.

**Human**: Now show us the weekly thermostat sales. How does Holt’s method handle the changing trend? Initialize from the first 26 weeks and walk through the first updates with alpha = 0.20 and gamma = 0.10.

**Instructor**: Why would a flat multi-step forecast be unsatisfactory for the later weeks? What does the trend state measure, and why is the old level advanced by one trend increment before the update?

<!-- checkpoint: 8. Thermostat sales and Holt updates -->
**Assistant**: The thermostat series contains **52 weeks of sales**. Sales fluctuate around a fairly stable level in the first half, then develop an upward trend.

| Week | Sales | Week | Sales | Week | Sales | Week | Sales |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 206 | 14 | 189 | 27 | 172 | 40 | 255 |
| 2 | 245 | 15 | 244 | 28 | 210 | 41 | 303 |
| 3 | 185 | 16 | 209 | 29 | 205 | 42 | 282 |
| 4 | 169 | 17 | 207 | 30 | 244 | 43 | 291 |
| 5 | 162 | 18 | 211 | 31 | 218 | 44 | 280 |
| 6 | 177 | 19 | 210 | 32 | 182 | 45 | 255 |
| 7 | 207 | 20 | 173 | 33 | 206 | 46 | 312 |
| 8 | 216 | 21 | 194 | 34 | 211 | 47 | 296 |
| 9 | 193 | 22 | 156 | 35 | 273 | 48 | 307 |
| 10 | 230 | 23 | 206 | 36 | 248 | 49 | 281 |
| 11 | 212 | 24 | 188 | 37 | 262 | 50 | 308 |
| 12 | 192 | 25 | 188 | 38 | 258 | 51 | 280 |
| 13 | 162 | 26 | 162 | 39 | 233 | 52 | 345 |

![Weekly thermostat sales, Holt predictions, and the changing estimated trend. The shaded first 26 weeks supply the initialization.](/examples/smoothing-selfstudy-v2/thermostat_holt_turn08.png)

**Holt’s method tracks both a level and a trend.** Let

- \(y_t\): sales in week \(t\), in units;
- \(L_t\): estimated level after observing week \(t\);
- \(B_t\): estimated trend, in units per week.

With \(\alpha=0.20\) for level and \(\gamma=0.10\) for trend,

\[
\boxed{L_t=0.20y_t+0.80(L_{t-1}+B_{t-1})}
\]

\[
\boxed{B_t=0.10(L_t-L_{t-1})+0.90B_{t-1}.}
\]

The first equation compares actual sales with the old level **advanced by one week’s trend**. The second smooths the observed change in level to update the trend.

The forecast \(h\) weeks ahead is

\[
\boxed{\hat y_{t+h\mid t}=L_t+hB_t.}
\]

To initialize, I’ll continue our **time-zero convention**: fit a straight line to weeks 1–26, use its intercept as \(L_0\), and use its slope as \(B_0\). Then run the smoothing updates from week 1.

For those first 26 observations,

\[
\bar t=13.5,\qquad \bar y=\frac{5139}{26}=197.6538.
\]

The least-squares slope is

\[
B_0=
\frac{\sum_{t=1}^{26}(t-\bar t)(y_t-\bar y)}
{\sum_{t=1}^{26}(t-\bar t)^2}
=\frac{-538.5}{1462.5}
=-0.368205.
\]

The intercept is

\[
L_0=\bar y-B_0\bar t
=197.6538-(-0.368205)(13.5)
=202.624615.
\]

Thus, we start with a level of **202.62 units** and a slight downward trend of **0.37 units per week**.

**Week 1: \(y_1=206\).** Before observing it, the prediction is

\[
\hat y_{1\mid0}=L_0+B_0
=202.624615-0.368205
=202.256410.
\]

Update the level:

\[
L_1=0.20(206)+0.80(202.256410)
=203.005128.
\]

The level increased by \(203.005128-202.624615=0.380513\), so update the trend:

\[
B_1=0.10(0.380513)+0.90(-0.368205)
=-0.293333.
\]

The next prediction is

\[
\hat y_{2\mid1}=L_1+B_1
=\boxed{202.711795}.
\]

**Week 2: \(y_2=245\).** Sales are well above that prediction:

\[
L_2=0.20(245)+0.80(202.711795)
=211.169436,
\]

\[
B_2=0.10(211.169436-203.005128)
     +0.90(-0.293333)
=0.552431.
\]

The estimated trend turns positive, giving

\[
\hat y_{3\mid2}=211.169436+0.552431
=\boxed{211.721867}.
\]

**Week 3: \(y_3=185\).** This lower observation pulls both estimates back:

\[
L_3=0.20(185)+0.80(211.721867)=206.377493,
\]

\[
B_3=0.10(206.377493-211.169436)+0.90(0.552431)
=0.017993.
\]

The next prediction becomes \(206.395487\) units.

| Week | Actual sales | Prediction before update | Updated level \(L_t\) | Updated trend \(B_t\) | Next-week prediction |
|---|---:|---:|---:|---:|---:|
| 1 | 206 | 202.26 | 203.01 | −0.293 | 202.71 |
| 2 | 245 | 202.71 | 211.17 | +0.552 | 211.72 |
| 3 | 185 | 211.72 | 206.38 | +0.018 | 206.40 |
| 26 | 162 | 195.47 | 188.78 | −1.145 | 187.63 |
| 27 | 172 | 187.63 | 184.50 | −1.458 | 183.05 |
| 28 | 210 | 183.05 | 188.44 | −0.919 | 187.52 |
| 52 | 345 | 309.09 | 316.27 | +4.706 | 320.98 |

As the later sales rise, Holt’s trend estimate gradually becomes positive. By week 52, it estimates growth of about **4.71 units per week**, giving a week-53 forecast of

\[
\hat y_{53\mid52}=316.274957+4.705948
=\boxed{320.98\text{ units}}.
\]

Unlike simple exponential smoothing’s flat forecast, Holt’s future forecasts follow the latest estimated slope. New observations can keep changing that slope.

As with our cod time-zero initialization, predictions for weeks 1–26 are **fitted values**: their starting estimates use those same 26 observations. These calculations recreate the stated formulas and do not establish future forecasting accuracy.

Source: Bowerman, O’Connell, and Koehler, *Forecasting, Time Series, and Regression*, fourth edition, Table 8.1, p. 361; details in [DATA.md](/examples/smoothing-selfstudy-v2/turn08_DATA.md). [Sales data](/examples/smoothing-selfstudy-v2/thermostat.csv) · [Python script](/examples/smoothing-selfstudy-v2/thermostat_holt_turn08.py) · [All updates](/examples/smoothing-selfstudy-v2/thermostat_holt_updates_turn08.csv).

**Instructor**: Recorded table correction: weeks 21–24 are 194, 234, 156, and 206. The displayed table misplaces those entries; the data file, plotted observations, initial regression and numerical calculations use the correct values. The next recorded question checks the discrepancy. Attribution correction: Table 8.1 is on printed page 359, not page 361 as initially supplied to the recorded conversation. Verify the first update by hand: level 203.0051, trend −0.2933, next prediction 202.7118.

**Human**: Week 24 is 206 in the data, but your displayed table says 188. Check that discrepancy, then fit alpha and gamma and show how the fit and forecasts change.

**Instructor**: The fitted training error will improve when the parameters are optimized. What evidence would establish that the resulting forecasts improve?

<!-- checkpoint: 9. Fitting level and trend constants -->
**Assistant**: **Week 24 is 206 units. My displayed table was wrong; the calculations used the correct value.** I verified every sales input and prediction in the previous calculation file against the CSV.

The corrected entries are:

| Week | Sales |
|---|---:|
| 23 | 156 |
| 24 | **206** |
| 25 | 188 |
| 26 | 162 |

The first-26-week total remains **5,139**, so our time-zero initialization remains

\[
L_0=202.624615,\qquad B_0=-0.368205.
\]

To fit the smoothing parameters, hold those initial values fixed and minimize the squared one-step errors over all 52 weeks:

\[
\boxed{
(\hat\alpha,\hat\gamma)=
\underset{0\leq\alpha,\gamma\leq1}{\arg\min}
\sum_{t=1}^{52}
\left[y_t-(L_{t-1}+B_{t-1})\right]^2.
}
\]

For each candidate pair, rerun Holt’s updates from week 1:

\[
L_t=\alpha y_t+(1-\alpha)(L_{t-1}+B_{t-1}),
\]

\[
B_t=\gamma(L_t-L_{t-1})+(1-\gamma)B_{t-1}.
\]

Here \(L_t\) is the estimated sales level and \(B_t\) is the estimated change in units per week.

The numerical fit gives

\[
\boxed{\hat\alpha=0.24684,\qquad \hat\gamma=0.09505.}
\]

| Quantity | Original parameters | Fitted parameters |
|---|---:|---:|
| Level parameter \(\alpha\) | 0.20000 | **0.24684** |
| Trend parameter \(\gamma\) | 0.10000 | **0.09505** |
| Sum of squared errors | 39,182.47 | **38,884.24** |
| MSD, units² | 753.51 | **747.77** |
| MAD, units | 22.07 | **21.70** |
| MAPE | 10.06% | **9.92%** |

The fitted model puts somewhat more weight on each new observation when updating the level. Its trend smoothing parameter changes very little. The improvement in historical fit is modest.

![Original and fitted Holt predictions and forecasts for thermostat sales, with the estimated trend shown below. Dashed lines after week 52 are future point forecasts.](/examples/smoothing-selfstudy-v2/thermostat_fit_turn09.png)

After week 52, the estimated states are:

| State | Original | Fitted |
|---|---:|---:|
| Level \(L_{52}\), units | 316.275 | 315.946 |
| Trend \(B_{52}\), units per week | 4.706 | 4.504 |

Using \(\hat y_{52+h\mid52}=L_{52}+hB_{52}\), the fitted week-53 forecast is

\[
\hat y_{53\mid52}=315.945962+4.504033
=\boxed{320.45\text{ units}}.
\]

| Target week | Original forecast | Fitted forecast |
|---|---:|---:|
| 53 | 320.98 | **320.45** |
| 54 | 325.69 | **324.95** |
| 56 | 335.10 | **333.96** |
| 60 | 353.92 | **351.98** |

The fitted forecasts start slightly lower and rise more slowly. The difference grows with the forecast horizon because the fitted trend is smaller.

These are **historical fitting improvements**, not demonstrated improvements in future accuracy. Weeks 1–26 contribute to initialization, and all 52 weeks contribute to parameter selection. A later evaluation period would be needed to compare forecasting performance.

[Python script](/examples/smoothing-selfstudy-v2/thermostat_fit_turn09.py) · [Fit summary](/examples/smoothing-selfstudy-v2/thermostat_fit_summary_turn09.csv) · [All updates](/examples/smoothing-selfstudy-v2/thermostat_fit_updates_turn09.csv) · [Forecasts](/examples/smoothing-selfstudy-v2/thermostat_forecasts_turn09.csv) · [Source attribution and correction](/examples/smoothing-selfstudy-v2/turn09_DATA.md)

**Instructor**: The correction is an authentic recorded response to a table error, not a constructed failure. The full update CSV can be checked against the original data. The two-parameter objective keeps initial states fixed; different initialization defines a different fitting problem.

**Human**: Show the next three thermostat forecasts with prediction intervals. If week 53 sales are 330, how does the same forecast error update both level and trend, and change the forecasts?

**Instructor**: Predict the sign of both state adjustments after a week-53 sale of 330. Why should uncertainty about a trend matter more at longer horizons?

<!-- checkpoint: 10. Holt intervals and forecast revision -->
**Assistant**: Using the fitted Holt model, the next three forecasts are **320.45, 324.95, and 329.46 units**. Their prediction intervals widen with the horizon because future errors affect both the level and the trend.

Our fitted parameters and week-52 states are

\[
\alpha=0.246842,\quad \gamma=0.095055,\qquad
L_{52}=315.945962,\quad B_{52}=4.504033.
\]

The point forecast \(h\) weeks ahead is \(L_{52}+hB_{52}\).

| Target week | Horizon \(h\) | Forecast, units | Approximate 95% prediction interval |
|---|---:|---:|---:|
| 53 | 1 | **320.45** | **[265.79, 375.11]** |
| 54 | 2 | **324.95** | **[268.33, 381.57]** |
| 55 | 3 | **329.46** | **[270.61, 388.31]** |

These intervals assume independent, additive normal forecast innovations. Using the historical squared errors,

\[
s^2=\frac{38884.24}{52-2}=777.68,
\qquad s=27.89\text{ units}.
\]

This is an approximate adjustment for fitting two smoothing parameters, treating initialization as fixed. For our version of Holt’s method, the prediction standard error is

\[
s_h=s\sqrt{1+\sum_{j=1}^{h-1}
\left(\alpha+j\alpha\gamma\right)^2},
\]

and the interval is the point forecast \(\pm1.96s_h\). The sum is empty when \(h=1\). For example, week 53 has interval

\[
320.45\pm1.96(27.887)
=[265.79,\ 375.11].
\]

**Now observe week-53 sales of 330 units.** The forecast error is

\[
e_{53}=330-320.449995
=\boxed{9.550005\text{ units}}.
\]

To see how this same error updates both states, rewrite Holt’s equations as

\[
\boxed{L_t=L_{t-1}+B_{t-1}+\alpha e_t}
\]

and

\[
\boxed{B_t=B_{t-1}+\alpha\gamma e_t.}
\]

The second equation follows by substituting
\(L_t-L_{t-1}=B_{t-1}+\alpha e_t\) into the trend update. Thus, **the trend’s correction is \(\alpha\gamma e_t\)** under our notation.

For week 53, the level correction is

\[
\alpha e_{53}=0.246842(9.550005)=2.357341,
\]

so

\[
L_{53}=315.945962+4.504033+2.357341
=\boxed{322.807337}.
\]

The trend correction is

\[
\alpha\gamma e_{53}
=0.0234635(9.550005)=0.224076,
\]

so

\[
B_{53}=4.504033+0.224076
=\boxed{4.728109\text{ units per week}}.
\]

The above-forecast observation raises the estimated level and makes the projected growth slightly steeper.

![Holt forecasts and approximate prediction intervals before and after observing week-53 sales of 330. The updated forecasts rise and their slope increases.](/examples/smoothing-selfstudy-v2/thermostat_intervals_turn10.png)

Keeping the parameters and error variance fixed, the next three forecasts become:

| Target week | Previous forecast | Updated forecast | Updated 95% prediction interval |
|---|---:|---:|---:|
| 54 | 324.95 | **327.54** | **[272.88, 382.19]** |
| 55 | 329.46 | **332.26** | **[275.64, 388.88]** |
| 56 | 333.96 | **336.99** | **[278.14, 395.84]** |

For example,

\[
\hat y_{54\mid53}=L_{53}+B_{53}
=322.807337+4.728109=327.535446.
\]

For a target \(h\) weeks after week 53, the revision to its previous forecast is

\[
\boxed{(\alpha+h\alpha\gamma)e_{53}.}
\]

The level correction shifts every forecast upward; the trend correction adds an increasing amount at longer horizons. For week 54, the increase is \(2.357341+0.224076=2.581417\) units.

These are approximate prediction intervals for individual weekly sales. They omit uncertainty in initialization and fitted parameters, and their future coverage has not been verified.

[Python calculations](/examples/smoothing-selfstudy-v2/thermostat_intervals_turn10.py) · [Interval results](/examples/smoothing-selfstudy-v2/thermostat_intervals_turn10.csv) · [State update](/examples/smoothing-selfstudy-v2/thermostat_update_turn10.csv) · [Sources and assumptions](/examples/smoothing-selfstudy-v2/turn10_DATA.md)

**Instructor**: The error correction form gives level adjustment alpha times the error and trend adjustment alpha times gamma times the error. Gamma is not itself the direct error coefficient. The normal intervals condition on fitted states and smoothing parameters; they do not include every source of uncertainty. Optional depth: derive the sum of squared shock effects in the horizon variance.

**Human**: Why do the historical software reports give different thermostat forecasts? Compare their forecast plots with our calculation and show what changing initialization can do.

**Instructor**: If two programs use identical smoothing constants, must they give identical forecasts? Identify what else defines the calculation.

<!-- checkpoint: 11. Software and initialization -->
**Assistant**: The historical reports use different smoothing constants, and their internal starting states and interval calculations are not fully documented. **We can compare their reported forecasts directly, but we cannot attribute every difference to initialization.**

The local reference contains transcribed historical report values, not a current software run. All comparisons below use information through **week 52**, before observing week 53.

| Calculation | \(\alpha\) | \(\gamma\) | Week 53 | Week 54 | Week 55 |
|---|---:|---:|---:|---:|---:|
| Historical report: supplied constants | 0.247 | 0.095 | 320.397 | 324.890 | 329.383 |
| Historical report: automatic | 0.489596 | 0.066912 | 324.219 | 328.441 | 332.662 |
| Our fitted Holt calculation | 0.246842 | 0.095055 | 320.450 | 324.954 | 329.458 |

Our forecasts nearly match the supplied-constants report. The automatic report starts higher and projects a slightly smaller weekly increase.

![Left: transcribed historical forecast points and prediction intervals compared with our calculation. Right: forecasts obtained by changing the initial states and refitting the smoothing parameters.](/examples/smoothing-selfstudy-v2/thermostat_software_turn11.png)

The left panel redraws the **transcribed reference forecasts and intervals**. For week 53:

| Calculation | Forecast | Reported or calculated 95% interval |
|---|---:|---:|
| Historical: supplied constants | 320.397 | [264.389, 376.405] |
| Historical: automatic | 324.219 | [269.414, 379.025] |
| Our calculation | 320.450 | [265.792, 375.108] |

Our intervals use the additive normal error model explained previously. The historical interval convention is unspecified, so matching the point forecast does not imply matching its interval.

**How much can initialization explain?** Holt’s updates require four inputs: \(\alpha,\gamma,L_0,B_0\). Identical smoothing constants alone do not define an identical calculation.

I compared three explicit time-zero initializations, running each through all 52 weeks:

| Initialization | \(L_0\), units | \(B_0\), units/week |
|---|---:|---:|
| First-26-week regression, our original choice | 202.625 | −0.368 |
| First-26-week average, zero trend | 197.654 | 0 |
| First observation, zero trend | 206.000 | 0 |

First, hold the constants at the reported \(\alpha=0.247,\gamma=0.095\):

| Initialization | Fitted MSD | Week-53 forecast |
|---|---:|---:|
| First-26 regression | 747.774 | 320.447 |
| First-26 average, zero trend | 747.373 | 320.440 |
| First observation, zero trend | 749.360 | 320.447 |

These starting states change early fitted errors, but their direct effect on the final forecast has largely faded after 52 updates.

Next, **refit \(\alpha\) and \(\gamma\) for each initialization**, keeping the same all-52 squared-error criterion:

| Initialization | Fitted \(\alpha\) | Fitted \(\gamma\) | Week 53 | Week 54 | Week 55 |
|---|---:|---:|---:|---:|---:|
| First-26 regression | 0.24684 | 0.09505 | 320.450 | 324.954 | 329.458 |
| First-26 average, zero trend | 0.24717 | 0.08994 | 320.142 | 324.591 | 329.040 |
| First observation, zero trend | 0.25152 | 0.09237 | 320.296 | 324.765 | 329.233 |

Initialization now affects the forecasts through **two paths**: it changes the starting states, and it changes which smoothing parameters minimize historical error. The right panel shows this modest sensitivity.

There are useful limits to what this comparison establishes:

- **Known:** the automatic report uses substantially different constants. Using those constants with our initialization gives forecasts of **324.189, 328.397, and 332.605**, close to its reported values.
- **Known:** rounding our fitted constants to \(0.247,0.095\) changes the week-53 forecast from 320.450 to 320.447. Rounding alone does not explain its full difference from 320.397.
- **Unresolved:** undocumented starting states, treatment of early errors, parameter-selection rules, or interval conventions could explain remaining differences. These sensitivity calculations do not identify the software’s actual internal choices.

The reports’ fitting statistics also differ: their supplied-constants MSD is **802.179**, versus **747.774** under our explicitly stated calculation with the same rounded constants. Those values should not be ranked as forecast performance without establishing that they score the same errors in the same way.

[Comparison script](/examples/smoothing-selfstudy-v2/thermostat_software_turn11.py) · [Initialization results](/examples/smoothing-selfstudy-v2/thermostat_initialization_turn11.csv) · [Historical reference values](/examples/smoothing-selfstudy-v2/software_reference.csv) · [Reference provenance](/examples/smoothing-selfstudy-v2/software_reference.md) · [Attribution](/examples/smoothing-selfstudy-v2/turn11_DATA.md)

**Instructor**: The historical report markers and intervals are transcribed reference numbers, clearly distinguished from the newly executed formulas and initialization experiments. The original two software plots are recreated as a common-axis comparison. Their undocumented internal fitted paths cannot be claimed as newly reproduced. This is a useful stopping point for Lecture 19; stationary ARIMA is deferred.

**Instructor**: [Study guide and independent practice](/examples/smoothing-selfstudy-v2/STUDY-GUIDE.md) · [Data attribution](/examples/smoothing-selfstudy-v2/DATA.md) · [Reproduction order](/examples/smoothing-selfstudy-v2/reproduction-order.txt). These materials support independent interpretation and explanation; they do not announce exam coverage or a graded submission.
