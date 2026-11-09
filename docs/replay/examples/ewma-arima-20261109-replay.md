# Title: EWMA recap and introduction to ARIMA

**Instructor**: One class route: checkpoints 1–2 recap last week in about 12 minutes; checkpoints 3–8 develop ARIMA in about 55 minutes; leave time for questions. Checkpoint 9 is optional extension for another sitting. Pause before each answer and explain your reasoning.

**Instructor**: Learning goals. Update an EWMA by hand and explain its weights; choose between level, trend and seasonal smoothing; explain differencing and ACF/PACF; interpret an ARIMA model; follow a forecast-error calculation; check residual evidence and return forecasts to the original units. These are expected learning targets, not a claim that every student has already mastered last week’s material.

**Instructor**: Provenance. This replays a completed Codex CLI conversation recorded for this lesson. Human prompts are simulated instructor questions. Code and numerical analyses actually ran. Assistant responses retain their recorded wording, with only portable links changed. Instructor notes and pauses were added separately. Constructed examples are labeled. Playback displays saved results.

**Human**: Remind us of the cod-catch forecasting question and show the data. How does an exponentially weighted moving average learn from a new observation? Keep this a quick recap.

**Instructor**: Before revealing the answer: if the new catch is above our forecast, which direction should the level move? What controls how far it moves?

<!-- checkpoint: 1. EWMA recap: update the level -->
**Assistant**: The cod-catch question is: **Given the catches observed so far, how should we forecast next month’s catch?** We have 24 monthly observations, measured in tons. At each month, we can make a forecast, observe the new catch, and update our estimate of the current level.

![Monthly cod catches in tons over 24 months.](/examples/ewma-arima-20261109-v1/recap_cod_ewma_01.png)

The observations are:

| Months | Catches (tons), in order |
|---|---|
| 1–6 | 362, 381, 317, 297, 399, 402 |
| 7–12 | 375, 349, 386, 328, 389, 343 |
| 13–18 | 276, 334, 394, 334, 384, 314 |
| 19–24 | 344, 337, 345, 362, 314, 365 |

These historical teaching data come from Bowerman, O’Connell, and Koehler, *Forecasting, Time Series, and Regression*, fourth edition, Table 6.1, as documented in [DATA.md](/examples/ewma-arima-20261109-v1/DATA.md). Calendar dates are unspecified.

An **exponentially weighted moving average**, also called simple exponential smoothing, updates its level estimate using

\[
\ell_t=\alpha y_t+(1-\alpha)\ell_{t-1},
\qquad 0<\alpha\leq1.
\]

Here \(y_t\) is the new catch and \(\ell_{t-1}\) is the previous level estimate. The most useful way to read this equation is

\[
\boxed{\ell_t=\ell_{t-1}
+\alpha\underbrace{(y_t-\ell_{t-1})}_{\text{one-step forecast error}}.}
\]

**It learns by moving a fraction \(\alpha\) of the way toward the new observation.** A larger \(\alpha\) responds faster; a smaller \(\alpha\) smooths more strongly. Repeated updates give progressively smaller weights to older observations, which explains the word “exponentially.”

For a **constructed illustration**, choose \(\alpha=0.20\) and initialize \(\ell_1=y_1=362\). Applying the update through month 23 gives \(\ell_{23}=341.42\) tons. Month 24 then brings a catch of 365 tons:

\[
\begin{aligned}
\text{Forecast error}&=365-341.42=23.58,\\
\text{Adjustment}&=0.20(23.58)=4.72,\\
\ell_{24}&=341.42+4.72=346.14\text{ tons}.
\end{aligned}
\]

The higher catch pulls our level estimate upward, but we absorb only part of that surprise.

Under a level-only forecasting model, the **future month-25 point forecast is 346.14 tons**. The month-24 error above is a historical fitted error; month 25 has no observed catch here. The smoothing choice was illustrative, and this calculation does not supply a prediction interval.

[Executed script](/examples/ewma-arima-20261109-v1/recap_cod_ewma_01.py) · [Numerical results](/examples/ewma-arima-20261109-v1/recap_cod_ewma_01_results.json)

**Instructor**: Hand check: a previous level of341.42 tons, a catch of365 tons and alpha 0.20 give an adjustment of4.72 tons and a new level of346.14 tons. This initialization differs from the earlier full lesson; it is an explicit simple illustration.

**Human**: What should we now understand from last week? Show how the weights fade, remind us when Holt or Holt–Winters is needed, and explain how we would check a forecast.

**Instructor**: Before revealing the answer: explain when a flat forecast is inadequate. What would you need to know to call one forecasting method better?

<!-- checkpoint: 2. Last week: weights, trend, seasonality and checking -->
**Assistant**: From last week, we should understand **what smoothing learns, which patterns it can represent, and how to judge its forecasts.**

For EWMA, the updated level is
\[
\ell_t=\alpha y_t+(1-\alpha)\ell_{t-1}.
\]
Expanding the recursion shows how past observations enter:
\[
\ell_t=\alpha y_t+\alpha(1-\alpha)y_{t-1}
+\alpha(1-\alpha)^2y_{t-2}+\cdots.
\]
Each step back multiplies the weight by \(1-\alpha\).

![EWMA weights fade with the age of an observation.](/examples/ewma-arima-20261109-v1/recap_weights_checks_02.png)

With our **illustrative choice \(\alpha=0.20\)**, the newest observation receives weight \(0.20\); the next weights are \(0.16,\ 0.128,\ 0.1024,\ldots\). The newest five observations together receive about **67.2%** of the weight. Older observations still matter, but their influence fades. For a finite series initialized at its first observation, the remaining weight sits on that initial value.

The method should match the pattern we expect to persist:

| Pattern to forecast | Smoothing method | What it updates |
|---|---|---|
| Changing level, no persistent trend or seasonality | Simple exponential smoothing / EWMA | Level |
| Level plus persistent trend | Holt | Level and trend |
| Level, trend, and repeating seasonal pattern | Holt–Winters | Level, trend, and seasonality |

Holt lets forecasts continue an estimated trend. Holt–Winters also adjusts for seasonal position. **Additive seasonality** represents seasonal swings of roughly constant size; **multiplicative seasonality** represents swings that scale with the level and ordinarily requires positive observations. A calendar pattern needs evidence across repeated cycles.

To check forecasts, **recreate the information available when each forecast would have been made**:

1. Fit or initialize using earlier observations.
2. Forecast the next observation before seeing it.
3. Record \(e_t=y_t-\hat y_{t\mid t-1}\), then reveal the observation and update.
4. Repeat, comparing methods on identical dates and forecast horizons.

For a **retrospective cod-catch illustration**, initialize with months 1–18, keep \(\alpha=0.20\) fixed, and evaluate rolling one-month-ahead forecasts for months 19–24:

| Method | MAE (tons) | RMSE (tons) |
|---|---:|---:|
| EWMA | 14.73 | 18.85 |
| Forecast the previous month’s catch | 26.83 | 32.16 |

MAE measures average absolute error; RMSE penalizes large misses more strongly. EWMA wins this small comparison, but six historical errors provide limited evidence.

Also inspect errors over time: persistent bias, runs, or seasonal structure suggest missed information. Check prediction-interval coverage separately. Small fitted errors alone do not establish good future forecasts.

[Executed script](/examples/ewma-arima-20261109-v1/recap_weights_checks_02.py) · [Results](/examples/ewma-arima-20261109-v1/recap_weights_checks_02_results.json)

**Instructor**: Independent check: the newest five observations receive 1−0.8^5=0.67232 of the EWMA weight. Holt updates level and trend; Holt–Winters also updates seasonal states. The retrospective six-month comparison is small and does not establish future superiority.

**Human**: Now show us the paper-towel forecasting problem and its data. Why might we model the weekly changes instead of the sales levels?

**Instructor**: Before revealing the answer: what does a negative value in the supplied sales column mean? What would you look for in a plot of weekly changes?

<!-- checkpoint: 3. ARIMA begins: levels or changes? -->
**Assistant**: The paper-towel question is: **Using 120 weeks of observed sales, how should we forecast sales for week 121 and beyond?** Future observations are not supplied, so any forecasts beyond week 120 will be predictions we cannot yet compare with actual sales.

The data use a shifted scale. If \(y_t\) is the recorded value, then
\[
\text{actual sales}_t=100{,}000+10{,}000y_t
\quad\text{rolls}.
\]
For example, week 1 has \(y_1=15\), meaning **250,000 rolls**. Week 60 has \(y_{60}=-0.9127\), meaning **90,873 rolls**. A negative recorded value means sales below the baseline of 100,000 rolls.

![Paper-towel sales levels and their weekly changes.](/examples/ewma-arima-20261109-v1/paper_towels_changes_03.png)

These historical teaching data are from Bowerman, O’Connell, and Koehler, *Forecasting, Time Series, and Regression*, fourth edition, Table 9.1. See [DATA.md](/examples/ewma-arima-20261109-v1/DATA.md) for definitions and [the complete data](/examples/ewma-arima-20261109-v1/paper_towels.csv). Calendar dates are unspecified.

In the upper plot, the sales level moves through extended rises and falls. A single constant mean does not look like a comfortable description: the series spends long stretches at quite different levels. This motivates asking whether **changes have a more stable pattern than levels**.

Define the first difference by
\[
\boxed{w_t=\Delta y_t=y_t-y_{t-1}.}
\]
The lower plot shows these weekly changes. Differencing removes an accumulated level: it asks how much sales moved this week. For the last observation,
\[
w_{120}=15.6453-15.3410=0.3043,
\]
so sales increased by **3,043 rolls** from week 119 to week 120. Notice that the baseline of 100,000 rolls cancels when we subtract.

The changes fluctuate around a more stable-looking center. They can still have dependence: the sample correlation between consecutive changes is about **0.31**, compared with **0.97** between consecutive levels. These are descriptive estimates, not proof that differencing produces a stationary process. Stationarity means that properties such as the mean, variance, and dependence structure stay stable over time; we would need further checks.

This gives the route into **ARIMA(\(p,d,q\))**: model an appropriately differenced series using autoregressive and moving-average terms. One round of differencing corresponds to \(d=1\); we have not yet selected \(p\) or \(q\).

Finally, forecasts must return to the sales scale:
\[
\hat y_{121\mid120}=y_{120}+\hat w_{121\mid120}.
\]
For longer horizons, add the predicted changes cumulatively. **We model changes to simplify the pattern, then reconstruct the sales levels we need to forecast.**

[Executed script](/examples/ewma-arima-20261109-v1/paper_towels_changes_03.py) · [Results](/examples/ewma-arima-20261109-v1/paper_towels_changes_03_results.json)

**Instructor**: Stationarity concerns the process mean, variance and lag dependence. A stable-looking plot is evidence to investigate, not a proof. The baseline disappears when differencing; add it back when reporting sales in rolls.

**Human**: Show us the ACF and PACF for these levels and changes. How does removing the intervening lag help, and what model would you try first?

**Instructor**: Before revealing the answer: why can two observations two weeks apart be correlated even without a direct lag-two relationship? What should controlling for the middle week accomplish?

<!-- checkpoint: 4. ACF and PACF: identify a candidate -->
**Assistant**: **The weekly changes suggest trying ARIMA(\(0,1,1\)) first:** difference the sales once, then model the changes with an MA(1). This is a starting candidate, not a final selection.

![ACF and PACF of paper-towel sales levels and first differences. Shaded bands are approximate white-noise reference limits.](/examples/ewma-arima-20261109-v1/paper_towels_acf_pacf_04.png)

The **ACF at lag \(k\)** measures association between observations \(k\) weeks apart. The **PACF at lag \(k\)** measures their remaining linear association after controlling for the intervening lags.

For the **sales levels**, the ACF declines slowly: its first three values are approximately \(0.963,\ 0.907,\ 0.853\). Nearby weeks share a persistent level, and that association extends far into the past. This supports investigating differencing, although an ACF alone cannot establish that differencing is necessary.

For the **weekly changes**, the picture is much shorter:

| Lag | ACF of changes | PACF of changes |
|---|---:|---:|
| 1 | 0.307 | 0.307 |
| 2 | −0.065 | −0.175 |
| 3 | −0.072 | 0.006 |
| 4 | 0.105 | 0.133 |

There is a clear positive lag-1 ACF, followed by mostly small values. The shaded limits are approximately \(\pm0.18\), using \(1.96/\sqrt n\). They are rough, individual white-noise reference limits, not a model-selection rule; their interpretation is especially limited for potentially nonstationary levels.

**What does removing the intervening lag accomplish?** At lag 2, compare \(w_t\) with \(w_{t-2}\), where \(w_t=\Delta y_t\). Both can be associated with \(w_{t-1}\). To isolate what remains:

1. Regress \(w_t\) on \(w_{t-1}\) and keep its residual.
2. Regress \(w_{t-2}\) on \(w_{t-1}\) and keep its residual.
3. Correlate those residuals.

For these changes, the raw lag-2 correlation on those same rows is **−0.065**, while the partial correlation is **−0.179**, close to the plotted PACF estimate. Controlling for the middle observation exposes a more negative association. **Removing an intervening lag does not necessarily shrink a correlation.** It separates linear association through the middle lag from the association left over; it does not establish causation.

The identification guide is:

- **AR(\(p\))**: PACF cuts off after \(p\); ACF tails off.
- **MA(\(q\))**: ACF cuts off after \(q\); PACF tails off.

The changes look closer to the second pattern, motivating
\[
\boxed{\Delta y_t=\mu+\varepsilon_t+\theta\varepsilon_{t-1}.}
\]
This uses the **plus-sign MA convention**. Here \(\mu\) is the mean weekly change, and innovations are assumed to have zero mean, constant variance, and no serial correlation.

We would estimate the parameters, consider whether \(\mu\) is needed, check residual dependence, and compare chronological forecast errors with alternatives such as ARIMA(\(1,1,0\)).

[Executed script](/examples/ewma-arima-20261109-v1/paper_towels_acf_pacf_04.py) · [Results](/examples/ewma-arima-20261109-v1/paper_towels_acf_pacf_04_results.json)

**Instructor**: The lag-two partial correlation is the correlation of two residuals after removing the intervening lag. Different finite-sample estimators can differ slightly. Sample ACF/PACF patterns suggest candidates; they do not establish the true order.

**Human**: Before fitting, what is the difference between AR(1), MA(1), and the moving average we used last week? Show how each responds to a surprise.

**Instructor**: Before revealing the answer: does MA(1) mean averaging the last observed sales value? How long would one innovation remain in an AR(1) or MA(1) equation?

<!-- checkpoint: 5. AR, MA and EWMA: what carries a surprise? -->
**Assistant**: **AR(1) and MA(1) describe how a series is generated. Last week’s EWMA is a rule for updating a level estimate and forecasting.** The phrase “moving average” has different meanings here.

Let \(x_t\) denote a series centered around its mean. For paper towels, it could represent centered weekly changes. Let \(\varepsilon_t\) be an innovation: the new, unpredictable part, assumed to have zero mean, constant variance, and no serial correlation.

**AR(1): the previous observation carries the surprise forward.**
\[
x_t=\phi x_{t-1}+\varepsilon_t.
\]
With \(|\phi|<1\), its effect eventually fades. In a **constructed example**, start at zero, choose \(\phi=0.6\), and introduce an innovation of 10 with all later innovations zero:
\[
x_0=10,\qquad x_1=6,\qquad x_2=3.6,\qquad x_3=2.16,\ldots
\]
Each affected observation passes part of the effect to the next one. This produces a gradually declining ACF.

**MA(1): the previous innovation carries the surprise forward once.**
\[
x_t=\varepsilon_t+\theta\varepsilon_{t-1}.
\]
We use the **plus-sign MA convention**. With \(\theta=0.6\), the same isolated innovation gives
\[
x_0=10,\qquad x_1=6,\qquad x_2=0,\qquad x_3=0,\ldots
\]
The innovation appears directly today and through the lagged term next week. After that, it leaves the equation. The theoretical ACF is zero beyond lag 1. MA(1) averages **innovations**, which are unobserved and must be inferred when fitting.

![Constructed responses to a 10-unit surprise: AR(1), MA(1), and EWMA.](/examples/ewma-arima-20261109-v1/surprise_responses_05.png)

**EWMA: the surprise changes our estimate of the level.**
\[
\ell_t=\ell_{t-1}+\alpha(y_t-\ell_{t-1}).
\]
With an old level of zero, an observed value of 10, and \(\alpha=0.2\), the new level is
\[
\ell_0=0+0.2(10-0)=2.
\]
Our forecasts made immediately afterward are **2 at every future horizon**, under simple exponential smoothing’s level-only model.

There are two distinct paths in the EWMA panel:

- **Dashed:** if later observations equal their forecasts, there are no further surprises, so the updated level stays at 2.
- **Solid:** if later observations instead return to zero, the level declines as \(2,\ 1.6,\ 1.28,\ldots\). Those zeros create new negative forecast errors.

That distinction matters: **fading weights on past observations do not imply that today’s future forecasts fade toward zero.**

A fixed-window moving average is another averaging rule: it assigns equal weight to a fixed number of recent observations, then drops each observation when it leaves the window. Neither that rule nor EWMA is what “MA(1)” means in ARIMA.

[Executed script](/examples/ewma-arima-20261109-v1/surprise_responses_05.py) · [Results](/examples/ewma-arima-20261109-v1/surprise_responses_05_results.json)

**Instructor**: AR uses lagged observations; MA uses lagged innovations. EWMA updates a level estimate. In the constructed EWMA plot, later zeros are new negative surprises; they are not the forecasts issued at the original time.

**Human**: Fit the no-drift paper-towel model by minimizing squared forecast errors. Walk us through the first updates so we can check what the fitted MA coefficient means.

**Instructor**: Before revealing the answer: what is known when we forecast week 3? With the first error initialized to zero, compute the week-2 error from y1=15 and y2=14.4064.

<!-- checkpoint: 6. Fit the model and check the first errors -->
**Assistant**: The fitted **no-drift ARIMA(\(0,1,1\))** is
\[
\boxed{\Delta y_t=\varepsilon_t+0.3534\,\varepsilon_{t-1}.}
\]
We use the **plus-sign MA convention**. The coefficient is estimated from the data; it is not a known parameter.

Here \(y_t\) is paper-towel sales above 100,000 rolls, in units of 10,000 rolls, as defined in [DATA.md](/examples/ewma-arima-20261109-v1/DATA.md). No drift means we include no constant mean change.

To fit the model, condition on the observed first level \(y_1=15\) and **initialize \(e_1=0\)**. For each trial coefficient \(\theta\), calculate
\[
\begin{aligned}
\widehat{\Delta y}_{t\mid t-1}&=\theta e_{t-1},\\
\hat y_{t\mid t-1}&=y_{t-1}+\theta e_{t-1},\\
e_t&=y_t-\hat y_{t\mid t-1}
=\Delta y_t-\theta e_{t-1}.
\end{aligned}
\]
Then choose \(\theta\) to minimize
\[
S(\theta)=\sum_{t=2}^{120}e_t(\theta)^2.
\]
I searched within the invertible range, using bounds \(-0.999<\theta<0.999\). This is **conditional least squares**: the initialization is part of the calculation.

![Squared one-step forecast errors minimized over the MA coefficient.](/examples/ewma-arima-20261109-v1/paper_towels_fit_06.png)

The minimum occurs at \(\hat\theta=0.3533868\), with SSE **127.4788**. Setting \(\theta=0\), which forecasts the previous week’s level, gives SSE **143.8406**. This is an improvement in fit, not evidence yet of better future forecasts.

We can check the first updates using the fitted coefficient.

**Week 2:** With the initialized error equal to zero,
\[
\widehat{\Delta y}_{2\mid1}=0,\qquad
e_2=(14.4064-15)-0=-0.5936.
\]
The week-2 catch-up information is a negative forecast error.

**Week 3:** Carry part of that error into the next predicted change:
\[
\begin{aligned}
\widehat{\Delta y}_{3\mid2}
&=0.3533868(-0.5936)=-0.2097704,\\
\hat y_{3\mid2}&=14.4064-0.2097704=14.1966296,\\
e_3&=14.9383-14.1966296=0.7416704.
\end{aligned}
\]
Sales instead rose, so the new error is positive.

**Week 4:** That positive error now drives the adjustment:
\[
\begin{aligned}
\widehat{\Delta y}_{4\mid3}
&=0.3533868(0.7416704)=0.2620965,\\
\hat y_{4\mid3}&=14.9383+0.2620965=15.2003965,\\
e_4&=16.0374-15.2003965=0.8370035.
\end{aligned}
\]

Thus, **a positive surprise of one recorded unit adds about 0.3534 units to next week’s predicted change**. The coefficient multiplies the previous **forecast error**, not the previous observed change.

These are historical fitted errors, reconstructed using a coefficient estimated from all 120 weeks. We still need residual checks and chronological forecast comparisons before accepting the model.

[Executed script](/examples/ewma-arima-20261109-v1/paper_towels_fit_06.py) · [Results](/examples/ewma-arima-20261109-v1/paper_towels_fit_06_results.json) · [All fitted errors](/examples/ewma-arima-20261109-v1/paper_towels_fit_06_errors.csv)

**Instructor**: The plus-convention MA estimate is about 0.3534. Your legacy notes use a minus sign and report about negative 0.3534. Both describe the same model. The final fitted value is about 14.9560; the legacy line repeating 14.1966 was a typographical error. Initialization is explicit here, so compare historical software SSE only after checking its conventions.

**Human**: How would we decide whether this model is adequate? Check the residuals and a time-ordered forecast comparison, and help us challenge an AI that says a large Ljung–Box p-value proves it is the best model.

**Instructor**: Before revealing the answer: consider this deliberately constructed AI claim: “The Ljung–Box p-value exceeds 0.05, so this is the best model and its errors are independent.” What evidence supports or contradicts it?

<!-- checkpoint: 7. Check the model and audit an AI claim -->
**Assistant**: **The no-drift ARIMA(0,1,1) looks reasonable as a working model.** Its residuals show little remaining serial correlation, and it performs slightly better than two simple alternatives in a time-ordered comparison. Neither result proves that it is the best model.

First, check the fitted one-step errors,
\[
e_t=y_t-\hat y_{t\mid t-1}.
\]
An adequate conditional-mean model should leave errors centered near zero, with no useful pattern remaining in their history. We also inspect changing spread, unusual observations, and distributional assumptions when constructing prediction intervals.

![Fitted paper-towel errors and their residual ACF.](/examples/ewma-arima-20261109-v1/paper_towels_diagnostics_07.png)

The residual mean is **0.0055 recorded units**, close to zero. The lag-1 residual ACF is **0.006**, so the strong short-term association in the changes has largely disappeared. The plot shows some variation in error size; it does not establish constant variance or normality.

The Ljung–Box statistic combines residual autocorrelations through lag \(m\):
\[
Q=n(n+2)\sum_{k=1}^{m}\frac{\hat r_k^2}{n-k}.
\]
Using the usual approximate reference distribution, with one degree of freedom deducted for the fitted MA coefficient, gives:

| Maximum lag \(m\) | \(Q\) | Reference df | p-value |
|---|---:|---:|---:|
| 6 | 4.103 | 5 | 0.535 |
| 12 | 10.400 | 11 | 0.495 |
| 18 | 17.570 | 17 | 0.416 |

These checks **do not reject the hypothesis of zero residual autocorrelations through the selected lags**.

Next, check actual forecasting behavior. I fit through week 90, predicted week 91, revealed its observation, refitted, and continued through week 120. Every method used only observations available before its forecast. All three models excluded drift.

| Method | MAE | RMSE |
|---|---:|---:|
| Random walk: forecast the last observation | 0.7798 | 0.9374 |
| ARIMA(0,1,1) | **0.7541** | **0.8969** |
| ARIMA(1,1,0) | 0.7881 | 0.9406 |

Units are **10,000 rolls**, following [DATA.md](/examples/ewma-arima-20261109-v1/DATA.md). The MA model has the lowest errors here, but its advantage is modest. There are only 30 forecast errors, and this is a retrospective illustration using previously inspected historical data, not an untouched test. Performance at longer horizons could differ.

We should challenge the AI’s claim this way:

> “A large Ljung–Box p-value means this test did not detect residual autocorrelation at the chosen lags. It does not prove that the model is correct, or compare its forecast accuracy with other models.”

A p-value is **not the probability that a model is best**. The test can miss dependence, addresses only selected autocorrelations, and does not establish independence, normality, stable variance, or accurate prediction intervals. Adequacy requires several checks; choosing among adequate models also requires forecast comparisons matched to the intended horizon.

[Executed script](/examples/ewma-arima-20261109-v1/paper_towels_diagnostics_07.py) · [Diagnostic results](/examples/ewma-arima-20261109-v1/paper_towels_diagnostics_07_results.json) · [Time-ordered forecasts](/examples/ewma-arima-20261109-v1/paper_towels_diagnostics_07_forecasts.csv)

**Instructor**: The residual test addresses selected autocorrelations. The time-ordered comparison adds forecast evidence, but only for these 30 historical one-step predictions. Explain the size of the advantage and name one follow-up check before recommending a model.

**Human**: Now forecast the next few weeks in actual rolls. Why can the point forecasts stay flat while prediction intervals get wider?

**Instructor**: Before revealing the answer: future innovations have mean zero. Does that imply zero forecast uncertainty? Predict the shape of the point forecasts and interval widths before seeing the plot.

<!-- checkpoint: 8. Forecast in rolls and explain uncertainty -->
**Assistant**: The fitted model forecasts **about 258,889 rolls per week for weeks 121–126**, with increasingly wide prediction intervals.

| Week | Point forecast (rolls) | 95% prediction interval |
|---|---:|---:|
| 121 | 258,889 | 238,517–279,260 |
| 122 | 258,889 | 224,608–293,169 |
| 123 | 258,889 | 214,897–302,881 |
| 124 | 258,889 | 206,971–310,806 |
| 125 | 258,889 | 200,105–317,673 |
| 126 | 258,889 | 193,960–323,817 |

![Paper-towel forecasts in actual rolls, with widening 95% prediction intervals.](/examples/ewma-arima-20261109-v1/paper_towels_forecast_08.png)

Recall the fitted no-drift model, using the **plus-sign MA convention**:
\[
\Delta y_t=\varepsilon_t+\theta\varepsilon_{t-1},
\qquad \hat\theta=0.3534.
\]
The last observed value is \(y_{120}=15.6453\), and the last fitted innovation is \(e_{120}=0.6892642\). Therefore,
\[
\widehat{\Delta y}_{121\mid120}
=0.3533868(0.6892642)=0.2435768.
\]
Our next level forecast is
\[
\hat y_{121\mid120}=15.6453+0.2435768=15.8888768.
\]
Converting using the definition in [DATA.md](/examples/ewma-arima-20261109-v1/DATA.md),
\[
100{,}000+10{,}000(15.8888768)
\approx258{,}889\text{ rolls}.
\]

**Why do the point forecasts stay flat?** The latest innovation contributes to next week’s expected change. After that, both innovations in each future change are unknown and have conditional mean zero. Thus,
\[
\widehat{\Delta y}_{122\mid120}
=\widehat{\Delta y}_{123\mid120}=\cdots=0.
\]
Adding zero expected changes leaves the level forecast unchanged. This does **not** predict that actual sales will be identical each week.

**Why do the intervals widen?** Zero expected innovations still have nonzero variance. Future surprises accumulate in the sales level. Treating the fitted parameters and last innovation as known, the forecast errors begin as
\[
\begin{aligned}
y_{121}-\hat y_{121\mid120}&=\varepsilon_{121},\\
y_{122}-\hat y_{122\mid120}
&=(1+\theta)\varepsilon_{121}+\varepsilon_{122}.
\end{aligned}
\]
More generally, under independent innovations with common variance \(\sigma^2\),
\[
\boxed{
\operatorname{Var}(y_{120+h}-\hat y_{120+h\mid120})
=\sigma^2\left[1+(h-1)(1+\theta)^2\right].
}
\]
The mean stays flat while the variance grows with the forecast horizon.

For these intervals, I estimated \(\sigma^2\) as SSE divided by \(119-1\), giving **1.0803**, and used a Gaussian interval: forecast \(\pm1.96\) forecast standard errors. These are **plug-in, pointwise 95% prediction intervals**. They assume independent Gaussian innovations with constant variance and omit parameter and model uncertainty. They do not give a 95% guarantee that all six future observations fall inside their respective intervals.

[Executed script](/examples/ewma-arima-20261109-v1/paper_towels_forecast_08.py) · [Results](/examples/ewma-arima-20261109-v1/paper_towels_forecast_08_results.json) · [Forecast table](/examples/ewma-arima-20261109-v1/paper_towels_forecast_08.csv)

**Instructor**: Convert levels with 100000 + 10000*y, and multiply forecast errors or interval widths by 10000. The level baseline is added only once. An inventory decision also needs the costs of shortage and excess stock.

**Human**: How does this extend to general ARIMA(p,d,q), and how does our simple model connect back to EWMA?

**Instructor**: Before revealing the answer: which part of ARIMA(p,d,q) describes differencing? What would you need to show before calling the fitted paper-towel model the same as ordinary EWMA?

<!-- checkpoint: 9. Optional: general ARIMA and the EWMA connection -->
**Assistant**: **ARIMA(\(p,d,q\)) models a differenced series using past values and past innovations.** Our paper-towel model is the special case \(p=0,\ d=1,\ q=1\).

Define
\[
w_t=\Delta^d y_t.
\]
Then a general nonseasonal ARIMA model is
\[
\boxed{
w_t=c+\sum_{i=1}^{p}\phi_iw_{t-i}
+\varepsilon_t+\sum_{j=1}^{q}\theta_j\varepsilon_{t-j}.
}
\]
We retain the **plus-sign MA convention**.

| Part | Meaning | Forecasting role |
|---|---|---|
| \(p\): autoregressive order | Number of lagged \(w\) values | Carries information from past values |
| \(d\): differencing order | Number of successive differences | Removes accumulated changes in level |
| \(q\): moving-average order | Number of lagged innovations | Carries information from past surprises |

The constant \(c\) is not generally the mean of \(w_t\). When \(w_t\) is stationary,
\[
E(w_t)=\frac{c}{1-\sum_i\phi_i}.
\]
Our no-drift model sets \(c=0\).

To forecast, estimate the coefficients and past innovations, set future innovations to their conditional expectation of zero, and recursively predict \(w_t\). Then undo the differencing. With \(d=1\), add predicted changes to the latest observed level. Future innovations still contribute uncertainty, even though their expected values are zero.

The connection to EWMA comes directly from our model:
\[
y_t=y_{t-1}+\varepsilon_t+\theta\varepsilon_{t-1}.
\]
Let \(f_t\) be the one-step forecast of \(y_t\), and define \(e_t=y_t-f_t\). The next forecast is
\[
f_{t+1}=y_t+\theta e_t.
\]
Substitute \(y_t=f_t+e_t\):
\[
\boxed{f_{t+1}=f_t+(1+\theta)e_t.}
\]
Compare this with the EWMA forecast update:
\[
f_{t+1}=f_t+\alpha e_t.
\]
Thus, with matching initialization,
\[
\boxed{\alpha=1+\theta}
\]
under our plus-sign convention.

**There is an important qualification for our fitted model.** Last week’s illustrative EWMA used \(\alpha=0.20\), corresponding to \(\theta=-0.80\). It moves only partway toward a new observation.

Our paper-towel estimate is instead \(\hat\theta=0.3534\), giving an equivalent update gain of **1.3534**. That exceeds the usual EWMA range \(0<\alpha\leq1\), so this fitted model does not represent ordinary exponentially weighted averaging with positive weights.

We can check the last update:
\[
\begin{aligned}
f_{120}&=14.956036,\\
e_{120}&=15.6453-14.956036=0.689264,\\
f_{121}&=14.956036+1.353387(0.689264)\\
&=15.888877.
\end{aligned}
\]
The forecast moves **past** the latest observation because the positive MA coefficient predicts a further positive change following a positive surprise.

Simple exponential smoothing therefore corresponds to a particular no-drift ARIMA(0,1,1) forecasting rule. Our fitted ARIMA model has the same algebraic update form, but its estimated response lies outside ordinary EWMA averaging.

[Executed calculation](/examples/ewma-arima-20261109-v1/arima_ewma_connection_09.py) · [Results](/examples/ewma-arima-20261109-v1/arima_ewma_connection_09_results.json)

**Instructor**: Optional depth. Keep the MA sign convention visible. The connection to simple smoothing has parameter restrictions; the fitted paper-towel model need not fall inside the usual convex-weight EWMA range. Seasonal ARIMA is reserved for another lesson.

**Instructor**: Further reading: [ARIMA models, Forecasting: Principles and Practice](https://otexts.com/fpp3/arima.html). Practice explaining evidence and checking AI claims, as in the current practice materials. This lesson makes no promise about future exam coverage.
