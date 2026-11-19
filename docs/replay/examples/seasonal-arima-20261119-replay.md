# Title: Seasonal ARIMA: the hotel-room story

**Instructor**: Continue from nonseasonal ARIMA with the hotel-room example. Follow the transformations before introducing seasonal model terms, then identify, fit, check and forecast. Use checkpoints 1–10 and 12 for a class meeting; checkpoint 11 is an optional retrospective evaluation.

**Instructor**: Learning goals. Interpret the occupancy units; distinguish regular and seasonal differences; explain seasonal MA and AR; use short and seasonal lags to propose a model; interpret its constant; check residual evidence; and reconstruct forecasts and uncertainty in rooms per day. Make your own judgment at each discussion pause.

**Instructor**: Provenance. This replays a completed Codex CLI conversation. Human prompts are simulated instructor questions. Code and numerical analyses actually ran. Assistant responses retain their recorded wording, with only asset links made portable. Instructor notes were added separately, and constructed examples are labeled. Playback displays saved results. The numerical example comes from Bowerman, O’Connell and Koehler, Forecasting, Time Series, and Regression, fourth edition, Chapter 11, Table 11.1 and Examples 11.1–11.3. Newly estimated results are not historical software output.

**Human**: Remind us of the hotel-room forecasting question and show us the data. What new pattern do we need to handle?

**Instructor**: Before revealing: What is the response, and why might last August help predict next August?

<!-- checkpoint: 1. The hotel question -->
**Assistant**: Traveler’s Rest operates **four hotels**. The forecasting question is: **How many rooms will be occupied on an average day in each future month?** These forecasts support staffing, supplies, and budgets.

We have 168 monthly observations covering 14 consecutive years, beginning in January. Each value is the **monthly average number of occupied rooms per day across the four hotels**. It is neither an occupancy percentage nor the total number of room nights in that month. Calendar years and total hotel capacity are not supplied.

![Hotel-room data over time and aligned by month within each year](/examples/seasonal-arima-20261119-v1/turn01_hotel_data.png)

The upper panel shows the observations in time order. Two features stand out. First, the level rises over time: average daily occupancy is about 570 rooms in the first year and 875 in the last year. Second, peaks and troughs recur regularly within that upward movement.

The lower panel makes that repetition easier to see by aligning the 14 years by calendar month. Occupancy generally climbs into summer, peaks in July or August, then falls in September. In fact, the largest observation within **every observed year** occurs in July or August. February and November tend to be relatively quiet, with a rebound in December. The yearly curves differ, so the repetition is not exact.

The new pattern is **seasonality**: a systematic tendency for similar behavior to recur at a fixed position within a repeating calendar cycle. For these monthly data, the seasonal period is **12 months**.

This matters because an August observation may resemble previous August observations in ways that its immediately preceding months cannot fully explain. Our nonseasonal ARIMA tools already address changing levels and dependence across nearby observations. We now need to account for the recurring annual pattern as well. That is an observation about the data, not yet a choice of model.

[Data](/examples/seasonal-arima-20261119-v1/hotel_monthly.csv) · [Analysis script](/examples/seasonal-arima-20261119-v1/turn01_show_data.py) · [Numerical results](/examples/seasonal-arima-20261119-v1/turn01_data_summary.json)

**Instructor**: Each observation is a monthly average of occupied rooms per day across four hotels. The seasonal period is 12 months. Keep those units when interpreting forecasts.

**Human**: The seasonal swings seem to grow with the level. Would taking fourth roots help? Show what changes before we difference anything.

**Instructor**: Before revealing: Would a scale transformation remove the recurring peak, or change its size? Explain.

<!-- checkpoint: 2. Fourth roots -->
**Assistant**: **Yes, taking fourth roots helps make the seasonal swings more uniform, although it does not make them perfectly constant.** Let \(y_t\) denote the monthly average number of occupied rooms per day across the four hotels. Define

\[
z_t=y_t^{1/4}.
\]

This transformation acts on each observation separately. We have not subtracted any observations or taken differences.

![Original hotel-room series and its fourth root, with individual years aligned by month](/examples/seasonal-arima-20261119-v1/turn02_fourth_root.png)

The upper row shows the original data; the lower row shows their fourth roots. The left panels preserve time order, while the right panels align the years by calendar month.

A fourth root compresses larger values more strongly. Its slope is
\[
\frac{d}{dy}y^{1/4}=\frac{1}{4y^{3/4}},
\]
which decreases as \(y\) increases. Consequently, a fixed increase in occupied rooms produces a smaller change on the transformed scale when occupancy is already high. This is useful when the seasonal swings grow with the level.

For a concrete comparison, use each year’s maximum minus its minimum as a simple measure of its swing:

| Scale | First year | Last year | Last / first |
|---|---:|---:|---:|
| Original rooms per day | 248 | 392 | 1.58 |
| Fourth-root scale | 0.514 | 0.589 | 1.15 |

The ratio within each scale is the useful comparison, since the two scales have different units. The later year’s swing is substantially larger on the original scale and only moderately larger after transformation. These ranges are descriptive summaries: they include irregular variation and changes within the year, so they do not isolate a pure seasonal component.

**What remains?** Summer peaks still occur in the same months because fourth roots preserve the ordering of observations. The upward movement also remains. Thus, the transformation helps with the changing **size** of the fluctuations; it does not remove either the annual pattern or the changing level. Approximate stabilization is a reasonable working judgment here, not an established claim of constant variance or stationarity.

[Script](/examples/seasonal-arima-20261119-v1/turn02_fourth_root.py) · [Numerical results](/examples/seasonal-arima-20261119-v1/turn02_fourth_root_results.json)

**Instructor**: This transformation changes the scale. The calendar order remains. The next prompt names the transformed series $y_t^*$ to match the notation used throughout the rest of the lesson.

**Human**: Let’s call the fourth-root series y*. What happens if we difference it one month at a time? Show the ACF before and after.

**Instructor**: Before revealing: Would subtracting July from August remove a pattern that repeats every summer? Predict the ACF at lag 12.

<!-- checkpoint: 3. Regular differences -->
**Assistant**: Write the fourth-root series as \(y_t^*\). Taking **one-month first differences** gives

\[
w_t=\Delta y_t^*=y_t^*-y_{t-1}^*,
\qquad t=2,\ldots,168.
\]

Each value now measures the change from the previous month on the fourth-root scale. We lose the first observation, leaving 167 differences.

![Fourth-root hotel series and its one-month differences, with ACFs through lag 48](/examples/seasonal-arima-20261119-v1/turn03_first_difference_acf.png)

Before differencing, the series has an upward movement together with its annual oscillation. After differencing, the upward movement is much less apparent: the changes fluctuate around a roughly stable level. However, the repeated annual pattern remains clearly visible. Increases and decreases tend to occur at similar times each year.

The ACFs show the distinction particularly well. Recall that the ACF at lag \(k\) measures the association between observations \(k\) months apart.

| Lag (months) | ACF of \(y_t^*\) | ACF of \(\Delta y_t^*\) |
|---|---:|---:|
| 1 | 0.794 | −0.002 |
| 12 | 0.856 | 0.905 |
| 24 | 0.717 | 0.825 |
| 36 | 0.589 | 0.758 |
| 48 | 0.465 | 0.689 |

The lag-1 correlation nearly disappears, but the correlations at annual intervals remain very large. For example, a particular month’s change strongly resembles the change occurring in the same month a year earlier. Differencing August against July still leaves a recurring summer change; differencing September against August still leaves a recurring decline.

Thus, **one-month differencing reduces the changing level but leaves substantial seasonal structure**. A small lag-1 ACF does not mean that the differenced series is white noise. We must inspect longer lags, especially those corresponding to the annual cycle.

For both panels, the sample ACF subtracts that series’ own sample mean and uses the lag-zero sum of squares as its denominator. The shaded bands are approximate pointwise white-noise references, \(\pm1.96/\sqrt n\); they are not a test establishing stationarity.

[Script](/examples/seasonal-arima-20261119-v1/turn03_first_difference.py) · [Numerical results](/examples/seasonal-arima-20261119-v1/turn03_first_difference_results.json)

**Instructor**: A near-zero lag-1 ACF can coexist with strong annual dependence. Inspect seasonal lags as well as adjacent observations.

**Human**: What if we compare each month with the same month a year earlier? Show that seasonal difference and its ACF.

**Instructor**: Before revealing: Which month should be subtracted from month 13? How many usable differences will remain?

<!-- checkpoint: 4. Seasonal differences -->
**Assistant**: Comparing each month with the same month a year earlier gives the **seasonal difference**

\[
v_t=\Delta_{12}y_t^*=y_t^*-y_{t-12}^*,
\qquad t=13,\ldots,168.
\]

Here, \(\Delta_{12}\) means “subtract the observation 12 months earlier.” We apply it directly to the fourth-root series \(y_t^*\). This leaves **156 observations**, since the first year supplies the comparison values.

For example, January of year 2 is compared with January of year 1, and August of year 2 with August of year 1.

![Seasonal differences of fourth-root hotel occupancy and their ACF](/examples/seasonal-arima-20261119-v1/turn04_seasonal_difference_acf.png)

The seasonal difference largely removes the recurring summer peaks and autumn declines. If two successive years had exactly the same seasonal pattern and level, their corresponding months would subtract to zero. Here, occupancy generally grows, so the differences have a positive sample mean, about **0.0425 on the fourth-root scale**. This measures average year-over-year growth on that scale, not growth in rooms per day.

The ACF changes substantially:

| Lag (months) | ACF of one-month differences | ACF of seasonal differences |
|---|---:|---:|
| 1 | −0.002 | 0.174 |
| 12 | 0.905 | −0.348 |
| 24 | 0.825 | −0.063 |
| 36 | 0.758 | 0.049 |
| 48 | 0.689 | −0.031 |

The large positive correlations persisting across successive years have disappeared. **Comparing like calendar months has addressed much of the annual repetition that one-month differencing left behind.**

The result still has structure: the lag-12 correlation is negative, and lag 3 is also noticeably negative, about −0.260. A negative lag-12 correlation means that an above-average year-over-year increase tends to be followed, a year later, by a below-average increase. It does not necessarily mean occupancy itself falls.

This is evidence that seasonal differencing is useful, but it does not establish that the result is white noise or fully stationary. The ACF uses the same mean-subtracted, fixed-denominator convention as before; shading gives approximate pointwise white-noise reference bands.

[Script](/examples/seasonal-arima-20261119-v1/turn04_seasonal_difference.py) · [Numerical results](/examples/seasonal-arima-20261119-v1/turn04_seasonal_difference_results.json)

**Instructor**: A seasonal difference compares the same calendar month in adjacent years. A positive mean on the fourth-root scale describes average year-over-year growth on that scale.

**Human**: Would taking both regular and seasonal differences improve this? Show the combined calculation and compare the evidence.

**Instructor**: Before revealing: Is more differencing automatically better? Expand the combined difference and identify its four observations.

<!-- checkpoint: 5. Both differences -->
**Assistant**: **Taking both differences centers the series near zero, but the evidence does not show a clear improvement over seasonal differencing alone.**

Starting with \(v_t=y_t^*-y_{t-12}^*\), take its one-month difference:

\[
\begin{aligned}
w_t&=v_t-v_{t-1}\\
&=(y_t^*-y_{t-12}^*)-(y_{t-1}^*-y_{t-13}^*)\\
&=\boxed{y_t^*-y_{t-1}^*-y_{t-12}^*+y_{t-13}^*.}
\end{aligned}
\]

This is \(\Delta\Delta_{12}y_t^*\). Applying the differences in the opposite order gives the same result. The first available value is at \(t=14\), leaving 155 observations.

For example,
\[
w_{14}=4.70248-4.77070-4.70008+4.73107
       \approx-0.03723.
\]
It says the January-to-February change in year 2 was about 0.0372 lower than that change in year 1, on the fourth-root scale.

![Seasonal differences compared with combined regular and seasonal differences, with ACFs](/examples/seasonal-arima-20261119-v1/turn05_combined_differences.png)

The time plots use the same vertical scale. Both lack the original large annual oscillation. The combined differences fluctuate near zero, but they also show more month-to-month movement.

| Evidence | Seasonal only | Both differences |
|---|---:|---:|
| Sample mean | 0.0425 | 0.0004 |
| Sample standard deviation | 0.0299 | 0.0382 |
| ACF at lag 1 | 0.174 | −0.404 |
| ACF at lag 3 | −0.260 | −0.238 |
| ACF at lag 12 | −0.348 | −0.337 |
| ACF at lag 24 | −0.063 | −0.131 |

The positive mean after seasonal differencing is **not by itself a reason to difference again**. A stationary series can have a nonzero constant mean. Its first and last 36 observations average 0.0497 and 0.0397, respectively, suggesting some change but not establishing a continuing trend.

The extra difference barely changes the lag-12 correlation and introduces a substantial negative lag-1 correlation. That is a warning of possible **overdifferencing**, meaning unnecessary differencing may have introduced dependence. It is not proof.

For these data, seasonal differencing alone remains a plausible starting point. Both versions retain dependence, so neither is white noise. ACF conventions and approximate reference bands match the previous comparison.

[Script](/examples/seasonal-arima-20261119-v1/turn05_combined_differences.py) · [Numerical results](/examples/seasonal-arima-20261119-v1/turn05_combined_results.json)

**Instructor**: The combined difference is $y_t^*-y_{t-1}^*-y_{t-12}^*+y_{t-13}^*$. It starts at month 14 and leaves 155 values. Choose transformations from the evidence; zero mean alone is not the goal.

**Human**: How do seasonal moving-average and autoregressive terms work? Show what one surprise can do a year later in each case.

**Instructor**: Before revealing: How would a seasonal MA term and a seasonal AR term carry one surprise into later years?

<!-- checkpoint: 6. Seasonal MA and AR -->
**Assistant**: Seasonal terms describe **dependence that remains after transformation and any chosen differencing**. Let \(x_t\) be that series after subtracting its mean, and let \(\varepsilon_t\) be a new, unpredictable surprise with mean zero and constant variance, uncorrelated across months.

Start with a **seasonal moving-average term**:

\[
\boxed{x_t=\varepsilon_t+\Theta\varepsilon_{t-12}.}
\]

This seasonal MA(1) uses the current surprise and the surprise **one seasonal period earlier**. The “1” counts years of memory here, not months.

Suppose \(\Theta=0.7\), and an unexpected event produces a \(+1\) surprise in July. It contributes \(+1\) immediately and \(+0.7\) the following July. Two Julys later, its contribution is zero: the equation remembers shocks only at lags 0 and 12. This uses a **plus-sign convention**; textbooks using a minus sign would reverse the coefficient’s sign for the same mechanism.

Now consider a **seasonal autoregressive term**:

\[
\boxed{x_t=\Phi x_{t-12}+\varepsilon_t.}
\]

This seasonal AR(1) carries forward the **series value** from the same month last year. With \(\Phi=0.7\), the original \(+1\) surprise raises this July’s value by 1. Next July inherits \(0.7\); the July after that inherits \(0.7(0.7)=0.49\). The effect keeps propagating.

![Constructed responses to one surprise under seasonal moving-average and autoregressive mechanisms](/examples/seasonal-arima-20261119-v1/turn06_seasonal_mechanisms.png)

These are **constructed illustrations**, with all earlier deviations and all other surprises set to zero. They are not hotel-data fits, and the unit is an arbitrary unit of \(x_t\).

| Months after surprise | Seasonal MA effect | Seasonal AR effect |
|---|---:|---:|
| 0 | 1 | 1 |
| 12 | 0.7 | 0.7 |
| 24 | 0 | 0.49 |
| 36 | 0 | 0.343 |

Both mechanisms can produce an echo one year later. The distinction appears afterward: **MA remembers a past surprise for a limited time; AR repeatedly transmits its effect through past values.** For this pure seasonal AR mechanism, \(|\Phi|<1\) makes the echoes decay. Negative coefficients can produce negative echoes, with alternating signs across years in the AR case.

[Script](/examples/seasonal-arima-20261119-v1/turn06_seasonal_mechanisms.py) · [Numerical results](/examples/seasonal-arima-20261119-v1/turn06_seasonal_mechanisms_results.json)

**Instructor**: Keep seasonal differencing separate from seasonal dependence. A seasonal difference changes the working series. Seasonal AR or MA terms describe dependence within that series. Check the displayed MA sign convention before interpreting coefficients. At higher orders, seasonal MA(Q) uses shocks at 12, 24, up to 12Q months; seasonal AR(P) uses working-series values at 12, 24, up to 12P months.

**Human**: Return to the hotel’s seasonal differences. What do the ACF and PACF through lag 24 suggest at the short lags and the seasonal lags?

**Instructor**: Before revealing: Inspect short lags first, then lags 12 and 24. Which features support a tentative model, and which are ambiguous?

<!-- checkpoint: 7. Tentative identification -->
**Assistant**: For the hotel, return to **seasonal differences only**:

\[
v_t=y_t^*-y_{t-12}^*,\qquad n=156.
\]

The ACF measures correlation across a given lag. The **PACF** measures the remaining linear association after accounting for the intervening lags.

![ACF and PACF of the hotel’s seasonal differences through lag 24](/examples/seasonal-arima-20261119-v1/turn07_seasonal_acf_pacf.png)

The shaded reference bands are approximately \(\pm0.157\). Selected values help distinguish the short-lag behavior from the annual behavior:

| Lag | ACF | PACF |
|---|---:|---:|
| 1 | 0.174 | 0.174 |
| 2 | 0.014 | −0.017 |
| 3 | −0.260 | −0.268 |
| 4 | −0.169 | −0.086 |
| 5 | −0.231 | −0.199 |
| 12 | −0.348 | −0.349 |
| 24 | −0.063 | −0.127 |

**At the short lags**, lag 1 is modestly positive and just outside the reference band, while lag 2 is near zero. But lags 3 and 5 are noticeably negative in both plots, and the ACF also has a negative value at lag 4. This is not a clean textbook signature of a simple nonseasonal AR(1) or MA(1). There is short-term dependence to investigate, but selecting an order from lag 1 alone would overlook substantial structure.

**At the seasonal lags**, the negative ACF at 12 followed by a small ACF at 24 is compatible with a **seasonal MA(1)** mechanism: its ideal pure-process ACF has a spike at 12 and is zero at later seasonal multiples. The PACF remains negative at 24, although smaller and inside the reference band, which is also compatible with that interpretation.

A pure seasonal AR(1) instead has an ideal PACF cutoff after lag 12 and geometrically decaying ACF values at seasonal multiples. These two observed seasonal lags cannot sharply distinguish the mechanisms, especially with additional short-term dependence.

The evidence therefore suggests **a seasonal MA term as a candidate, with unresolved short-lag structure**. It does not establish a final model. The isolated PACF spike at lag 19 also deserves restraint: examining many lags creates opportunities for chance exceedances.

The ACF uses the previous fixed-denominator convention; the PACF uses unadjusted Yule–Walker estimation.

[Script](/examples/seasonal-arima-20261119-v1/turn07_seasonal_acf_pacf.py) · [Numerical results](/examples/seasonal-arima-20261119-v1/turn07_seasonal_acf_pacf_results.json)

**Instructor**: The following prompt tries the legacy subset-AR candidate at lags 1, 3 and 5 with seasonal MA(1). The sample patterns motivate investigation, but they do not uniquely identify those terms. Fitting and diagnostic checks follow.

**Human**: Let’s try AR terms at lags 1, 3 and 5 with a seasonal MA(1) term. Fit that candidate, and explain whether it needs a constant.

**Instructor**: Before revealing: Is the equation intercept the same as the mean of the working series? What happens when past values also enter the equation?

<!-- checkpoint: 8. Fit and constant -->
**Assistant**: **This candidate strongly favors including a constant.** I fitted it to the 156 seasonal differences
\[
v_t=y_t^*-y_{t-12}^*,
\]
with no additional one-month difference.

Using the plus-sign convention for the seasonal MA term, the model is
\[
v_t=c+\phi_1v_{t-1}+\phi_3v_{t-3}+\phi_5v_{t-5}
+\varepsilon_t+\Theta\varepsilon_{t-12}.
\]

The fitted equation is
\[
\boxed{
v_t=0.04811+0.2428v_{t-1}-0.2187v_{t-3}
-0.1587v_{t-5}
+\varepsilon_t-0.5149\varepsilon_{t-12}.
}
\]

| Parameter | Estimate | Standard error |
|---|---:|---:|
| Constant \(c\) | 0.04811 | 0.00669 |
| AR lag 1 | 0.2428 | 0.0857 |
| AR lag 3 | −0.2187 | 0.0862 |
| AR lag 5 | −0.1587 | 0.0861 |
| Seasonal MA lag 12 | −0.5149 | 0.0908 |

The constant is **not the mean of \(v_t\)**, because the AR terms also contribute to that mean. Taking expectations gives
\[
\mu=c+(\phi_1+\phi_3+\phi_5)\mu,
\]
so
\[
\hat\mu=\frac{\hat c}{1-\hat\phi_1-\hat\phi_3-\hat\phi_5}
=0.04240.
\]
This closely matches the observed mean seasonal difference, 0.04252. It represents positive average year-over-year growth on the fourth-root scale. Seasonal differencing can remove recurring calendar effects while leaving this nonzero mean.

The comparison supports retaining it:

| Candidate | Log likelihood | AIC | BIC |
|---|---:|---:|---:|
| With constant | 353.72 | −695.43 | −677.13 |
| Without constant | 321.67 | −633.33 | −618.08 |

Lower AIC and BIC favor the constant by large margins. Without it, the AR coefficients sum to about 0.982, suggesting the model compensates for the omitted positive mean through very persistent dynamics.

Both fits converged without warnings using Gaussian state-space maximum likelihood, with stationary initialization and stationarity and invertibility enforced. These comparisons use identical observations and likelihood conventions. The lag-5 coefficient remains uncertain, and residual checks are still needed before accepting the candidate.

[Saved fitting script](/examples/seasonal-arima-20261119-v1/turn08_fit_candidate.py) · [Parameters and numerical results](/examples/seasonal-arima-20261119-v1/turn08_candidate_fit_results.json)

**Instructor**: For the displayed subset-AR model, the stationary working-series mean is $\mu=c/(1-\phi_1-\phi_3-\phi_5)$. The mean describes a year-over-year difference of fourth roots. Estimation method and initialization can change coefficients from historical software output.

**Human**: Does this candidate leave any structure in the residuals? Show us the diagnostic evidence.

**Instructor**: Before revealing: Which residual patterns would make you revisit this model? What can a large test p-value establish?

<!-- checkpoint: 9. Diagnostic checks -->
**Assistant**: **The diagnostics show no compelling remaining serial correlation.** The candidate appears to have captured the main short-term and seasonal dependence, although the residual distribution has some departures from normality.

Here, residuals are **one-step prediction errors** for the seasonal differences \(v_t=y_t^*-y_{t-12}^*\), divided by their model-implied standard deviations. I used the saved fitted parameters, including the constant.

![Residual time plot, ACF, normal Q–Q plot, and residuals versus fitted values](/examples/seasonal-arima-20261119-v1/turn09_residual_diagnostics.png)

The residual ACF is reassuring. No correlation through lag 36 exceeds the approximate pointwise reference bands, \(\pm0.157\). In particular, the seasonal correlations are now very small:

\[
\hat\rho(12)=-0.001,\qquad
\hat\rho(24)=-0.013,\qquad
\hat\rho(36)=0.042.
\]

Recall that seasonal differencing alone left a lag-12 ACF of about −0.348. The fitted seasonal MA term has accounted for that prominent feature.

The **Ljung–Box test** checks residual correlations jointly through a specified lag:

| Through lag | Test statistic | Reference degrees of freedom | \(p\)-value |
|---|---:|---:|---:|
| 12 | 6.36 | 8 | 0.607 |
| 24 | 24.88 | 20 | 0.206 |
| 36 | 36.95 | 32 | 0.251 |

I subtract four degrees of freedom for the three fitted AR coefficients and one seasonal MA coefficient. Under this convention, neither the intercept nor the innovation variance is counted. These approximate tests provide no strong evidence against zero residual autocorrelations. Omitting the first 12 prediction errors gives similar conclusions, so initialization is not driving the result.

The time plot shows no obvious recurring annual pattern or sustained trend. The residual spread also looks reasonably stable: mean squared standardized errors are 0.93 in the first half and 1.07 in the second. The plot against fitted values shows no clear funnel.

The Q–Q plot does show a heavier negative tail, including an error near −3 at month 87. A Jarque–Bera normality check gives \(p=0.129\), so normality is not rejected, but the tail deviations remain relevant when assessing prediction intervals.

**This candidate passes these residual checks reasonably well.** That supports its adequacy for describing dependence; it does not establish independence, optimality, or future forecast accuracy.

[Diagnostic script](/examples/seasonal-arima-20261119-v1/turn09_residual_diagnostics.py) · [Numerical results](/examples/seasonal-arima-20261119-v1/turn09_residual_diagnostics_results.json)

**Instructor**: Inspect the plots and joint evidence. Failure to reject selected residual correlations does not prove independence or future forecast accuracy.

**Human**: Forecast the next year. Walk us through the first month and show how we return to rooms per day, including uncertainty.

**Instructor**: Before revealing: If we predict a seasonal difference, what must be added back before returning to rooms per day? Is raising a transformed mean to the fourth power always an expected room count?

<!-- checkpoint: 10. Forecasts in rooms -->
**Assistant**: **The fitted candidate forecasts another summer peak, reaching about 1,174 occupied rooms per day in August.** These are monthly averages across the four hotels.

![Next-year hotel-room forecasts with pointwise 95% prediction intervals](/examples/seasonal-arima-20261119-v1/turn10_next_year_forecast.png)

For the first month, \(t=169\), forecast the seasonal difference using information through \(t=168\):
\[
\begin{aligned}
\hat v_{169|168}
={}&0.04811+0.2428v_{168}-0.2187v_{166}\\
&-0.1587v_{164}-0.5149\hat\varepsilon_{157}.
\end{aligned}
\]
The future surprise has conditional mean zero. The constant and AR terms contribute \(0.07447\); the estimated seasonal innovation contribution is \(-0.02812\). Together,
\[
\hat v_{169|168}=0.04635.
\]
The historical innovation is inferred by the state-space model, so its uncertainty is included in prediction.

Next, **undo the seasonal difference**. Last January, \(t=157\), had 811 occupied rooms per day:
\[
y_{157}^*=811^{1/4}=5.33648.
\]
Since \(y_{169}^*=y_{157}^*+v_{169}\),
\[
\hat y_{169|168}^*=5.33648+0.04635=5.38283.
\]

Finally, **undo the fourth root**. Raising the forecast to the fourth power gives
\[
5.38283^4=839.55.
\]
This is approximately the predictive **median**, not the mean. Under the fitted Gaussian forecast, with root-scale mean \(m\) and variance \(s^2\),
\[
E(Y)=m^4+6m^2s^2+3s^4.
\]
For January, \(s=0.02473\), giving a mean of **839.65 rooms per day**. The correction is small here.

The root-scale 95% interval is \([5.33436,5.43131]\). Both endpoints are positive, so taking their fourth powers gives **[810, 870] rooms per day**, rounded.

| Month | Forecast mean | 95% prediction interval |
|---|---:|---:|
| Jan | 840 | 810–870 |
| Feb | 768 | 739–797 |
| Mar | 777 | 748–807 |
| Apr | 867 | 835–900 |
| May | 852 | 820–885 |
| Jun | 974 | 938–1,011 |
| Jul | 1,146 | 1,105–1,188 |
| Aug | 1,174 | 1,132–1,216 |
| Sep | 902 | 868–936 |
| Oct | 902 | 868–937 |
| Nov | 783 | 752–814 |
| Dec | 889 | 856–924 |

These are **pointwise prediction intervals** for each future monthly average, conditional on the fitted parameters. They include future innovations and state uncertainty, but exclude parameter and model-selection uncertainty. They do not provide simultaneous 95% coverage for the whole year. No future observations are supplied to verify these forecasts.

[Script](/examples/seasonal-arima-20261119-v1/turn10_next_year_forecast.py) · [Numerical results](/examples/seasonal-arima-20261119-v1/turn10_next_year_forecast_results.json) · [Forecast CSV](/examples/seasonal-arima-20261119-v1/turn10_next_year_forecast.csv)

**Instructor**: For horizons up to 12 months, the same-month value from the preceding year is observed. Beyond that horizon, the reconstruction must use earlier forecasts. Inverse transformation and uncertainty require explicit assumptions.

**Human**: How would this have done if we had forecast the last observed year? Compare it with using last year’s same-month values.

**Instructor**: Before revealing: Optional extension after class. At a past forecast date, which observations were available, and what would be a reasonable seasonal benchmark?

<!-- checkpoint: 11. Optional historical evaluation -->
**Assistant**: **The candidate beats last year’s same-month values overall, but not in every month.** Its mean absolute error is about 19 rooms per day, compared with 26 for the simpler forecast.

I placed the forecast origin at the end of year 13, \(t=156\). I refitted the candidate, including its constant, using only months 1–156, then forecast all 12 months of year 14 without updating as observations arrived. The training fit converged without warnings.

The comparison forecast is **seasonal naive**:
\[
\hat Y_{156+h|156}=Y_{144+h},\qquad h=1,\ldots,12.
\]
It predicts each month using the observed value for that same month in year 13.

![Retrospective year-14 forecasts and monthly errors for the candidate and seasonal-naive baseline](/examples/seasonal-arima-20261119-v1/turn11_retrospective_comparison.png)

All errors below are measured on the original **rooms-per-day scale**. The candidate uses the predictive mean after accounting for the fourth-power back-transformation.

| Measure | Candidate | Same month last year |
|---|---:|---:|
| Mean absolute error (MAE) | 18.65 | 25.58 |
| Root mean squared error (RMSE) | 21.96 | 34.61 |
| Mean error: forecast − observed | +5.88 | −23.92 |

The candidate reduces MAE by **27%** and RMSE by **37%**. It has the smaller absolute error in **7 of the 12 months**.

The error plot explains part of the advantage. The same-month baseline preserves the annual shape but makes no allowance for growth. It therefore underpredicts this year overall. The candidate incorporates positive average year-over-year change, reducing that systematic underprediction, although it slightly overpredicts overall.

Individual months still favor the simpler method. In August, for example, the observed value is 1,124: last year’s value predicts 1,125, while the candidate predicts about 1,149. In July, the candidate predicts about 1,111 against an observed 1,110.

The candidate’s pointwise 95% prediction intervals contain **11 of 12 observations**. One year provides little evidence about long-run interval coverage.

This remains a **retrospective illustration**: we chose the transformation and candidate structure after examining all 14 years. Refitting avoids using year-14 values to estimate parameters, but does not undo that earlier selection. The result supports the candidate’s usefulness in this year, not a general claim of superior forecasting.

[Script](/examples/seasonal-arima-20261119-v1/turn11_retrospective_comparison.py) · [Numerical results](/examples/seasonal-arima-20261119-v1/turn11_retrospective_results.json) · [Monthly forecasts](/examples/seasonal-arima-20261119-v1/turn11_retrospective_forecasts.csv)

**Instructor**: All 168 observations have already appeared in course analyses. This retrospective comparison illustrates time-ordered evaluation; it is not an untouched test and does not establish future performance.

**Human**: A made-up AI summary says seasonal differencing makes a seasonal MA term unnecessary, and raising the transformed mean to the fourth power gives expected occupancy. What should we correct?

**Instructor**: Before revealing: Explain the two claims in the next prompt separately. Which evidence supports each conclusion, and what is still missing?

<!-- checkpoint: 12. Checking an AI claim -->
**Assistant**: We should correct **both claims**. Each confuses two operations that serve different purposes.

**1. Seasonal differencing does not automatically remove seasonal dependence.**

For the hotel, seasonal differencing formed
\[
v_t=y_t^*-y_{t-12}^*.
\]
It substantially reduced the recurring annual pattern. But its ACF still had a sizeable lag-12 correlation, about **−0.348**. Thus, comparing corresponding months did not make the resulting changes independent across years.

A seasonal MA term describes dependence remaining in those changes:
\[
v_t=c+\phi_1v_{t-1}+\phi_3v_{t-3}+\phi_5v_{t-5}
+\varepsilon_t+\Theta\varepsilon_{t-12}.
\]
In our fitted candidate, \(\hat\Theta=-0.515\). After fitting, the residual lag-12 ACF was approximately **−0.001**. That supports retaining seasonal dependence in this candidate, although it does not prove that this is the only suitable model. Differencing itself can also induce correlation, so a seasonal spike needs interpretation.

**2. Raising the transformed mean to the fourth power does not generally give expected occupancy.**

Let \(Z\) be a future fourth-root observation, conditional on the observed history. Then
\[
Y=Z^4,\qquad E(Y)=E(Z^4),
\]
and generally
\[
E(Z^4)\ne [E(Z)]^4.
\]

If the predictive distribution of \(Z\) is Gaussian with mean \(m\) and variance \(s^2\), its fourth moment gives
\[
\boxed{E(Y)=m^4+6m^2s^2+3s^4.}
\]
The correction depends on forecast uncertainty and is positive when \(s^2>0\). For our next-January forecast, \(m^4=839.55\), while the corrected mean is **839.65 rooms per day**. The difference happens to be small here. Because the forecast distribution lies overwhelmingly above zero, \(m^4\) is approximately the predictive median.

A corrected summary would read:

> Seasonal differencing reduces recurring annual structure, but seasonal correlation may remain and require additional modeling. Expected occupancy must account for uncertainty when reversing the fourth-root transformation; simply taking the fourth power gives a different forecast summary.

These corrections follow from the [seasonal-difference evidence](/examples/seasonal-arima-20261119-v1/turn07_seasonal_acf_pacf_results.json), [residual diagnostics](/examples/seasonal-arima-20261119-v1/turn09_residual_diagnostics_results.json), and [forecast calculations](/examples/seasonal-arima-20261119-v1/turn10_next_year_forecast_results.json).

**Instructor**: This deliberately constructed claim practices checking reasoning and explaining a follow-up. It does not specify future assessment coverage.

**Instructor**: Further reading: [Seasonal ARIMA models, Forecasting: Principles and Practice](https://otexts.com/fpp3/arima.html#seasonal-arima-models). Explain the transformations and units before trusting a forecast.
