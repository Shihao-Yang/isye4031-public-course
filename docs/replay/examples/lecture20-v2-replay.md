# Title: Lecture 20: Holt–Winters and seasonal forecasting

**Instructor**: Lectures 19 and 20 devote two full lessons to smoothing. Stationary models and ARIMA follow in a later lesson. Work through this lesson in short sessions, writing your own answer before revealing each response.

**Instructor**: Learning goals. Identify additive and proportional seasonal patterns, initialize and update all seasonal states, forecast across seasonal boundaries, interpret uncertainty, and compare forecast procedures using evidence available at each origin.

**Instructor**: This page replays an actual completed Codex CLI conversation. Human prompts were simulated for teaching; the assistant executed the calculations and produced the figures. Its responses are preserved, with only portable file links changed. Instructor notes and discussion pauses were added separately. Historical software reference values are labeled and are not represented as newly executed software.

**Human**: For our seasonal forecasting lesson, show the mountain-bike, Tiger Sports Drink, and Tasty Cola data. How do their seasonal patterns differ, and what should our forecasts capture?

**Instructor**: Compare the early and late years. Are the seasonal gaps more stable in units or as ratios? Explain why a forecast can fall after a peak even when the underlying trend rises.

<!-- checkpoint: 1. Seasonal patterns in the original data -->
**Assistant**: All three series show **growth and a repeating seasonal cycle**. The main difference is whether the seasonal swings stay roughly constant in units or grow with the sales level.

![Mountain-bike, Tiger Sports Drink, and Tasty Cola sales. Left panels show observations in time order; right panels overlay years to reveal seasonal timing and changes in amplitude.](/examples/smoothing-selfstudy-v2/seasonal_patterns_turn12.png)

For **mountain bikes**, these are quarterly sales in units:

| Year | Q1 | Q2 | Q3 | Q4 |
|---|---:|---:|---:|---:|
| 1 | 10 | 31 | 43 | 16 |
| 2 | 11 | 33 | 45 | 17 |
| 3 | 13 | 34 | 48 | 19 |
| 4 | 15 | 37 | 51 | 21 |

Sales peak in **Q3** and bottom out in **Q1**. The overall level rises, but the seasonal spread changes relatively little: the Q3–Q1 difference grows from \(43-10=33\) to \(51-15=36\) bikes. This suggests **additive seasonality**, where seasons contribute roughly fixed numbers of units.

For **Tiger Sports Drink**, quarterly sales are in **thousands of cases**:

| Year | Q1 | Q2 | Q3 | Q4 |
|---|---:|---:|---:|---:|
| 1 | 72 | 116 | 136 | 96 |
| 2 | 77 | 123 | 146 | 101 |
| 3 | 81 | 131 | 158 | 109 |
| 4 | 87 | 140 | 167 | 120 |
| 5 | 94 | 147 | 177 | 128 |
| 6 | 102 | 162 | 191 | 134 |
| 7 | 106 | 170 | 200 | 142 |
| 8 | 115 | 177 | 218 | 149 |

Tiger also peaks in **Q3** and bottoms out in **Q1**, but its seasonal spread grows substantially:

\[
136-72=64,\qquad 218-115=103.
\]

The peak-to-trough ratios are much more stable:

\[
\frac{136}{72}\approx1.89,\qquad
\frac{218}{115}\approx1.90.
\]

This suggests **multiplicative seasonality**: seasonal effects scale with the level.

For **Tasty Cola**, monthly sales are in **hundreds of cases**:

| Month | Year 1 | Year 2 | Year 3 |
|---|---:|---:|---:|
| January | 189 | 244 | 298 |
| February | 229 | 296 | 378 |
| March | 249 | 319 | 373 |
| April | 289 | 370 | 443 |
| May | 260 | 313 | 374 |
| June | 431 | 556 | 660 |
| July | 660 | 831 | 1004 |
| August | 777 | 960 | 1153 |
| September | 915 | 1152 | 1388 |
| October | 613 | 759 | 904 |
| November | 485 | 607 | 715 |
| December | 277 | 371 | 441 |

Cola has a **12-month cycle**, with a September peak and a January trough. The seasonal swings grow with the level: the September–January difference increases from \(915-189=726\) to \(1388-298=1090\), while the corresponding ratios remain roughly similar, at 4.84 and 4.66. Multiplicative seasonality is a reasonable candidate here too.

These raw spreads include some within-year trend, so they are descriptive evidence, not definitive model-selection tests.

Our forecasts should capture three features:

| Series | Seasonal period \(m\) | Pattern to preserve | Starting model choice |
|---|---:|---|---|
| Mountain bikes | 4 quarters | Rising level; Q3 peak, Q1 trough; fairly stable seasonal spread | Additive Holt–Winters |
| Tiger Sports Drink | 4 quarters | Rising level; Q3 peak, Q1 trough; growing seasonal spread | Multiplicative Holt–Winters |
| Tasty Cola | 12 months | Rising level; September peak, January trough; growing seasonal spread | Multiplicative Holt–Winters |

Holt’s method supplies a projected level \(L_t+hB_t\). Seasonal forecasting adds a seasonal effect \(S\):

\[
\text{Additive:}\qquad
\hat y_{t+h\mid t}=L_t+hB_t+S_{\text{target season}},
\]

\[
\text{Multiplicative:}\qquad
\hat y_{t+h\mid t}=(L_t+hB_t)S_{\text{target season}}.
\]

In the additive model, \(S\) is measured in sales units. In the multiplicative model, it is a dimensionless factor. Thus, a forecast can correctly **fall after a seasonal peak even while the underlying trend continues upward**.

These plots use the mountain-bike series from Table 6.8 in [bikes.csv](/examples/smoothing-selfstudy-v2/bikes.csv). The separately supplied [Chapter 8 bike series](/examples/smoothing-selfstudy-v2/bikes_ch8.csv) has different later observations and should remain distinct.

Sources: Bowerman, O’Connell, and Koehler, fourth edition, Tables 6.8, 8.2, and 7.1, documented in [DATA.md](/examples/smoothing-selfstudy-v2/turn12_DATA.md). [Tiger data](/examples/smoothing-selfstudy-v2/tiger.csv) · [Cola data](/examples/smoothing-selfstudy-v2/cola.csv) · [Plotting script](/examples/smoothing-selfstudy-v2/seasonal_data_turn12.py) · [Descriptive results](/examples/smoothing-selfstudy-v2/seasonal_summary_turn12.csv).

**Instructor**: Seasonality is additive when differences are roughly constant in units and multiplicative when they are roughly proportional to the level. The proportional interpretation also allows decreasing amplitude when the level falls. The original bike, Tiger and Cola time plots are recreated here, with year overlays added.

**Human**: The two bike datasets differ. Show that difference, then use the Chapter 8 series to explain how we initialize the level, trend, and four additive seasonal factors.

**Instructor**: The earlier bike plot and the worked smoothing example use different values after quarter 8. What would happen if we silently mixed their plots, fitted coefficients and forecast tables?

<!-- checkpoint: 2. Two bike series and initial states -->
**Assistant**: The datasets agree for the first eight quarters, then diverge. **The Chapter 8 series has stronger growth in years 3 and 4.** Each list below is ordered Q1, Q2, Q3, Q4.

| Year | Table 6.8 series | Chapter 8 series | Chapter 8 minus Table 6.8 |
|---|---|---|---|
| 1 | 10, 31, 43, 16 | 10, 31, 43, 16 | 0, 0, 0, 0 |
| 2 | 11, 33, 45, 17 | 11, 33, 45, 17 | 0, 0, 0, 0 |
| 3 | 13, 34, 48, 19 | **14, 36, 50, 21** | +1, +2, +2, +2 |
| 4 | 15, 37, 51, 21 | **19, 41, 55, 25** | +4, +4, +4, +4 |

![The two mountain-bike series diverge after quarter 8. Initial additive seasonal factors estimated from the first two years are negative in Q1 and Q4 and positive in Q2 and Q3.](/examples/smoothing-selfstudy-v2/bikes_initialization_turn13.png)

For initialization, we need a **seasonally adjusted level**, a **trend**, and **four seasonal effects**. I’ll use an explicit two-year initialization based on the first eight Chapter 8 observations. This is one reasonable convention; different conventions can produce different starting states.

Write the initial model as

\[
y_t\approx L_0+tB_0+S_q,
\]

where \(t\) is the quarter index, \(q\) is the quarter of the year, \(L_0\) is the time-zero level, \(B_0\) is growth in bikes per quarter, and \(S_q\) is an additive seasonal adjustment.

**First, estimate the trend from the two annual averages.**

\[
\bar y_1=\frac{10+31+43+16}{4}=25,
\]

\[
\bar y_2=\frac{11+33+45+17}{4}=26.5.
\]

The averages are four quarters apart, so

\[
\boxed{B_0=\frac{26.5-25}{4}=0.375\text{ bikes per quarter}.}
\]

Comparing complete years helps keep recurring seasonal differences out of the trend estimate.

**Next, align the level with the correct time.** The first year’s average corresponds to the midpoint of quarters 1–4, at \(t=2.5\). Thus,

\[
25=L_0+2.5B_0,
\]

giving

\[
\boxed{L_0=25-2.5(0.375)=24.0625.}
\]

Our initial seasonally adjusted baseline is therefore

\[
L_0+tB_0=24.0625+0.375t.
\]

**Finally, remove that baseline and average the residuals for each season.**

For example, the two Q1 observations occur at \(t=1\) and \(t=5\):

\[
\begin{aligned}
S_{\mathrm{Q1}}
&=\frac{[10-(24.0625+0.375)]
+[11-(24.0625+5(0.375))]}{2}\\
&=\frac{-14.4375-14.9375}{2}\\
&=\boxed{-14.6875}.
\end{aligned}
\]

Applying the same calculation to all four quarters:

| Season | Year-1 detrended sales | Year-2 detrended sales | Initial seasonal factor |
|---|---:|---:|---:|
| Q1 | −14.4375 | −14.9375 | **−14.6875** |
| Q2 | +6.1875 | +6.6875 | **+6.4375** |
| Q3 | +17.8125 | +18.3125 | **+18.0625** |
| Q4 | −9.5625 | −10.0625 | **−9.8125** |

These factors sum to zero:

\[
-14.6875+6.4375+18.0625-9.8125=0.
\]

That constraint separates the overall level from seasonality. Q3 adds about **18.06 bikes** to the baseline; Q1 subtracts about **14.69 bikes**.

If we begin updating **after the eight-quarter initialization window**, the corresponding level at quarter 8 is

\[
L_8=L_0+8B_0=27.0625,
\]

with initial trend \(0.375\) and the four factors above. Before any further updates, the quarter-9 forecast is

\[
\hat y_{9\mid8}
=L_8+B_0+S_{\mathrm{Q1}}
=27.0625+0.375-14.6875
=\boxed{12.75\text{ bikes}}.
\]

Additive Holt–Winters then updates the level, trend, and the seasonal factor for the quarter just observed, using \(\alpha,\gamma,\delta\), respectively.

Because both datasets have identical first eight observations, **this initialization is identical for both**. Their later smoothing updates and fitted forecasts will differ.

Sources: Table 6.8, p. 318, and the Chapter 8 series in Figure 8.9, p. 370, documented in [DATA.md](/examples/smoothing-selfstudy-v2/turn13_DATA.md). [Quarter-by-quarter differences](/examples/smoothing-selfstudy-v2/bikes_difference_turn13.csv) · [Initialization calculations](/examples/smoothing-selfstudy-v2/bikes_init_turn13.csv) · [Python script](/examples/smoothing-selfstudy-v2/bikes_initialization_turn13.py).

**Instructor**: Both source versions are retained. This response uses a two-cycle initialization to explain the components. The next checkpoint switches explicitly to the source’s all-16-observation regression convention, which reproduces its worked arithmetic. Neither convention should be hidden when comparing results.

**Human**: Now initialize from a regression on all 16 Chapter 8 observations and average the detrended values by quarter. Walk through the first additive Holt–Winters updates with alpha = 0.20, gamma = 0.10, and delta = 0.10.

**Instructor**: Compute the quarter-one forecast before revealing the result. After sales of 10, which seasonal factor changes? Should the quarter-two error be positive or negative when actual sales are 31?

<!-- checkpoint: 3. Additive initialization and updates -->
**Assistant**: Using all 16 Chapter 8 observations gives the initial regression

\[
\boxed{\hat y_t=20.85+0.980882t.}
\]

We will use its **intercept as the time-zero level**, its **slope as the initial trend**, and the average regression residual for each quarter as its additive seasonal factor.

For \(t=1,\ldots,16\),

\[
\bar t=8.5,\qquad \bar y=\frac{467}{16}=29.1875.
\]

Thus,

\[
B_0=
\frac{\sum(t-\bar t)(y_t-\bar y)}
{\sum(t-\bar t)^2}
=\frac{333.5}{340}
=\boxed{0.980882},
\]

\[
L_0=\bar y-B_0\bar t
=29.1875-0.980882(8.5)
=\boxed{20.85}.
\]

The trend is measured in **bikes per quarter**.

To initialize seasonality, subtract the regression baseline from each observation:

\[
r_t=y_t-(20.85+0.980882t).
\]

Then average these detrended values by quarter:

| Season | Year 1 residual | Year 2 residual | Year 3 residual | Year 4 residual | Average seasonal factor |
|---|---:|---:|---:|---:|---:|
| Q1 | −11.8309 | −14.7544 | −15.6779 | −14.6015 | **−14.2162** |
| Q2 | +8.1882 | +6.2647 | +5.3412 | +6.4176 | **+6.5529** |
| Q3 | +19.2074 | +17.2838 | +18.3603 | +19.4368 | **+18.5721** |
| Q4 | −8.7735 | −11.6971 | −11.6206 | −11.5441 | **−10.9088** |

For example,

\[
S_{\mathrm{Q1}}
=\frac{-11.8309-14.7544-15.6779-14.6015}{4}
=-14.2162.
\]

The four factors sum to zero, apart from numerical rounding. For the recursion starting at \(t=1\), assign them to

\[
S_{-3},S_{-2},S_{-1},S_0
\]

in Q1–Q4 order. This makes \(S_{t-4}\) the factor for the same quarter of the previous cycle.

The additive Holt–Winters equations are

\[
\hat y_{t\mid t-1}=L_{t-1}+B_{t-1}+S_{t-4},
\]

\[
\boxed{L_t=0.20(y_t-S_{t-4})+0.80(L_{t-1}+B_{t-1}),}
\]

\[
\boxed{B_t=0.10(L_t-L_{t-1})+0.90B_{t-1},}
\]

\[
\boxed{S_t=0.10(y_t-L_t)+0.90S_{t-4}.}
\]

The level update removes the old seasonal effect. The trend update smooths the change in level. The seasonal update removes the **updated level** and smooths what remains.

**Quarter 1: actual sales are 10 bikes.**

The initial prediction is

\[
\hat y_{1\mid0}
=20.85+0.980882-14.216176
=\boxed{7.614706}.
\]

Update the level:

\[
L_1
=0.20[10-(-14.216176)]
 +0.80(20.85+0.980882)
=\boxed{22.307941}.
\]

Update the trend:

\[
B_1
=0.10(22.307941-20.85)+0.90(0.980882)
=\boxed{1.028588}.
\]

Update the Q1 seasonal factor:

\[
S_1
=0.10(10-22.307941)+0.90(-14.216176)
=\boxed{-14.025353}.
\]

Sales exceeded the prediction, so the level and trend rise, and Q1’s seasonal adjustment becomes slightly less negative.

**Quarter 2: actual sales are 31 bikes.**

Use Q2’s initial seasonal factor, \(S_{-2}=6.552941\):

\[
\hat y_{2\mid1}
=22.307941+1.028588+6.552941
=\boxed{29.889471}.
\]

Then

\[
L_2=0.20(31-6.552941)+0.80(22.307941+1.028588)
=\boxed{23.558635},
\]

\[
B_2=0.10(23.558635-22.307941)+0.90(1.028588)
=\boxed{1.050799},
\]

\[
S_2=0.10(31-23.558635)+0.90(6.552941)
=\boxed{6.641784}.
\]

Continuing gives:

| Quarter index | Season | Actual | Prediction before update | Updated level | Updated trend | Updated seasonal factor |
|---|---|---:|---:|---:|---:|---:|
| 1 | Q1 | 10 | 7.6147 | 22.3079 | 1.0286 | −14.0254 |
| 2 | Q2 | 31 | 29.8895 | 23.5586 | 1.0508 | +6.6418 |
| 3 | Q3 | 43 | 43.1815 | 24.5731 | 1.0472 | +18.5575 |
| 4 | Q4 | 16 | 14.7115 | 25.8780 | 1.0729 | −10.8057 |
| 5 | Q1 | 11 | 12.9256 | 26.5658 | 1.0344 | −14.1794 |

At quarter 5, the seasonal cycle returns to Q1. We use its **updated** factor \(S_1=-14.025353\):

\[
\hat y_{5\mid4}
=25.878008+1.072939-14.025353
=12.925595.
\]

That is how the model carries each quarter’s learned seasonal effect into the next year.

![Regression initialization and additive Holt–Winters fitted values for the Chapter 8 bike series, followed by forecasts made at quarter 16.](/examples/smoothing-selfstudy-v2/bikes_hw_turn14.png)

Because all 16 observations helped initialize the model, these historical predictions are **fitted values**, including the early ones. The updates recreate the stated formulas; they do not provide an honest future accuracy evaluation.

[Regression and seasonal calculations](/examples/smoothing-selfstudy-v2/bikes_hw_initialization_turn14.csv) · [All updates](/examples/smoothing-selfstudy-v2/bikes_hw_updates_turn14.csv) · [Python script](/examples/smoothing-selfstudy-v2/bikes_hw_turn14.py) · [Source attribution](/examples/smoothing-selfstudy-v2/turn14_DATA.md)

**Instructor**: The original worked example’s quarter-two error has a sign typo: 31−29.8895 is +1.1105. The executed calculation uses the positive value. The source also has an isolated 0.9089 trend typo; the correct regression slope is 0.980882. The linked ledger contains all 16 updates.

**Human**: Show the full initial-versus-optimized bike fits and forecasts. Can the historical software comparison be reproduced using the earlier bike series and initializing level and trend from its first year?

**Instructor**: Does a fitted seasonal smoothing constant of zero remove the seasonal pattern? Can constants optimized for one dataset and initialization be called optimal for a different setup?

<!-- checkpoint: 4. Complete bike fits and forecasts -->
**Assistant**: **Yes. The historical bike accuracy numbers can be reproduced to their reported precision**, using the earlier bike series and a particular combination of starting values: first-year regression for level and trend, but all-16 regression residuals for seasonality.

First, here is the complete comparison for our **Chapter 8 calculation**, retaining the initialization from the previous answer.

We minimize

\[
\mathrm{SSE}(\alpha,\gamma,\delta)
=\sum_{t=1}^{16}
\left[y_t-(L_{t-1}+B_{t-1}+S_{t-4})\right]^2,
\]

with \(L_0=20.85\), \(B_0=0.980882\), and the four initial seasonal factors fixed.

| Quantity | Initial parameters | Optimized parameters |
|---|---:|---:|
| \(\alpha\), level | 0.20 | **0.560644** |
| \(\gamma\), trend | 0.10 | **0** |
| \(\delta\), seasonality | 0.10 | **0** |
| MAD, bikes | 1.0705 | **0.8494** |
| MSD, bikes² | 1.5760 | **1.1748** |
| MAPE | 5.6639% | **4.7019%** |

Here, zero trend and seasonal smoothing mean those components **remain at their initial estimates**. The model still has a trend and seasonality; it updates only the level.

The full fitted values are:

| Quarter index | Actual | Initial fit | Optimized fit |
|---|---:|---:|---:|
| 1 | 10 | 7.615 | 7.615 |
| 2 | 31 | 29.889 | 30.702 |
| 3 | 43 | 43.181 | 43.869 |
| 4 | 16 | 14.711 | 14.882 |
| 5 | 11 | 12.926 | 13.182 |
| 6 | 33 | 34.242 | 33.709 |
| 7 | 45 | 46.919 | 46.311 |
| 8 | 17 | 18.143 | 17.076 |
| 9 | 14 | 15.489 | 14.707 |
| 10 | 36 | 36.832 | 36.061 |
| 11 | 50 | 49.429 | 49.027 |
| 12 | 21 | 21.155 | 21.072 |
| 13 | 19 | 18.633 | 18.705 |
| 14 | 41 | 40.398 | 40.621 |
| 15 | 55 | 53.422 | 53.833 |
| 16 | 25 | 25.340 | 25.987 |

Forecasts made after quarter 16 use the projected level plus the appropriate seasonal factor:

| Target | Initial forecast | Optimized forecast |
|---|---:|---:|
| Quarter 17, Q1 | 22.867 | **23.107** |
| Quarter 18, Q2 | 44.614 | **44.857** |
| Quarter 19, Q3 | 57.620 | **57.857** |
| Quarter 20, Q4 | 29.062 | **29.357** |

![Top: initial and optimized Chapter 8 bike fits and forecasts. Bottom: the earlier series under the initialization that reproduces the historical accuracy numbers, including a separate reoptimization. All plotted forecasts are our formula-based calculations.](/examples/smoothing-selfstudy-v2/bikes_fits_forecasts_turn15.png)

For the **historical comparison**, use the earlier [bikes.csv](/examples/smoothing-selfstudy-v2/bikes.csv) series. Regression on its first year,

\[
(10,31,43,16),
\]

gives

\[
\boxed{L_0=17.5,\qquad B_0=3.}
\]

That starting slope is much steeper than our Chapter 8 all-16 estimate of \(0.980882\).

The seasonal initialization also matters. Using the earlier series’ **all-16 regression residuals**, averaged by quarter, gives

\[
\boxed{
(S_{\mathrm{Q1}},S_{\mathrm{Q2}},S_{\mathrm{Q3}},S_{\mathrm{Q4}})
=(-14.520588,\ 6.326471,\ 18.673529,\ -10.479412).
}
\]

With these starting values, the calculations reproduce both historical rows:

| Setup | MAD | MSD | MAPE |
|---|---:|---:|---:|
| Historical initial report: \((.20,.10,.10)\) | 6.0284 | 41.5876 | 29.2106% |
| Our recreation | **6.0284** | **41.5876** | **29.2106%** |
| Historical supplied constants: \((.561,0,0)\) | 4.0380 | 17.4331 | 19.8810% |
| Our recreation | **4.0380** | **17.4331** | **19.8810%** |

Initializing the seasonal factors from the first year alone does **not** reproduce those numbers. The agreement requires the stated combination of first-year level/trend and all-16 seasonal estimation. It establishes numerical agreement, without proving the original software’s internal procedure.

There is another important distinction: **the historical supplied constants are not the optimum for this earlier-series setup under our criterion**. Reoptimizing it gives approximately

\[
(\alpha,\gamma,\delta)=(0.809389,\ 0.422058,\ 0),
\]

with MSD **3.8322**, below 17.4331.

| Earlier-series calculation | Q17 | Q18 | Q19 | Q20 |
|---|---:|---:|---:|---:|
| Initial \((.20,.10,.10)\) | 22.741 | 44.646 | 58.125 | 30.092 |
| Historical supplied \((.561,0,0)\) | 22.245 | 46.092 | 61.439 | 35.286 |
| Reoptimized for this setup | 17.723 | 39.022 | 51.821 | 23.120 |

With \(\gamma=0\), the historical supplied-constants calculation retains the steep initial trend of **3 bikes per quarter**. That helps explain its increasingly high forecasts.

All these errors describe historical fit. In particular, the all-16 seasonal initialization uses the full series, so none of these comparisons demonstrates future forecasting accuracy.

[Calculation script](/examples/smoothing-selfstudy-v2/bikes_comparison_turn15.py) · [Fit summaries](/examples/smoothing-selfstudy-v2/bikes_fit_summary_turn15.csv) · [Full fitted values](/examples/smoothing-selfstudy-v2/bikes_fitted_values_turn15.csv) · [Eight-quarter forecasts](/examples/smoothing-selfstudy-v2/bikes_forecasts_turn15.csv) · [Historical reference](/examples/smoothing-selfstudy-v2/software_reference.csv) · [Attribution and conventions](/examples/smoothing-selfstudy-v2/turn15_DATA.md)

**Instructor**: The initial and supplied-constant software panels are numerically recreated to their reported accuracy measures. They use the earlier bike series, first-year level/trend regression, and full-series seasonal initialization. Matching the metrics supports this reconstruction but does not prove the historical program’s internal settings. The source’s label “optimal” applies to the Chapter 8 setup, not automatically to the earlier-series software plot.

**Human**: How do we forecast the next two years and form prediction intervals for the Chapter 8 bikes? Compare both parameter settings, and use the error-correction form to explain what happens when a season comes around again.

**Instructor**: Which stored seasonal factor belongs to horizon 5 when the seasonal period is four? Predict how a positive innovation affects next year’s forecast for the same quarter.

<!-- checkpoint: 5. Seasonal forecasts across two years -->
**Assistant**: For the Chapter 8 bikes, **two years means eight quarterly forecasts**. Both parameter settings preserve the Q3 peak and Q1 trough while projecting growth.

At quarter 16, forecast \(h\) quarters ahead using

\[
\hat y_{16+h\mid16}=L_{16}+hB_{16}+S_{\text{target quarter}},
\]

where \(L_{16}\) is the latest level, \(B_{16}\) is growth per quarter, and \(S\) is the latest seasonal factor.

| Setting | \(\alpha\) | \(\gamma\) | \(\delta\) | \(L_{16}\) | \(B_{16}\) |
|---|---:|---:|---:|---:|---:|
| Initial | 0.20 | 0.10 | 0.10 | 36.1813 | 0.9544 |
| Optimized | 0.560644 | 0 | 0 | 36.3426 | 0.9809 |

For example, the initial-setting forecast for quarter 17, a Q1, is

\[
36.181314+0.954375-14.269183
=\boxed{22.866506\text{ bikes}}.
\]

The two-year forecasts and **approximate 95% prediction intervals** are:

| Target | Initial forecast | Initial interval | Optimized forecast | Optimized interval |
|---|---:|---|---:|---|
| Q17: year 5, Q1 | 22.87 | [20.41, 25.33] | 23.11 | [20.98, 25.23] |
| Q18: year 5, Q2 | 44.61 | [42.09, 47.13] | 44.86 | [42.42, 47.29] |
| Q19: year 5, Q3 | 57.62 | [55.03, 60.21] | 57.86 | [55.15, 60.57] |
| Q20: year 5, Q4 | 29.06 | [26.40, 31.73] | 29.36 | [26.40, 32.32] |
| Q21: year 6, Q1 | 26.68 | [23.88, 29.49] | 27.03 | [23.84, 30.22] |
| Q22: year 6, Q2 | 48.43 | [45.53, 51.34] | 48.78 | [45.37, 52.19] |
| Q23: year 6, Q3 | 61.44 | [58.43, 64.45] | 61.78 | [58.17, 65.39] |
| Q24: year 6, Q4 | 32.88 | [29.76, 36.00] | 33.28 | [29.48, 37.08] |

![Eight quarterly bike forecasts with approximate prediction intervals for the initial and optimized Chapter 8 models.](/examples/smoothing-selfstudy-v2/bikes_intervals_turn16.png)

To understand the updates and intervals, define the one-step forecast error

\[
e_t=y_t-(L_{t-1}+B_{t-1}+S_{t-4}).
\]

Our additive Holt–Winters equations can be rewritten as

\[
\boxed{
\begin{aligned}
L_t&=L_{t-1}+B_{t-1}+\alpha e_t,\\
B_t&=B_{t-1}+\alpha\gamma e_t,\\
S_t&=S_{t-4}+\delta(1-\alpha)e_t.
\end{aligned}}
\]

The seasonal coefficient is **\(\delta(1-\alpha)\)** because our seasonal update subtracts the newly updated level:

\[
y_t-L_t=S_{t-4}+(1-\alpha)e_t.
\]

Under the initial settings, an error of \(e_t\) therefore contributes:

| Component | Correction |
|---|---:|
| Level | \(0.20e_t\) |
| Trend | \(0.02e_t\) |
| Seasonal factor for this quarter | \(0.08e_t\) |

**When the same season returns four quarters later**, its forecast includes all three corrections. Relative to the forecast made just before observing \(y_t\), the revision for \(y_{t+4}\) is

\[
\underbrace{0.20e_t}_{\text{level}}
+\underbrace{4(0.02e_t)}_{\text{four quarters of trend}}
+\underbrace{0.08e_t}_{\text{returning season}}
=\boxed{0.36e_t}.
\]

This describes the immediate revision caused by that observation; later observations can cause further revisions.

Under the optimized settings, \(\gamma=\delta=0\). An error changes only the level, by \(0.560644e_t\). The seasonal pattern remains fixed, so the returning season receives **no additional seasonal correction**.

For prediction intervals, assume these errors are independent normal innovations with variance \(\sigma^2\). An intervening error \(j\) quarters before the target has coefficient

\[
w_j=\alpha+j\alpha\gamma+
\delta(1-\alpha)\mathbf1_{\{j\text{ is a multiple of }4\}},
\]

where the indicator is one when the error occurred in the target’s season. Hence,

\[
\operatorname{Var}(y_{16+h}-\hat y_{16+h\mid16})
=\sigma^2\left[1+\sum_{j=1}^{h-1}w_j^2\right].
\]

For both fits, I used the historical plug-in estimate \(s^2=\mathrm{SSE}/16=\mathrm{MSD}\):

\[
s_{\text{initial}}=1.2554,\qquad
s_{\text{optimized}}=1.0839\text{ bikes}.
\]

Each interval is the point forecast plus or minus \(1.96\) times its prediction standard error.

The optimized model starts with less estimated noise, but its larger \(\alpha\) transmits future errors more strongly into the level. **Its intervals become wider than the initial model’s from horizon 3 onward.** Smaller fitting error therefore does not automatically imply narrower long-range intervals.

These intervals condition on the fitted starting states and parameters. They omit estimation uncertainty, and the all-16 initialization makes the historical errors fitted errors. Their actual future coverage has not been established.

[Python calculations](/examples/smoothing-selfstudy-v2/bikes_intervals_turn16.py) · [Forecasts and intervals](/examples/smoothing-selfstudy-v2/bikes_intervals_turn16.csv) · [Sources and assumptions](/examples/smoothing-selfstudy-v2/turn16_DATA.md)

**Instructor**: This response first uses SSE/T for its noise estimate. The next recorded follow-up reproduces the textbook’s SSE/(T−3) convention. The seasonal forecast uses the latest available factor for the target season, cycling through the stored factors beyond the first year. It never uses an unobserved future seasonal state.

**Human**: Use the worked example’s variance estimate, SSE divided by 16 minus 3. Recompute the first three bike intervals and show how that convention changes the two-year interval plot.

**Instructor**: If the variance divisor falls from 16 to 13, do the point forecasts change? Calculate the factor by which interval widths change.

<!-- checkpoint: 6. The textbook interval convention -->
**Assistant**: Using the worked example’s convention,

\[
\boxed{s^2=\frac{\mathrm{SSE}}{16-3}=\frac{\mathrm{SSE}}{13},}
\]

makes the intervals wider. The point forecasts remain unchanged.

| Parameter setting | SSE | Revised \(s^2\) | Revised \(s\), bikes |
|---|---:|---:|---:|
| Initial: \((0.20,0.10,0.10)\) | 25.2166 | 1.9397 | 1.3927 |
| Optimized: \((0.560644,0,0)\) | 18.7975 | 1.4460 | 1.2025 |

We retain the same horizon variance formula and normal multiplier:

\[
\hat y_{16+h\mid16}
\ \pm\
1.96s\sqrt{
1+\sum_{j=1}^{h-1}
\left[\alpha+j\alpha\gamma+
\delta(1-\alpha)\mathbf1_{\{4\mid j\}}\right]^2
}.
\]

For example, the initial model’s one-quarter interval becomes

\[
22.8665\pm1.96\sqrt{\frac{25.2166}{13}}
=\boxed{[20.1368,\ 25.5962]}.
\]

The first three revised **95% prediction intervals** are:

| Target | Initial forecast | Revised interval | Optimized forecast | Revised interval |
|---|---:|---|---:|---|
| Q17, Q1 | 22.87 | **[20.14, 25.60]** | 23.11 | **[20.75, 25.46]** |
| Q18, Q2 | 44.61 | **[41.82, 47.41]** | 44.86 | **[42.16, 47.56]** |
| Q19, Q3 | 57.62 | **[54.75, 60.49]** | 57.86 | **[54.85, 60.87]** |

Changing the divisor multiplies every prediction standard error, and therefore every interval width, by

\[
\sqrt{\frac{16}{13}}=1.1094.
\]

Thus, **all intervals are 10.94% wider**, at every horizon and for both parameter settings.

![Two-year bike forecasts with revised SSE/13 prediction intervals. Purple dotted lines show the previous SSE/16 bounds; the forecasts are unchanged.](/examples/smoothing-selfstudy-v2/bikes_variance_convention_turn17.png)

The plot’s seasonal shape and relative widening across horizons remain the same. Only the uncertainty scale changes. These remain conditional approximations that omit initialization and parameter-estimation uncertainty.

[Revised eight-quarter results](/examples/smoothing-selfstudy-v2/bikes_intervals_turn17.csv) · [Python calculations](/examples/smoothing-selfstudy-v2/bikes_variance_turn17.py) · [Sources and convention](/examples/smoothing-selfstudy-v2/turn17_DATA.md)

**Instructor**: Use this checkpoint for the source-matching interval values. Dividing by T−3 is the textbook’s plug-in convention, not an exact degrees-of-freedom result for an estimated dynamic model with fitted initial states. All intervals remain conditional approximations. The detailed shock-weight derivation is optional; interpreting the interval is core.

**Human**: How does multiplicative Holt–Winters work for Tiger Sports Drink? Initialize from the first 16 quarters and walk through the first updates with alpha = 0.20, gamma = 0.10, and delta = 0.10.

**Instructor**: What units do the seasonal factors have now? Explain why deseasonalization uses division. Predict the direction of each update when sales exceed the forecast.

<!-- checkpoint: 7. Multiplicative initialization and updates -->
**Assistant**: Multiplicative Holt–Winters represents seasonality with **multipliers**. A strong quarter increases sales by a proportion of the underlying level, so its seasonal swing grows as sales grow.

For Tiger, \(y_t\) is quarterly sales in **thousands of cases**. Let \(L_t\) denote the seasonally adjusted level, \(B_t\) the trend in thousands of cases per quarter, and \(S_t\) the dimensionless seasonal multiplier.

We initialize at time zero using quarters 1–16, then run updates from quarter 1.

**First, fit a regression to the first 16 observations.**

\[
\bar t=8.5,\qquad \bar y=\frac{1860}{16}=116.25.
\]

The slope and intercept are

\[
B_0=\frac{840}{340}=\boxed{2.470588},
\]

\[
L_0=116.25-2.470588(8.5)=\boxed{95.25}.
\]

Thus, the regression baseline is

\[
\hat y_t^{\,\text{baseline}}=95.25+2.470588t.
\]

**Next, divide sales by the baseline and average by quarter.** For example, the first Q1 ratio is

\[
\frac{72}{95.25+2.470588(1)}=0.736795.
\]

Averaging the four Q1 ratios gives

\[
\frac{0.736795+0.715594+0.689448+0.683062}{4}
=0.706225.
\]

The four raw factors average \(0.99997345\). I normalize them to average exactly one by dividing each by that number, while retaining the regression level and trend.

| Season | Raw average ratio | Initial normalized multiplier |
|---|---:|---:|
| Q1 | 0.706225 | **0.706243** |
| Q2 | 1.111389 | **1.111418** |
| Q3 | 1.293694 | **1.293729** |
| Q4 | 0.888586 | **0.888610** |

Q1 sales are initially about **29.4% below** the baseline; Q3 sales are about **29.4% above** it. Assign these initial factors to \(S_{-3},S_{-2},S_{-1},S_0\), respectively.

![Tiger sales and multiplicative Holt–Winters predictions, alongside the initial seasonal multipliers.](/examples/smoothing-selfstudy-v2/tiger_hw_turn18.png)

With \(\alpha=0.20,\gamma=0.10,\delta=0.10\), the equations are

\[
\hat y_{t\mid t-1}=(L_{t-1}+B_{t-1})S_{t-4},
\]

\[
\boxed{L_t=0.20\frac{y_t}{S_{t-4}}
+0.80(L_{t-1}+B_{t-1}),}
\]

\[
\boxed{B_t=0.10(L_t-L_{t-1})+0.90B_{t-1},}
\]

\[
\boxed{S_t=0.10\frac{y_t}{L_t}+0.90S_{t-4}.}
\]

Division removes multiplicative seasonality. The level and trend remain in sales units; the seasonal factors remain ratios.

**Quarter 1: actual sales are 72.**

The prediction is

\[
\hat y_{1\mid0}
=(95.25+2.470588)(0.706243)
=\boxed{69.014516}.
\]

Deseasonalize the observation:

\[
\frac{72}{0.706243}=101.947863.
\]

Update the level:

\[
L_1=0.20(101.947863)+0.80(95.25+2.470588)
=\boxed{98.566043}.
\]

Update the trend:

\[
B_1=0.10(98.566043-95.25)+0.90(2.470588)
=\boxed{2.555134}.
\]

Finally, update Q1’s multiplier:

\[
S_1
=0.10\left(\frac{72}{98.566043}\right)
+0.90(0.706243)
=\boxed{0.708666}.
\]

Sales exceeded the prediction, so the level and trend rise, and the Q1 multiplier rises slightly.

**Quarter 2: actual sales are 116.**

Use Q2’s starting multiplier, \(1.111418\):

\[
\hat y_{2\mid1}
=(98.566043+2.555134)(1.111418)
=\boxed{112.387908}.
\]

Then

\[
L_2
=0.20\left(\frac{116}{1.111418}\right)
+0.80(98.566043+2.555134)
=\boxed{101.771174},
\]

\[
B_2
=0.10(101.771174-98.566043)+0.90(2.555134)
=\boxed{2.620133},
\]

\[
S_2
=0.10\left(\frac{116}{101.771174}\right)
+0.90(1.111418)
=\boxed{1.114257}.
\]

Continuing the same calculation:

| Quarter index | Season | Actual | Prediction | Updated level | Updated trend | Updated seasonal multiplier |
|---|---|---:|---:|---:|---:|---:|
| 1 | Q1 | 72 | 69.0145 | 98.5660 | 2.5551 | 0.708666 |
| 2 | Q2 | 116 | 112.3879 | 101.7712 | 2.6201 | 1.114257 |
| 3 | Q3 | 136 | 135.0540 | 104.5375 | 2.6348 | 1.294452 |
| 4 | Q4 | 96 | 95.2344 | 107.3446 | 2.6520 | 0.889181 |
| 5 | Q1 | 77 | 77.9509 | 109.7282 | 2.6252 | 0.707973 |

When Q1 returns at quarter 5, its prediction uses the **updated** Q1 factor:

\[
\hat y_{5\mid4}
=(107.344624+2.651989)(0.708666)
=77.950914.
\]

Actual sales of 77 fall slightly below that prediction, pulling the estimates downward relative to their projected values.

For future forecasts, multiply the projected level by the latest factor for the target season:

\[
\boxed{\hat y_{t+h\mid t}=(L_t+hB_t)S_{\text{target season}}.}
\]

The first 16 historical predictions are fitted values because those observations helped initialize the model. These are calculations under the stated normalization and update conventions, not proprietary software output or future accuracy results.

[Initialization calculations](/examples/smoothing-selfstudy-v2/tiger_initialization_turn18.csv) · [All updates](/examples/smoothing-selfstudy-v2/tiger_updates_turn18.csv) · [Python script](/examples/smoothing-selfstudy-v2/tiger_hw_turn18.py) · [Source attribution](/examples/smoothing-selfstudy-v2/turn18_DATA.md)

**Instructor**: The source rounds seasonal factors and regression coefficients in its hand arithmetic. This recording keeps full precision and normalizes the factors to mean one, so a few displayed final decimals differ. Positive seasonal factors and valid levels are needed for these ratio updates. The complete 32-quarter ledger is linked.

**Human**: Fit the Tiger smoothing constants and compare the initial and optimized fitted curves and forecasts. How do these results compare with the historical software reports?

**Instructor**: The fitted constants match the historical report after rounding, but the error totals differ. What conclusion is justified, and which inputs would you check next?

<!-- checkpoint: 8. Initial and fitted Tiger models -->
**Assistant**: The fitted Tiger constants are

\[
\boxed{\hat\alpha=0.335515,\qquad
\hat\gamma=0.045566,\qquad
\hat\delta=0.133978.}
\]

These round to the historical report’s **\((0.336,0.046,0.134)\)**. However, our fitted errors differ substantially from the reported software errors.

I kept our first-16-quarter initialization fixed:

\[
L_0=95.25,\qquad B_0=2.470588,
\]

with normalized seasonal multipliers

\[
(0.706243,\ 1.111418,\ 1.293729,\ 0.888610).
\]

For each candidate parameter triple, I ran all 32 updates and minimized

\[
\boxed{
\mathrm{SSE}(\alpha,\gamma,\delta)
=\sum_{t=1}^{32}
\left[y_t-(L_{t-1}+B_{t-1})S_{t-4}\right]^2.
}
\]

| Quantity | Initial | Optimized |
|---|---:|---:|
| \(\alpha\), level | 0.20 | **0.335515** |
| \(\gamma\), trend | 0.10 | **0.045566** |
| \(\delta\), seasonality | 0.10 | **0.133978** |
| SSE | 177.2758 | **168.4457** |
| MAD | 1.9721 | **1.8671** |
| MSD | 5.5399 | **5.2639** |
| MAPE | 1.5982% | **1.4957%** |

Sales and MAD are measured in **thousands of cases**. The optimized model gives more weight to new observations when updating level and seasonality, while smoothing changes in trend more gradually.

![Tiger initial and optimized fitted curves and two-year forecasts, alongside the historical reported MSD values. Historical bars are transcribed reference values; forecast curves are our calculations.](/examples/smoothing-selfstudy-v2/tiger_fit_turn19.png)

The fitted curves almost overlap. Their forecasts also differ only modestly. At quarter 32, the estimated level and trend are:

| State | Initial | Optimized |
|---|---:|---:|
| \(L_{32}\) | 167.8857 | 168.1170 |
| \(B_{32}\) | 2.2436 | 2.3027 |

Using

\[
\hat y_{32+h\mid32}=(L_{32}+hB_{32})S_{\text{target quarter}},
\]

the next two years’ forecasts are:

| Target | Initial forecast | Optimized forecast |
|---|---:|---:|
| Q33: Q1 | 119.90 | **120.05** |
| Q34: Q2 | 190.40 | **190.65** |
| Q35: Q3 | 225.74 | **226.37** |
| Q36: Q4 | 157.50 | **157.97** |
| Q37: Q1 | 126.22 | **126.54** |
| Q38: Q2 | 200.32 | **200.82** |
| Q39: Q3 | 237.35 | **238.29** |
| Q40: Q4 | 165.49 | **166.17** |

Both preserve the Q3 peak, Q1 trough, and growing seasonal swings.

The historical accuracy comparison is different:

| Calculation | MAD | MSD | MAPE |
|---|---:|---:|---:|
| Historical initial report | 11.808 | 207.153 | 9.692% |
| Our initial calculation | **1.972** | **5.540** | **1.598%** |
| Historical supplied-constants report | 10.971 | 149.232 | 8.771% |
| Reported constants with our initialization | **1.867** | **5.264** | **1.495%** |

Using the historical rounded constants in our calculation gives almost exactly our optimized result. **Matching smoothing constants does not reproduce the historical error statistics.** Starting states and the treatment of early predictions also matter.

As an initialization check, regression on Tiger’s first year gives \(L_0=82\) and \(B_0=9.2\). Keeping our seasonal factors but using those states raises MSD to:

- **197.860** with the initial constants;
- **139.964** with the historical supplied constants.

Those values move much closer to the historical reports, showing how a steep initial trend can inflate early errors. They still do not reproduce the reports exactly, so the remaining difference is unresolved.

The available historical reference supplies accuracy measures, but no Tiger forecast coordinates. The plotted forecasts are our formula-based results. These comparisons concern historical fit, not demonstrated future accuracy.

[Python script](/examples/smoothing-selfstudy-v2/tiger_fit_turn19.py) · [Fit summary](/examples/smoothing-selfstudy-v2/tiger_fit_summary_turn19.csv) · [All fitted values](/examples/smoothing-selfstudy-v2/tiger_fitted_values_turn19.csv) · [Forecasts](/examples/smoothing-selfstudy-v2/tiger_forecasts_turn19.csv) · [Historical reference](/examples/smoothing-selfstudy-v2/software_reference.csv) · [Attribution](/examples/smoothing-selfstudy-v2/turn19_DATA.md)

**Instructor**: The original Tiger time plot and initial-versus-fitted comparison are recreated using the source’s numerical series. Historical software accuracy values are displayed separately. Their exact fit paths remain unresolved because the complete initialization is not documented; the new curves are explicitly calculated from the first 16 quarters. This limitation is preserved, not concealed by altering the data.

**Human**: Compare additive and multiplicative Holt–Winters for Tasty Cola using 0.20 for all three constants. Show the fitted curves and next-year forecasts with prediction intervals, and explain the differences.

**Instructor**: Before revealing the curves, predict which model will give the larger September peak and lower January trough. Does multiplicative seasonality force us to assume multiplicative random errors?

<!-- checkpoint: 9. Cola additive and multiplicative forecasts -->
**Assistant**: **Multiplicative seasonality fits Tasty Cola better in this calculation.** As sales grow, its seasonal peaks and troughs spread farther apart. The multiplicative model projects that widening; the additive model uses seasonal adjustments measured in fixed sales units.

Both models use

\[
\alpha=\gamma=\delta=0.20,\qquad m=12.
\]

Sales are measured in **hundreds of cases**. I initialized both models from months 1–24: the regression gives \(L_0=283.6159\) and \(B_0=17.8174\). Additive seasonal factors are average regression residuals by month; multiplicative factors are average ratios to the regression baseline, normalized to average one. Updates then start at month 1.

Their forecast equations are

\[
\text{Additive:}\quad
\hat y_{36+h\mid36}=L_{36}+hB_{36}+S_{\text{target month}},
\]

\[
\text{Multiplicative:}\quad
\hat y_{36+h\mid36}=(L_{36}+hB_{36})S_{\text{target month}}.
\]

For example, an additive September effect adds a number of cases. A multiplicative September factor increases the projected baseline by a proportion.

![Tasty Cola additive and multiplicative Holt–Winters fitted curves and next-year forecasts, with approximate 95% prediction intervals. The shaded historical region supplies initialization.](/examples/smoothing-selfstudy-v2/cola_comparison_turn20.png)

The historical fitting measures favor the multiplicative model:

| Measure | Additive | Multiplicative |
|---|---:|---:|
| MAD, hundreds of cases | 63.51 | **33.57** |
| MSD, squared sales units | 6,596.08 | **1,587.75** |
| MAPE | 13.97% | **7.09%** |

The additive model struggles to follow the growing seasonal spread. Its next-year forecast has higher troughs and a lower September peak than the multiplicative forecast.

The next-year point forecasts and **approximate 95% prediction intervals** are:

| Month | Additive forecast | Additive interval | Multiplicative forecast | Multiplicative interval |
|---|---:|---|---:|---|
| January | 575.4 | [409.1, 741.7] | 402.9 | [321.1, 484.0] |
| February | 636.8 | [465.8, 807.8] | 485.2 | [400.5, 570.4] |
| March | 661.9 | [484.6, 839.1] | 510.8 | [422.2, 598.9] |
| April | 719.3 | [534.3, 904.3] | 588.5 | [493.4, 684.0] |
| May | 677.8 | [483.3, 872.3] | 505.8 | [409.8, 601.6] |
| June | 904.3 | [698.8, 1109.8] | 861.4 | [731.3, 990.6] |
| July | 1173.2 | [955.1, 1391.4] | 1288.4 | [1102.5, 1474.1] |
| August | 1297.7 | [1065.4, 1530.0] | 1479.1 | [1253.8, 1704.0] |
| September | 1470.3 | [1222.4, 1718.1] | 1738.0 | [1458.9, 2016.3] |
| October | 1092.8 | [828.0, 1357.5] | 1134.3 | [929.7, 1338.3] |
| November | 943.6 | [660.7, 1226.5] | 888.8 | [711.7, 1066.1] |
| December | 714.1 | [411.8, 1016.3] | 521.5 | [393.9, 648.8] |

To make the uncertainty comparison explicit, **both models assume independent normal errors on the original sales scale**. Multiplicative *seasonality* does not itself require multiplicative *errors*.

Using the variance convention from our worked examples,

\[
s^2=\frac{\mathrm{SSE}}{36-3},
\]

gives estimated error standard deviations of **84.83** for the additive model and **41.62** for the multiplicative model.

For the additive model, I propagated forecast-error variance analytically. For the multiplicative model, I simulated **100,000 future paths**, updating the states after every simulated observation, and used their 2.5th and 97.5th percentiles. The recorded seed is **20261030**.

Two interval differences are visible:

- Additive interval widths increase with the horizon.
- Multiplicative interval widths also depend on the target season. Uncertainty in the projected baseline is amplified by a large seasonal multiplier, so the September interval is particularly wide.

These are formula-based results under the stated initialization and error assumptions. The first 24 observations helped initialize the models, so the fitting comparison does not establish future accuracy. The intervals omit parameter and initialization uncertainty.

[Python calculations](/examples/smoothing-selfstudy-v2/cola_comparison_turn20.py) · [Fit summary](/examples/smoothing-selfstudy-v2/cola_fit_summary_turn20.csv) · [Fitted values](/examples/smoothing-selfstudy-v2/cola_fitted_values_turn20.csv) · [Forecasts and intervals](/examples/smoothing-selfstudy-v2/cola_forecasts_intervals_turn20.csv) · [Sources and assumptions](/examples/smoothing-selfstudy-v2/turn20_DATA.md)

**Instructor**: Both curves use the same 0.20 constants and an explicit first-24-month initialization. The original software plot reports additive MAPE 16.01%, MAD 59.78, MSD 9425.48, and multiplicative MAPE 9.58%, MAD 46.93, MSD 6812.61. These are transcribed historical values, not the metrics of the newly executed curves. Differences in initialization and interval conventions are kept explicit. Pointwise 95% intervals do not give a 95% guarantee for the whole next-year path.

**Human**: Would the cola model comparison hold up when forecasting later observations? Compare the procedures with a seasonal-naive baseline using time-ordered evaluation, and show what the forecast errors reveal.

**Instructor**: Describe the information available at origin 24 and at origin 25. Which forecasts can update after January is observed? What can a mean error close to zero hide?

<!-- checkpoint: 10. A chronological forecast comparison -->
**Assistant**: **Yes, multiplicative Holt–Winters still wins in this time-ordered comparison.** It outperforms both additive Holt–Winters and seasonal naive on all three accuracy measures. The errors also reveal remaining problems that the overall scores hide.

I used **months 1–24 for initialization and training**, then evaluated months 25–36. Both Holt–Winters procedures retain \(\alpha=\gamma=\delta=0.20\); no constants were tuned using the evaluation year.

The seasonal-naive baseline predicts the same month’s sales from the previous year:

\[
\boxed{\hat y_{t\mid t-1}=y_{t-12}.}
\]

We compare two forecasting tasks:

- **Year ahead:** issue all 12 forecasts at month 24.
- **Rolling one step:** forecast the next month, observe its sales, update the states, and repeat.

Seasonal naive gives identical forecasts under these two protocols for this single evaluation year, because every lagged observation comes from year 2.

![Third-year Cola forecasts and signed forecast errors. Left: forecasts issued at month 24. Right: forecasts updated monthly. Positive errors mean underprediction.](/examples/smoothing-selfstudy-v2/cola_evaluation_turn21.png)

The year-ahead results are:

| Procedure | MAD | MSD | MAPE | Mean error |
|---|---:|---:|---:|---:|
| Seasonal naive | 112.75 | 16,073.75 | 16.74% | +112.75 |
| Additive Holt–Winters | 103.13 | 13,703.64 | 23.34% | −78.57 |
| **Multiplicative Holt–Winters** | **68.92** | **5,559.43** | **11.45%** | −68.92 |

The rolling one-step results are:

| Procedure | MAD | MSD | MAPE | Mean error |
|---|---:|---:|---:|---:|
| Seasonal naive | 112.75 | 16,073.75 | 16.74% | +112.75 |
| Additive Holt–Winters | 98.07 | 13,272.13 | 17.36% | +5.99 |
| **Multiplicative Holt–Winters** | **43.77** | **2,416.76** | **7.35%** | +15.79 |

MAD and mean error are in **hundreds of cases**; MSD is in squared sales units. Errors are \(e_t=y_t-\hat y_t\), so positive values indicate underprediction.

The error plots reveal three useful findings.

**Seasonal naive misses growth.** Every third-year observation exceeds its corresponding second-year value, so all 12 errors are positive. For September,

\[
e_{33}=1388-1152=236,
\]

an underprediction of **23,600 cases**. Repeating last year captures the seasonal timing but misses the rising level.

**The additive model still misses the seasonal shape.** With monthly updates, it overpredicts early-year sales and underpredicts the summer peak. For example:

\[
e_{\text{January}}=-172.29,\qquad
e_{\text{September}}=+205.80.
\]

Its mean error of only \(+5.99\) looks reassuring because large positive and negative errors cancel. MAD and MSD expose those large misses. Its year-ahead MAPE is even worse than seasonal naive’s, despite having lower MAD and MSD.

**The multiplicative model improves substantially, but its errors are not patternless.** Its year-ahead forecasts overpredict all 12 observations. Updating monthly reduces this problem, although the rolling errors switch from negative early in the year to positive later:

| Month | Multiplicative rolling error |
|---|---:|
| January | −55.30 |
| May | −16.70 |
| September | +100.55 |
| December | +42.89 |

That sequence suggests the estimated trend and seasonal pattern still need scrutiny. A lower average error does not establish that the remaining errors behave like independent, zero-mean noise.

For example, the rolling January forecast was \(353.30\), against actual sales of \(298\):

\[
e_{25}=298-353.30=-55.30.
\]

The model then used January’s observed sales to update February’s forecast. The year-ahead procedure had no such opportunity, which explains its larger errors.

This supports multiplicative seasonality **for these procedures and this evaluation year**. The exercise is retrospective on data we have already inspected, so it is not a newly untouched test. It nevertheless separates training from later outcomes and shows why evaluation must match the intended forecasting horizon.

[Evaluation script](/examples/smoothing-selfstudy-v2/cola_evaluation_turn21.py) · [Accuracy results](/examples/smoothing-selfstudy-v2/cola_evaluation_metrics_turn21.csv) · [Every forecast and error](/examples/smoothing-selfstudy-v2/cola_evaluation_errors_turn21.csv) · [Sources and evaluation conventions](/examples/smoothing-selfstudy-v2/turn21_DATA.md)

**Instructor**: The complete procedures use only the prefix available at each origin. This is a historical comparison of already-inspected data, so it does not establish an untouched test result. The plots show that even the better procedure has structured errors. The planning horizon and operational loss should determine which comparison matters.

**Human**: Give us a short worked example for checking an AI-generated seasonal forecast. Show the consequences of plausible mistakes visually, and explain how we can catch them ourselves.

**Instructor**: Try the January calculation yourself. Then explain how a wrong month, an after-observation update, or a missing unit conversion could produce a plausible-looking forecast table.

<!-- checkpoint: 11. Audit an AI forecast -->
**Assistant**: Use **one forecast you can reconstruct by hand** before trusting an AI-generated forecast table.

For Cola, the multiplicative model’s January forecast, made after month 24, is

\[
\hat y_{25\mid24}=(L_{24}+B_{24})S_{\mathrm{Jan}}
=(635.7126+15.2701)(0.542723)
=\boxed{353.303\text{ hundreds of cases}}.
\]

That means **35,330 cases**. January’s actual sales were 298 hundreds of cases, so its forecast error was

\[
e_{25}=298-353.303=-55.303.
\]

![Three deliberate mistakes: shifting the seasonal index moves the peak, using current sales makes errors artificially smaller, and forgetting the unit conversion changes the forecast by a factor of 100.](/examples/smoothing-selfstudy-v2/seasonal_audit_turn22.png)

Three plausible mistakes show why this check matters:

| Mistake | Consequence | How to catch it |
|---|---|---|
| Use February’s factor for January | January forecast becomes **410.181**, and the forecast peak shifts one month early. | Label every seasonal factor with its month. Check that September remains the peak. |
| Update using January’s actual sales before scoring January | A reconstructed value becomes **342.242**, appearing more accurate than the forecast actually issued. | Write down the forecast **before** revealing the observation. |
| Label “hundreds of cases” as “cases” | Report **353 cases** instead of **35,330**. | Carry units through the equation and conversion. |

The leakage mistake is especially deceptive. Suppose the AI reports the after-observation reconstruction

\[
\tilde y_{25}=0.20y_{25}+0.80\hat y_{25\mid24}.
\]

Its error is automatically

\[
y_{25}-\tilde y_{25}=0.80e_{25}.
\]

Applying this mistake throughout the evaluation year reduces MSD from **2,416.76 to 1,546.73**, exactly a factor of \(0.80^2=0.64\). That apparent improvement comes from seeing the answers, not forecasting better.

Before accepting the output, check **one arithmetic calculation, one seasonal label, the units, and the information available at the forecast date**. Then compare later forecast errors with seasonal naive. Those checks test the procedure behind a convincing-looking curve.

These are deliberately constructed mistakes applied to our Cola calculation. [Python script](/examples/smoothing-selfstudy-v2/seasonal_audit_turn22.py) · [Numerical results](/examples/smoothing-selfstudy-v2/seasonal_audit_turn22.csv) · [Source attribution](/examples/smoothing-selfstudy-v2/DATA.md)

**Instructor**: The mistakes in this checkpoint are deliberately constructed. They complement the authentic recorded thermostat transcription correction and the bike source-data audit. Finish by explaining one forecast, one diagnostic concern, and the next evidence you would request without relying on the assistant’s wording.

**Instructor**: [Study guide and independent practice](/examples/smoothing-selfstudy-v2/STUDY-GUIDE.md) · [Data attribution](/examples/smoothing-selfstudy-v2/DATA.md) · [Reproduction order](/examples/smoothing-selfstudy-v2/reproduction-order.txt). These materials support independent interpretation and explanation; they do not announce exam coverage or a graded submission.
