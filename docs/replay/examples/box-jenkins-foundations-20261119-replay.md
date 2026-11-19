# Title: ARIMA foundations: the paper-towel story

**Instructor**: Follow the paper-towel example from its business question through stationarity, differences, ACF and PACF. Allow about 75 minutes including discussion. Model fitting continues in a later lesson.

**Instructor**: Learning goals. Explain the sales units and calculate differences; construct lagged pairs and read an ACF; distinguish cutoff from decay; explain what PACF controls for; check whether an AI conclusion follows from the evidence. Make your own judgment at each pause before revealing the answer.

**Instructor**: Provenance. This is a completed Codex CLI conversation recorded for this lesson. Human prompts are simulated instructor questions. The code and numerical analyses actually ran. This classroom edition selects 12 recorded exchanges. One trial answer that moved ahead too early is omitted; the full 13-turn record is preserved privately. Selected responses retain their recorded wording; only asset links were made portable. Instructor notes were added separately. Playback displays saved results. Source: Bowerman, O’Connell and Koehler, Forecasting, Time Series, and Regression, fourth edition, Chapter 9, Examples 9.1–9.5.

**Human**: Remind us of the paper-towel sales problem and show us the data. What are we trying to predict?

**Instructor**: Before revealing: What is one observation, and what future quantity must the company predict?

<!-- checkpoint: 1. Sales question -->
**Assistant**: The Olympia Paper Company wants to **predict future weekly paper-towel sales** so it can plan production, storage, and budgets. We have sales from **120 consecutive weeks**. The calendar dates are unspecified, so we label them week 1 through week 120.

![Paper-towel sales across 120 observed weeks.](/examples/arima-foundations-20261119-v1/turn01_paper_towel_data.png)

Each point is one week’s observed sales. The horizontal axis shows time; the vertical axis shows sales in **thousands of rolls**. For example, a height of 250 means 250,000 rolls sold that week. The connected points help us follow the observations in their time order.

The data file uses a scaled measurement: it records sales above 100,000 rolls, in units of 10,000 rolls. Thus, its first value of 15 means

\[
100{,}000 + 10{,}000 \times 15 = 250{,}000 \text{ rolls}.
\]

The graph converts those recorded values back into sales quantities.

Our immediate prediction target is **sales in week 121**, the first week beyond the observed data. We may also want sales predictions for subsequent weeks. Those future observations are not supplied.

For students familiar with regression, the response is still a numerical quantity: weekly sales. What changes is that the observations have a meaningful time order. We can see stretches of higher and lower sales, which motivates asking whether past weeks help predict a future week. The picture alone does not establish how useful that information will be.

[Data](/examples/arima-foundations-20261119-v1/paper_towels.csv) · [Python script](/examples/arima-foundations-20261119-v1/turn01_paper_towel_data.py) · [Numerical results](/examples/arima-foundations-20261119-v1/turn01_paper_towel_results.json)

**Instructor**: Check the units: a recorded value of 15 means 250,000 rolls. A negative recorded value can still mean positive sales.

**Human**: What does it mean for a time series to be stationary, and why should we check that before building a forecasting model?

**Instructor**: Before revealing: Could a series fluctuate a lot and still have stable statistical properties? Explain.

<!-- checkpoint: 2. Stationarity -->
**Assistant**: A time series is **stationary** when its statistical behavior stays the same as time passes. The observations can fluctuate substantially, but the rules describing those fluctuations remain stable.

For forecasting, we usually mean three things:

- **A constant mean:** the series fluctuates around the same average level, without a systematic upward or downward trend.
- **A constant variance:** the size of the fluctuations stays similar over time.
- **A stable relationship across time:** the relationship between sales this week and sales a fixed number of weeks later depends on that separation, not on which calendar week we start from.

Stationarity does **not** mean that sales are constant, or that successive weeks are independent. Sales can remain high for several weeks and then remain low for several weeks while still following stable statistical rules.

Why check this before building a forecasting model? **We use the past to learn something that we hope will carry into the future.** If the average sales level keeps increasing, an average learned from older observations may describe current demand poorly. If fluctuations become larger over time, uncertainty estimated from earlier weeks may understate how variable future sales could be. Many forecasting models assume that some statistical features remain stable, so we need to assess whether that assumption is reasonable.

For our paper-towel data, the first step is to inspect the time plot for sustained changes in level or variability. Long stretches of high or low sales deserve attention, but they do not by themselves prove nonstationarity. A stationary series can also produce such stretches.

We observe only one finite history. Checking stationarity therefore means assessing whether stability is a useful working assumption, not proving that the underlying process will stay unchanged forever.

**Instructor**: The legacy workflow is: find a suitable working series, tentatively identify a model, estimate its parameters, check it, then forecast. Today develops the first step and the correlation tools needed for the second. Weak stationarity means a constant mean and covariance depending only on lag; this includes constant finite variance.

**Human**: Can changes from one week to the next give us a more stable series? Show us how first and second differences work on these sales.

**Instructor**: Before revealing: Calculate the change from 250,000 to 244,064 rolls. If the next week is 249,383, what is the change in that change?

<!-- checkpoint: 3. First and second differences -->
**Assistant**: Yes. **Weekly changes can have a more stable average level than sales themselves.** This transformation is called *differencing*. It can help with a changing level, but it does not automatically produce a stationary series.

Let \(y_t\) mean sales in week \(t\), measured in rolls. The **first difference**, written \(\Delta y_t\), is

\[
\Delta y_t = y_t-y_{t-1}.
\]

Here, \(\Delta\) means “take the change from the preceding week.” A positive difference means sales increased; a negative difference means they decreased.

Using our first three observations:

| Week | Sales (rolls) | First difference (rolls) |
|---|---:|---:|
| 1 | 250,000 | Unavailable |
| 2 | 244,064 | −5,936 |
| 3 | 249,383 | +5,319 |

The **second difference**, written \(\Delta^2 y_t\), applies the same operation to the first differences:

\[
\Delta^2 y_t=\Delta y_t-\Delta y_{t-1}.
\]

For week 3, this gives

\[
\Delta^2 y_3=5{,}319-(-5{,}936)=11{,}255.
\]

Sales moved from a decline to an increase, so the weekly change increased by 11,255 rolls. **Second differencing means differencing twice**, not subtracting sales from two weeks ago.

![First and second differences of paper-towel sales, with matching vertical scales.](/examples/arima-foundations-20261119-v1/turn03_sales_differences.png)

The first differences fluctuate around roughly zero, without the broad changes in sales level seen in the original plot. Their spread looks reasonably similar across time, although some stretches have larger movements. This supports considering them more stable; it does not prove stationarity.

The second differences also fluctuate around zero, but look more jagged. Their sample standard deviation, a measure of spread, is about **13,042 rolls**, compared with **11,041 rolls** for first differences. Taking another difference has increased the observed variability. More differencing does not necessarily mean more stability.

Starting with 120 sales observations leaves **119 first differences** and **118 second differences**, because each subtraction needs a preceding value.

[Python script](/examples/arima-foundations-20261119-v1/turn03_sales_differences.py) · [Numerical results](/examples/arima-foundations-20261119-v1/turn03_sales_differences_results.json) · [Calculated data](/examples/arima-foundations-20261119-v1/turn03_sales_differences.csv)

**Instructor**: In the file’s units, the first two differences are -0.5936 and 0.5319. Their difference is 1.1255. Algebra gives $\Delta^2 y_t=y_t-2y_{t-1}+y_{t-2}$. Keep the original values so that any later predictions of changes can be converted back to sales.

**Human**: Put the original sales and first differences side by side using the data file’s units. Which should we use as our working series?

**Instructor**: Before revealing: Which picture would make a constant mean a more plausible working assumption? What would the picture still leave uncertain?

<!-- checkpoint: 4. The working series -->
**Assistant**: **Use the first differences as our working series for now.** Their plot gives us a more plausible starting point for a series with a stable level.

![Original sales and first differences in the data file’s units. The vertical scales differ so each pattern is readable.](/examples/arima-foundations-20261119-v1/turn04_working_series.png)

On the left, let \(y_t\) denote the recorded sales value for week \(t\): **sales above 100,000 rolls, measured in units of 10,000 rolls**. A value of 15 therefore represents 250,000 rolls sold. A negative value is valid and means sales fell below the 100,000-roll baseline.

On the right, the first difference is

\[
\Delta y_t=y_t-y_{t-1},
\]

where \(\Delta y_t\) means the change from the previous week. Its unit is also **10,000 rolls**, but it measures a change in sales. The baseline cancels when we subtract. A difference of \(+1\) means sales increased by 10,000 rolls; zero means sales were unchanged.

The original series has broad movements in level, including a substantial decline and recovery around the middle of the record. That makes a constant average level questionable. The first differences fluctuate around approximately zero, with no similarly persistent movement in level. Their spread also looks reasonably consistent, although individual large changes remain.

These observations support **working with first differences without taking a second difference at this stage**. They do not prove stationarity: a finite plot cannot establish that the underlying mean, variance, and relationships across time are constant.

Our working series now describes **weekly sales changes**. The business quantity we ultimately care about remains weekly sales.

[Python script](/examples/arima-foundations-20261119-v1/turn04_working_series.py) · [Numerical results](/examples/arima-foundations-20261119-v1/turn04_working_series_results.json)

**Instructor**: The first difference starts at week 2, so the legacy notation is $z_2,\ldots,z_{120}$, with $b=2$ and $n-b+1=119$ observations. If we took second differences, the first available week would be 3.

**Human**: Before we go further, how do we measure the relationship between sales three weeks apart? Show the paired observations and work through the sample autocorrelation.

**Instructor**: Before revealing: Which observation pairs with week 1 at lag 3? How many such pairs can we form from 120 weeks?

<!-- checkpoint: 5. Pairs three weeks apart -->
**Assistant**: We measure the relationship by **pairing sales with sales three weeks later**. This separation is called a **lag of 3**. Here we use the original sales, in the file’s units, not our working series of first differences.

Let \(y_t\) denote recorded sales in week \(t\). Each pair is \((y_{t-3},y_t)\):

| Earlier week | Later week | Earlier sales | Later sales |
|---:|---:|---:|---:|
| 1 | 4 | 15.0000 | 16.0374 |
| 2 | 5 | 14.4064 | 15.6320 |
| 3 | 6 | 14.9383 | 14.3975 |
| 4 | 7 | 16.0374 | 13.8959 |
| 5 | 8 | 15.6320 | 14.0765 |
| … | … | … | … |
| 117 | 120 | … | … |

There are **117 pairs** from 120 observations. We shift the series by three weeks, then use every overlapping pair.

![Each point pairs original sales with sales three weeks later, in units of 10,000 rolls above the baseline.](/examples/arima-foundations-20261119-v1/turn05_lag3_sales.png)

The **sample autocorrelation** measures how strongly a series is linearly related to its own past values. Using the usual time-series convention, its lag-3 value is

\[
r_3=
\frac{\displaystyle\sum_{t=4}^{120}(y_t-\bar y)(y_{t-3}-\bar y)}
{\displaystyle\sum_{t=1}^{120}(y_t-\bar y)^2},
\]

where \(r_3\) is the lag-3 sample autocorrelation, \(\bar y\) is the average of all 120 sales values, and \(\sum\) means “add the indicated terms.”

**First, subtract the overall mean**, \(\bar y=11.5839083\). For the week-1/week-4 pair, the product of deviations is

\[
(15-11.5839083)(16.0374-11.5839083)
\approx15.2135.
\]

Pairs on the same side of the mean contribute positive products; pairs on opposite sides contribute negative products.

**Then add and standardize:**

\[
r_3=\frac{1961.9914}{2299.4753}
=\boxed{0.8532}.
\]

This is a strong positive observed association: high sales tend to accompany high sales three weeks later, and low sales accompany low sales.

This convention uses one overall mean and all observations in the denominator. Ordinary Pearson correlation of the two paired columns uses separate column means and gives a slightly different value, **0.8697**.

The association does not prove causation or stationarity. Broad changes in the original sales level can contribute to this large correlation.

[All paired observations](/examples/arima-foundations-20261119-v1/turn05_lag3_pairs.csv) · [Script](/examples/arima-foundations-20261119-v1/turn05_lag3_sales.py) · [Numerical results](/examples/arima-foundations-20261119-v1/turn05_lag3_results.json)

**Instructor**: The lag-three table and scatterplot recreate the original Example 9.2. The standard sample ACF uses one overall mean and a common denominator. Pearson correlation of the two shortened columns can differ slightly.

**Human**: For the original sales, show just lags 1–14 with two-standard-error limits that account for earlier correlations. How do we read this graph?

**Instructor**: Before revealing: If the lag-three relationship is strong, what would you expect the nearby ACF bars to look like?

<!-- checkpoint: 6. Reading the ACF -->
**Assistant**: Here are the **original sales correlations at lags 1–14**, with limits that widen to account for correlations at earlier lags.

![Original-sales sample ACF at lags 1–14, with plus and minus two Bartlett standard-error limits.](/examples/arima-foundations-20261119-v1/turn07_bartlett_acf.png)

Each blue stem shows the sample correlation between sales and sales the indicated number of weeks apart. The orange dashed curves mark **plus and minus two estimated standard errors around zero**.

We use **Bartlett’s approximation**:

\[
\operatorname{SE}(r_k)\approx
\sqrt{\frac{1+2\sum_{j=1}^{k-1}r_j^2}{n}}.
\]

Here, \(r_k\) is the sample autocorrelation at lag \(k\), \(n=120\) is the number of sales observations, and \(j\) indexes the earlier lags. The summation adds their squared sample correlations.

At lag 1, there are no earlier correlations to include, so the limits are \(\pm2/\sqrt{120}\), approximately **±0.183**. At lag 3, the calculation includes the large correlations at lags 1 and 2, widening the limits to **±0.387**. Earlier dependence increases the estimated sampling uncertainty.

Read the graph by comparing each stem with the limits **at that lag**:

- **Lags 1–7 exceed the upper limit.** Their positive correlations stand out under this approximation.
- **Lag 8 sits just inside:** its correlation is 0.5787 and its upper limit is 0.5793. Such a close boundary should not drive a strong conclusion.
- **Lags 9–14 remain inside.** This does not establish that their population correlations are zero.

The overall pattern is still striking: original sales have strong positive correlations that decline slowly as the separation increases.

These are approximate, individual-lag reference limits, not a guarantee covering all 14 lags together. Bartlett’s approximation also relies on assumptions about a stationary process. Because the original sales show broad changes in level, we should treat this graph as descriptive evidence and interpret its thresholds cautiously. It does not prove stationarity or identify a forecasting model.

[Python script](/examples/arima-foundations-20261119-v1/turn07_bartlett_acf.py) · [Numerical results](/examples/arima-foundations-20261119-v1/turn07_bartlett_acf_results.json)

**Instructor**: SAC and sample ACF name the same tool. Bands guide interpretation of individual lags under assumptions. Crossing a band does not establish causation, and staying inside it does not prove zero population correlation. The standardized ratio is $r_k/\operatorname{SE}(r_k)$; magnitude above 2 is the same visual rule as crossing the two-standard-error limits. This is an approximate screening rule, not an exact small-sample t test.

**Human**: What does it mean for an ACF to cut off or die down? Show the different shapes, including fast and slow decay.

**Instructor**: Before revealing: Describe how a cutoff differs from a gradual decay. Can a decaying pattern alternate signs?

<!-- checkpoint: 7. Cutoff and decay -->
**Assistant**: An ACF **cuts off** when its correlations become exactly zero after some lag. It **dies down** when its correlations approach zero gradually as the lag increases.

These constructed examples show the distinction. They are theoretical shapes, not estimates from the paper-towel data.

![Constructed ACF shapes showing a cutoff after lag 2, fast positive decay, slow positive decay, and alternating decay.](/examples/arima-foundations-20261119-v1/turn08_acf_shapes.png)

**Cutoff:** In the left panel, correlations are positive at lags 1 and 2, then exactly zero at every later lag. There is a definite endpoint to the nonzero correlations.

**Fast decay:** The blue curve on the right shrinks rapidly toward zero. The correlations remain mathematically nonzero, but soon become very small. In practice, a sample may not let us distinguish these tiny correlations from zero.

**Slow decay:** The orange curve remains appreciably positive across many lags. Observations far apart in time still have substantial linear association. Our original sales ACF has shown this kind of persistent positive pattern, although its exact shape differs from this illustration.

**Alternating decay:** The purple curve switches between negative and positive correlations while shrinking in magnitude. “Dies down” refers to the shrinking size of the correlations, so it does not require a smooth downward curve.

A crucial distinction is **theoretical ACF versus sample ACF**. A theoretical cutoff means exact zeros beyond a particular lag. A sample ACF calculated from finite data will usually keep showing small positive and negative values there because of sampling variation.

Consequently, a sample pattern consistent with a cutoff would have later correlations fluctuating near zero, judged alongside their standard errors. One stem falling inside a reference band does not establish a cutoff. Likewise, one later stem crossing a band does not automatically rule one out.

We read the **overall pattern across lags**, with uncertainty, instead of treating each stem as a separate verdict.

[Python script](/examples/arima-foundations-20261119-v1/turn08_acf_shapes.py) · [Numerical values](/examples/arima-foundations-20261119-v1/turn08_acf_shapes_results.json)

**Instructor**: These shape illustrations are constructed teaching examples. Use their shapes to describe evidence; finite-sample bars will not reproduce ideal population patterns exactly.

**Human**: Can the correlations also die down in a wave? Show how that differs from alternating signs at every lag.

**Instructor**: Before revealing: In a decaying wave, must the sign change at every lag? Sketch how a full cycle might look.

<!-- checkpoint: 8. Damped waves -->
**Assistant**: Yes. An ACF can **die down in a wave**, with positive and negative stretches whose size gradually shrinks. This is called a **damped oscillation**.

![Constructed ACFs comparing a damped wave with sign alternation at every lag. Gray dashed curves show shrinking envelopes, not standard-error limits.](/examples/arima-foundations-20261119-v1/turn09_acf_waves.png)

In the **left panel**, the correlations move through a wave: positive, near zero, negative, near zero, then positive again. The sign can stay the same across several consecutive lags. Successive positive peaks and negative troughs get smaller, so the ACF approaches zero.

A positive correlation means observations separated by that lag tend to be on the same side of their mean. A negative correlation means they tend to be on opposite sides. The wave therefore indicates that this relationship changes with the separation between observations.

In the **right panel**, the sign switches at **every lag**: negative at lag 1, positive at lag 2, negative at lag 3, and so on. Its magnitude also shrinks, but there are no stretches of several positive or several negative correlations.

For these constructed examples, let \(k\) denote the lag and \(\rho_k\) the theoretical autocorrelation at that lag. We used

\[
\begin{aligned}
\text{Damped wave:}\quad &\rho_k=0.88^k\cos(\pi k/4),\\
\text{Alternating decay:}\quad &\rho_k=(-0.88)^k.
\end{aligned}
\]

The factor \(0.88^k\) supplies the shrinking envelope. The cosine supplies a wave that repeats every eight lags; \(\pi\) is the usual circle constant. In the second formula, the negative base forces a sign change at every step.

These are **illustrations, not paper-towel estimates**. A sample ACF would show sampling variation around such shapes. A few sign changes alone would not establish a damped wave; we would look for a coherent sequence of shrinking peaks and troughs.

[Python script](/examples/arima-foundations-20261119-v1/turn09_acf_waves.py) · [Numerical values](/examples/arima-foundations-20261119-v1/turn09_acf_waves_results.json)

**Instructor**: A damped wave can stay positive or negative across several consecutive lags. Both the wave and the alternating example die down as their magnitudes shrink. These are constructed illustrations.

**Human**: Now show the first-difference ACF over lags 1–14 with the same kind of limits. What changed, and is this a sensible working series?

**Instructor**: Before revealing: Predict how the ACF should change when the slow-moving level is removed. Would a stationary series have to have zero correlation at every lag?

<!-- checkpoint: 9. ACF after differencing -->
**Assistant**: **Yes, first differences are a sensible working series.** Their ACF shows much less persistent correlation than the original sales.

![First-difference ACF at lags 1–14 with plus and minus two Bartlett standard-error limits, using the same vertical scale as the original-sales graph.](/examples/arima-foundations-20261119-v1/turn10_difference_acf.png)

Each stem measures the correlation between **weekly sales changes** separated by the indicated number of weeks. The dashed curves use the same Bartlett approximation as before, now calculated from the **119 first differences** and their earlier correlations.

| Lag | First-difference ACF | Two-standard-error limits |
|---:|---:|---:|
| 1 | 0.307 | ±0.183 |
| 2 | −0.065 | ±0.200 |
| 3 | −0.072 | ±0.201 |
| 9 | −0.174 | ±0.210 |
| 14 | 0.019 | ±0.217 |

**Only lag 1 crosses its limits among lags 1–14.** Its positive correlation means that a weekly change above the average change tends to be followed by another change above average. The relationship is moderate, so individual weeks can easily behave differently.

Two things changed from the original-sales graph. First, the large positive correlations that declined slowly have disappeared. Beyond lag 1, the correlations fluctuate relatively close to zero. Second, the reference limits widen only modestly, because the earlier correlations entering Bartlett’s calculation are much smaller.

This pattern is **consistent with an apparent cutoff after lag 1**, allowing for sampling variation. It does not establish that every later population correlation is exactly zero.

Together with the time plot, which showed changes fluctuating around a roughly stable level with reasonably similar spread, this supports keeping first differences as our working series. **Stationarity permits correlation:** the remaining lag-1 association is not itself a reason to difference again.

The evidence supports a practical choice, not proof of stationarity. The limits are approximate and apply to individual lags.

[Python script](/examples/arima-foundations-20261119-v1/turn10_difference_acf.py) · [Numerical results](/examples/arima-foundations-20261119-v1/turn10_difference_acf_results.json)

**Instructor**: Choosing first differences as a working series does not say that the changes are independent. This is the point to stop if the class needs more time before PACF.

**Human**: What does partial autocorrelation measure? Show what changes at lag 2 when we account for the intervening week.

**Instructor**: Before revealing: Week t and week t-2 both relate to week t-1. How could we examine what association remains after accounting for the middle week?

<!-- checkpoint: 10. Partial correlation -->
**Assistant**: **Partial autocorrelation measures the linear relationship between observations a given number of weeks apart, after accounting for the intervening observations.** At lag 2, there is just one intervening week.

We’ll use our working series. Let \(x_t=y_t-y_{t-1}\) denote the sales change in week \(t\), where \(y_t\) is recorded sales. We compare:

\[
\underbrace{x_{t-2}}_{\text{earlier change}},
\qquad
\underbrace{x_{t-1}}_{\text{intervening change}},
\qquad
\underbrace{x_t}_{\text{current change}}.
\]

Ordinary lag-2 correlation compares the earlier and current changes directly. Partial autocorrelation asks: **after removing what each change shares linearly with the intervening change, how are the remaining parts related?**

Using familiar regression steps:

1. Regress the current change on the intervening change, including an intercept. Keep the residuals, the parts left unexplained.
2. Regress the earlier change on that same intervening change, again including an intercept. Keep those residuals.
3. Correlate the two sets of residuals.

![Lag-2 relationship in first differences before and after adjusting both observations for the intervening change.](/examples/arima-foundations-20261119-v1/turn11_partial_correlation.png)

Using the same **117 matched triples**, the ordinary correlation is **−0.065**. After adjustment, the residual correlation is **−0.179**: the negative relationship becomes more pronounced, although it remains modest.

Why can adjustment strengthen the relationship? Both endpoint changes are positively related to the intervening change. That shared association pushes their unadjusted relationship in a positive direction. Removing it reveals a more negative remaining association.

For consistency with our earlier ACF calculations, the standard ACF-based lag-2 sample PACF is

\[
\widehat{\alpha}_2
=\frac{r_2-r_1^2}{1-r_1^2}
\approx-0.175,
\]

where \(\widehat{\alpha}_2\) denotes the estimated lag-2 partial autocorrelation, and \(r_1\) and \(r_2\) are our sample autocorrelations at lags 1 and 2. Its slight difference from −0.179 comes from finite-sample calculation conventions.

This adjustment describes **remaining linear association**. It does not establish a causal effect of an earlier week on the current week.

[Python script](/examples/arima-foundations-20261119-v1/turn11_partial_correlation.py) · [Numerical results](/examples/arima-foundations-20261119-v1/turn11_partial_correlation_results.json) · [Matched observations](/examples/arima-foundations-20261119-v1/turn11_partial_correlation_rows.csv)

**Instructor**: Residualizing both endpoints connects PACF to multiple regression. Different finite-sample estimators need not match exactly; distinguish the residual-correlation illustration from the textbook ACF-based calculation.

**Human**: Show the full PACF of the first differences over lags 1–14 using that ACF-based calculation. How do we read it alongside the ACF?

**Instructor**: Before revealing: Why might ACF and PACF give different values at lag 2? What pattern would you describe before naming any model?

<!-- checkpoint: 11. The paper-towel PACF -->
**Assistant**: Here is the **PACF of the first differences at lags 1–14**, calculated from the same sample ACF we have been using.

![First-difference ACF and ACF-based PACF, with approximate two-standard-error reference limits.](/examples/arima-foundations-20261119-v1/turn12_acf_pacf.png)

Read the panels as answering two related questions:

- **ACF:** How strongly are weekly sales changes related at a given separation, including associations shared with intervening weeks?
- **PACF:** How strongly are they related after accounting linearly for all intervening weeks?

At lag 1, there are no intervening observations, so the ACF and PACF agree. At lag 2, the PACF accounts for lag 1. At lag 3, it accounts for lags 1 and 2, and so on.

The calculation uses the **Yule–Walker equations**, which translate the sample autocorrelations into the coefficients needed for each successive linear adjustment. This extends our earlier ACF-based lag-2 calculation.

| Lag | ACF | PACF |
|---:|---:|---:|
| 1 | 0.307 | 0.307 |
| 2 | −0.065 | −0.175 |
| 3 | −0.072 | 0.006 |
| 4 | 0.105 | 0.133 |
| 9 | −0.174 | −0.178 |
| 14 | 0.019 | 0.010 |

The ACF retains Bartlett’s varying limits. For the PACF, the usual approximate reference limits are

\[
\pm\frac{2}{\sqrt n}\approx\pm0.183,
\]

where \(n=119\) is the number of first differences. These limits are commonly used when assessing zero partial correlations beyond a possible cutoff; they are not universal standard errors for every nonzero partial correlation.

**Lag 1 is the only crossing in either panel.** The PACF at lag 2 becomes more negative after adjustment, as we saw previously. Lags 2 and 9 sit near the lower PACF limit, so their inside/outside status should not carry much weight by itself.

Together, the graphs show that the clearest association is between consecutive weekly changes. Later relationships are smaller and less decisive. Neither panel establishes an exact population cutoff, and this sample PACF does not show a convincing smooth decay pattern. We should preserve that uncertainty when interpreting its shape.

[Python script](/examples/arima-foundations-20261119-v1/turn12_acf_pacf.py) · [Numerical results](/examples/arima-foundations-20261119-v1/turn12_acf_pacf_results.json)

**Instructor**: SPAC and sample PACF name the same tool. A borderline bar is uncertain evidence. The model families that can generate these patterns belong to the next lesson.

**Human**: Consider this made-up AI claim: “The first differences look stationary and their ACF cuts off after lag 1, so the changes are independent and our forecast is validated.” What is wrong, and what should we do next?

**Instructor**: Before revealing: Constructed AI claim: “The first differences look stationary and their ACF cuts off after lag 1, so the changes are independent and the forecast is validated.” Identify the unsupported steps and the evidence needed next.

<!-- checkpoint: 12. Checking an AI claim -->
**Assistant**: The claim jumps from a useful descriptive pattern to conclusions the evidence does not support.

**“The first differences look stationary” is a reasonable working assessment, not proof.** Their time plot looks more stable than the original sales, and their ACF lacks the original series’ slow decline. Those observations support using first differences provisionally. They cannot establish that the underlying statistical behavior stays constant.

**“Their ACF cuts off after lag 1” overstates the sample evidence.** We observed a clear lag-1 correlation, with later correlations inside approximate reference limits over lags 1–14. That is consistent with a possible cutoff. Being inside a band does not establish an exactly zero population correlation.

**“So the changes are independent” is incorrect.** The lag-1 correlation itself provides evidence of dependence between consecutive changes. Even a theoretical ACF that is exactly zero after lag 1 would still allow dependence at lag 1. More generally, zero correlation means no linear association; it does not imply independence. Variables can have nonlinear dependence while their correlation is zero.

**“Our forecast is validated” has no basis yet.** We have inspected historical observations and their correlations. We have not produced and evaluated predictions. Stationarity and ACF patterns describe aspects of a series; they do not demonstrate forecasting accuracy.

Our next step should be to keep first differences as a **provisional working series** and carry forward the evidence of short-term dependence, especially at lag 1. We should avoid declaring either independence or an exact cutoff.

Before building a forecasting model, we should also decide how its predictions will be evaluated: preserve time order, use earlier observations to predict later observations, and compare the predictions with what actually happened. That evaluation must concern the business target, **weekly sales**, even if our analysis works with sales changes.

A defensible statement is: **“First differences appear more stable, and their strongest observed correlation is at lag 1. This guides our next analysis; forecasting performance remains to be tested.”**

**Instructor**: The claim was deliberately constructed for discussion. This question practices interpreting output and checking reasoning, as in the current saved practice materials. It does not promise future exam coverage.

**Instructor**: Further reading: [Stationarity and differencing](https://otexts.com/fpp3/arima.html#stationarity-and-differencing) and [time-series graphics](https://otexts.com/fpp3/graphics.html), Forecasting: Principles and Practice. Next lesson introduces model mechanisms after these correlation tools.

**Instructor**: Continue with [Part II: identifying Box–Jenkins models](?transcript=./examples/box-jenkins-identification-20261119-replay.md). The sequence is foundations, identification, estimation and diagnostics, then seasonal models.
