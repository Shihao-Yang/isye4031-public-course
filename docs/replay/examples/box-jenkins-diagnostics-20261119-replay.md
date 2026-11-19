# Title: Box–Jenkins Part III: estimation and diagnostics

**Instructor**: Continue from model identification with the original paper-towel candidate. Work through estimation, uncertainty, residual checks and forecasts. Checkpoint 3 provides optional second-order detail. Seasonal models follow this lesson.

**Instructor**: Learning goals. Distinguish stationarity and invertibility; interpret estimation and convergence; assess coefficient uncertainty and parsimony; audit residual ACF/PACF and joint tests; identify a failed assumption; and interpret forecasts and intervals in original units.

**Instructor**: Provenance. This replays an actual Codex CLI conversation. Human prompts are simulated instructor questions. Numerical analyses ran in saved scripts. Selected assistant responses retain their recorded wording, with only asset links made portable. Instructor notes are separately labeled. Constructed illustrations are identified. Playback displays saved results. Source: Bowerman, O’Connell and Koehler, Forecasting, Time Series, and Regression, fourth edition, Table 9.1 and Chapter 10, especially Examples 10.1 and 10.4. Newly executed output is not historical SAS or Minitab output.

**Human**: Remind us of the paper-towel data and our tentative model. What still needs checking before we trust its forecasts?

**Instructor**: Before revealing: Which claims follow from tentative identification, and which require new evidence?

<!-- checkpoint: 1. The candidate and remaining checks -->
**Assistant**: The paper-towel data contain **120 consecutive weekly sales observations** from the Olympia Paper Company teaching example. Calendar dates are unspecified. We write the recorded series as \(y_t\), measured in units of 10,000 rolls above a baseline of 100,000 rolls:

\[
\text{sales in rolls}=100{,}000+10{,}000y_t.
\]

Thus, \(y_t=15\) means 250,000 rolls sold. A negative recorded value would mean sales below the baseline and is valid. The forecasting purpose is to support production, storage, and budget planning.

Our identification work led to a **tentative MA(1) model for first differences**. Define

\[
z_t=\Delta y_t=y_t-y_{t-1}.
\]

Using the textbook’s minus-sign convention, the candidate is

\[
z_t=a_t-\theta_1a_{t-1},
\]

where \(a_t\) is a zero-mean white-noise innovation with constant variance \(\sigma_a^2\), and \(\theta_1\) describes how the previous innovation enters this week’s change. We initially omit a constant, so the model assumes zero mean weekly change. In standard notation, \(z_t\) follows ARMA(0,1), and the sales level \(y_t\) follows **ARIMA(0,1,1) without drift**.

That is a candidate model, not yet a verified forecasting model. Before trusting its forecasts, we need to check:

- **Estimation:** Estimate \(\theta_1\) and \(\sigma_a^2\), assess parameter uncertainty, and check that estimation converges reliably. Also reconsider whether omitting the constant is supported.
- **Model assumptions:** Check whether first differencing produces a reasonably stable series. A finite MA(1) is causally stationary; invertibility is a separate condition, requiring \(|\theta_1|<1\).
- **Residual behavior:** After fitting, check for remaining serial correlation, shifts in the residual mean, changing variability, and unusual observations. Residuals should resemble the innovations assumed by the model. Failure to detect correlation would not prove independence.
- **Forecast performance:** Assess forecast errors and interval coverage using a time-ordered evaluation that uses only observations available at each forecast origin.

The identification pattern gave us a reason to try this model. Estimation and diagnostics must now establish whether it adequately describes the data. No model has been fitted or diagnostics run in this recap.

**Instructor**: Start from the paper-towel MA(1) candidate for first differences. Estimation, diagnostics and forecasting have different jobs.

**Human**: What do stationarity and invertibility each require? Show why they are different, including the second-order cases.

**Instructor**: Before revealing: Does a stationary MA process automatically let us recover shocks stably?

<!-- checkpoint: 2. Stationarity and invertibility -->
**Assistant**: **Stationarity concerns the behavior of the observed process over time. Invertibility concerns recovering innovations from present and past observations.** They impose restrictions on different parts of an ARMA model.

Let \(B\) denote the backshift operator, \(Bz_t=z_{t-1}\). Write the centered ARMA model as

\[
\phi(B)(z_t-\mu)=\theta(B)a_t,
\]

where \(a_t\) is white noise with mean zero and finite variance \(\sigma_a^2\), and

\[
\phi(B)=1-\phi_1B-\cdots-\phi_pB^p,\qquad
\theta(B)=1-\theta_1B-\cdots-\theta_qB^q.
\]

Assume these polynomials have no common factors.

A **weakly stationary** process has constant mean and variance, with covariance depending only on lag. The usual **causal stationary ARMA solution**, built from current and past innovations, requires every root of \(\phi(B)=0\) to lie **outside the unit circle**. Then

\[
z_t-\mu=\frac{\theta(B)}{\phi(B)}a_t
       =\sum_{j=0}^{\infty}\psi_j a_{t-j},
\]

with absolutely summable coefficients. Causality matters here: some explosive AR equations admit stationary solutions involving future innovations, which are not the causal models we use for forecasting.

**Invertibility** requires every root of \(\theta(B)=0\) to lie outside the unit circle. This allows the stable recovery

\[
a_t=\frac{\phi(B)}{\theta(B)}(z_t-\mu)
   =\sum_{j=0}^{\infty}\pi_j(z_{t-j}-\mu).
\]

For first-order models, the distinction is especially clear:

| Model | Causal stationarity | Invertibility |
|---|---|---|
| AR(1): \(z_t=\phi_1z_{t-1}+a_t\) | \(|\phi_1|<1\) | Automatic |
| MA(1): \(z_t=a_t-\theta_1a_{t-1}\) | Any finite \(\theta_1\) | \(|\theta_1|<1\) |

Why is every finite MA(1) stationary? Its population covariances are

\[
\gamma(0)=(1+\theta_1^2)\sigma_a^2,\quad
\gamma(1)=-\theta_1\sigma_a^2,\quad
\gamma(k)=0\quad(|k|>1),
\]

which depend only on lag. But recovering its innovations requires

\[
a_t=z_t+\theta_1z_{t-1}+\theta_1^2z_{t-2}+\cdots,
\]

which is stable only when \(|\theta_1|<1\).

For the **second-order cases**, applying the same polynomial-root conditions gives:

\[
\begin{array}{ll}
\text{AR(2) causal stationarity:}&
\phi_2>-1,\quad \phi_1+\phi_2<1,\quad \phi_2-\phi_1<1,\\[2mm]
\text{MA(2) invertibility:}&
\theta_2>-1,\quad \theta_1+\theta_2<1,\quad \theta_2-\theta_1<1.
\end{array}
\]

Here MA(2) uses \(z_t=a_t-\theta_1a_{t-1}-\theta_2a_{t-2}\). These inequalities put both roots outside the unit circle; checking each coefficient separately is insufficient. Every finite MA(2) remains stationary, including parameter values outside its invertibility region.

For our paper-towel candidate, the **differences** are stationary under the MA(1) assumptions. Invertibility additionally requires \(|\theta_1|<1\). The sales **levels** are modeled as integrated, so their stationarity is a separate issue.

**Instructor**: In the usual causal and invertible representation, check AR roots for stationarity and MA roots for invertibility. The differenced working series and original levels have different stationarity properties.

**Human**: Can you make those restrictions visual? Show stable versus unstable weights and check a few second-order examples.

**Instructor**: Before revealing: Can every coefficient be smaller than one and still produce unstable weights? Can a stable sequence oscillate?

<!-- checkpoint: 3. Optional: second-order restrictions -->
**Assistant**: **Stable weights eventually decay toward zero; unstable weights persist or grow.** The same calculation illustrates causal stationarity for an AR(2) and invertibility for an MA(2), but the weights have different meanings.

Let \(Bz_t=z_{t-1}\), and consider the expansion

\[
\frac{1}{1-c_1B-c_2B^2}=\sum_{j=0}^{\infty}w_jB^j.
\]

Multiplying both sides by the denominator gives the recursion

\[
w_0=1,\qquad w_1=c_1,\qquad
w_j=c_1w_{j-1}+c_2w_{j-2}.
\]

The figure shows these **calculated weights for fixed illustrative parameters**, with no simulation or sales-data fitting. Each panel uses its own vertical scale.

![Stable and unstable second-order expansion weights for four constructed examples](/examples/box-jenkins-diagnostics-20261119-v1/turn03_weights.png)

To interpret the plots, assume \(a_t\) is zero-mean white noise with finite variance.

- For **AR(2)**, \(z_t=c_1z_{t-1}+c_2z_{t-2}+a_t\). The expansion gives \(z_t=\sum_jw_ja_{t-j}\): the weights measure the effects of past innovations. Absolute summability supports the causal stationary solution.
- For **MA(2)**, \(z_t=a_t-c_1a_{t-1}-c_2a_{t-2}\). The expansion gives \(a_t=\sum_jw_jz_{t-j}\): the weights recover innovations from observations. Absolute summability gives invertibility. The finite MA(2) itself is stationary in all four examples.

In both interpretations, the relevant roots solve \(1-c_1B-c_2B^2=0\). Both must have modulus greater than one, equivalently,

\[
c_2>-1,\qquad c_1+c_2<1,\qquad c_2-c_1<1.
\]

The newly executed checks give:

| Example | \((c_1,c_2)\) | Root moduli | Restriction check |
|---|---|---|---|
| A | \((0.6,0.2)\) | \(4.193,\ 1.193\) | All pass |
| B | \((1.2,-0.5)\) | \(1.414,\ 1.414\) | All pass |
| C | \((0.8,0.4)\) | \(2.871,\ 0.871\) | \(c_1+c_2=1.2>1\) fails |
| D | \((0.5,-1.2)\) | \(0.913,\ 0.913\) | \(c_2>-1\) fails |

**B shows that stability does not require every coefficient to be below one in magnitude**, or every successive weight to shrink. Its oscillations eventually decay. **C shows that checking coefficients individually is insufficient**: both are below one, yet the weights grow. D has oscillations with a growing envelope.

Thus, A and B qualify as causal stationary AR(2) models or invertible MA(2) models. C and D fail those respective conditions.

[Python code](/examples/box-jenkins-diagnostics-20261119-v1/turn03_weights.py) · [Numerical results](/examples/box-jenkins-diagnostics-20261119-v1/turn03_weights.json)

**Instructor**: Optional depth. Root conditions apply to the whole polynomial; checking individual coefficients is insufficient.

**Human**: Return to paper towels. How do preliminary estimates become final estimates? Fit the candidate and show the search, convergence, and any difference between fitting methods.

**Instructor**: Before revealing: Why can matching the lag-one correlation give a useful starting point without being the final fit?

<!-- checkpoint: 4. From initial to final estimates -->
**Assistant**: The fitted candidate is **MA(1) for the 119 first differences**, with no constant:

\[
z_t=y_t-y_{t-1}=a_t-\theta_1a_{t-1}.
\]

Here \(a_t\) has mean zero and variance \(\sigma_a^2\). All results below were newly computed from the supplied four-decimal sales values.

A **preliminary estimate** supplies a starting point. For an MA(1), the population lag-one correlation satisfies

\[
\rho(1)=\frac{-\theta_1}{1+\theta_1^2}.
\]

Replacing this population quantity with the noisy sample estimate \(r_1=0.30665\), and choosing the invertible solution, gives

\[
\widetilde\theta_1=-0.34266.
\]

This moment estimate matches one feature of the data. A final estimate optimizes a specified fitting criterion using the full observed sequence.

![Conditional least-squares objective, exact Gaussian profile likelihood, and optimizer paths from four starting values](/examples/box-jenkins-diagnostics-20261119-v1/turn04_search.png)

For **conditional least squares**, we initialize the innovation before the differenced sample at zero and recursively calculate

\[
e_t(\theta_1)=z_t+\theta_1e_{t-1}(\theta_1).
\]

We then minimize \(S(\theta_1)=\sum_te_t(\theta_1)^2\).

For **exact Gaussian maximum likelihood**, we assume independent Gaussian innovations and use the joint distribution of all 119 differences, accounting for the unobserved initial innovation. The middle panel profiles out \(\sigma_a^2\): at each \(\theta_1\), it uses the variance maximizing the likelihood.

| Method | Estimate of \(\theta_1\) | Estimate of \(\sigma_a^2\) |
|---|---:|---:|
| Preliminary moments | −0.34266 | |
| Conditional least squares | −0.35339 | 1.07125 |
| Exact Gaussian likelihood | −0.35184 | 1.07076 |

The conditional variance estimate uses \(S/119\). The small difference between the final methods reflects their different treatment of the beginning of the sample and their different objectives.

**Convergence was checked explicitly.** Conditional minimization succeeded after 16 objective evaluations. Exact-likelihood searches starting at the moment estimate, \(0\), \(-0.7\), and \(0.8\) all reported convergence, taking 6, 6, 8, and 13 iterations. They agreed to five decimal places. An independent covariance-based likelihood calculation confirmed the optimum, with log likelihood \(-172.98775\). No warnings occurred.

Thus, using exact likelihood, our estimated equation is

\[
z_t=a_t+0.35184a_{t-1}.
\]

The plus sign follows from the textbook parameter being negative; the library’s plus-sign MA coefficient is \(+0.35184\). Also, \(|\widehat\theta_1|<1\), so the fit is invertible. **Numerical convergence establishes an estimated model; it does not establish diagnostic adequacy.**

[Saved code](/examples/box-jenkins-diagnostics-20261119-v1/turn04_fit.py) · [Parameters, search traces, and numerical results](/examples/box-jenkins-diagnostics-20261119-v1/turn04_fit.json)

**Instructor**: State the objective and initialization. Successful numerical convergence does not establish that the model describes the data.

**Human**: How certain are those parameter estimates, and does the paper-towel model need a constant? Show how to read the evidence without overclaiming.

**Instructor**: Before revealing: Does a large p-value prove the mean change is zero? Translate its interval into rolls.

<!-- checkpoint: 5. Uncertainty and the constant -->
**Assistant**: **The MA coefficient is distinguishable from zero, while the data provide little evidence for a constant.** That supports retaining our no-constant candidate, subject to diagnostics.

We fitted both models by exact Gaussian likelihood to the 119 first differences:

\[
z_t=\delta+a_t-\theta_1a_{t-1},
\]

where \(z_t=y_t-y_{t-1}\), \(\delta=E(z_t)\) is the mean weekly change, and \(\operatorname{Var}(a_t)=\sigma_a^2\). Setting \(\delta=0\) gives our original candidate. Both fits converged without warnings.

The newly computed uncertainty summaries are:

| Model and parameter | Estimate | Standard error | Approximate 95% interval |
|---|---:|---:|---:|
| No constant: \(\theta_1\) | −0.35184 | 0.08624 | [−0.52088, −0.18280] |
| No constant: \(\sigma_a^2\) | 1.07076 | 0.13882 | [0.79868, 1.34283] |
| With constant: \(\theta_1\) | −0.35184 | 0.08624 | [−0.52088, −0.18280] |
| With constant: \(\delta\) | 0.00552 | 0.12795 | [−0.24527, 0.25631] |

These use the **observed information matrix**, which measures likelihood curvature near the optimum. The intervals are Wald intervals, calculated as estimate \(\pm1.96\) standard errors.

![Approximate parameter intervals with and without a constant](/examples/box-jenkins-diagnostics-20261119-v1/turn05_uncertainty.png)

For \(H_0:\theta_1=0\), the no-constant fit gives

\[
z=\frac{-0.35184}{0.08624}=-4.080,\qquad p\approx0.000045.
\]

This supports including the MA term within the assumed model. It does **not** establish that MA(1) captures all dependence.

For \(H_0:\delta=0\),

\[
z=\frac{0.00552}{0.12795}=0.043,\qquad p\approx0.966.
\]

Allowing a constant barely improves the log likelihood: from \(-172.98775\) to \(-172.98681\). The likelihood-ratio statistic is \(0.00187\), with approximate \(p=0.966\). AIC also favors omitting it: \(349.98\) without a constant versus \(351.97\) with one.

**Non-rejection does not prove zero drift.** The estimated mean change is about 55 rolls per week, but its interval spans approximately −2,453 to +2,563 rolls per week. The data leave meaningful uncertainty about a persistent trend in sales levels.

These are **approximate likelihood z tests, not exact finite-sample t tests**. Their interpretation assumes the model and innovation assumptions are adequate, and the intervals do not account for selecting MA(1) after examining the data. Parameter significance and optimizer convergence therefore leave residual diagnostics as an essential next check.

[Saved code](/examples/box-jenkins-diagnostics-20261119-v1/turn05_uncertainty.py) · [Numerical results](/examples/box-jenkins-diagnostics-20261119-v1/turn05_uncertainty.json)

**Instructor**: Approximate likelihood inference is conditional on the chosen model. Parameter significance alone cannot establish residual adequacy.

**Human**: Would adding an AR term necessarily improve this model? Compare the fits, parameter correlations and uncertainty, and explain the role of parsimony.

**Instructor**: Before revealing: Would a small reduction in training error justify a much larger standard error?

<!-- checkpoint: 6. Parsimony and correlated estimates -->
**Assistant**: **Adding an AR term can improve the optimized in-sample likelihood, but it need not improve forecasting. Here, its fit improvement is negligible and parameter uncertainty increases substantially.**

We compared our no-constant ARIMA(0,1,1) with ARIMA(1,1,1), using exact Gaussian likelihood on the same 119 differences. The larger model is

\[
z_t=\phi_1z_{t-1}+a_t-\theta_1a_{t-1},
\]

where \(z_t=\Delta y_t\), \(\phi_1\) is the added AR coefficient, and \(a_t\) denotes independent Gaussian innovations with variance \(\sigma_a^2\). Setting \(\phi_1=0\) recovers the smaller model, so the maximized likelihood cannot decrease if optimization succeeds.

| Newly computed result | ARIMA(0,1,1) | ARIMA(1,1,1) |
|---|---:|---:|
| Log likelihood | −172.98775 | −172.97959 |
| Innovation variance | 1.07076 | 1.07059 |
| AIC | 349.98 | 351.96 |
| BIC | 355.53 | 360.30 |

Lower AIC and BIC are preferred. Both favor the smaller model after penalizing the additional parameter. The approximate likelihood-ratio test gives \(2\Delta\ell=0.01631\), with one degree of freedom and \(p=0.898\): little evidence that the AR term is needed.

The uncertainty comparison explains the cost:

| Parameter | Estimate | Standard error | Approximate 95% interval |
|---|---:|---:|---:|
| Smaller model: \(\theta_1\) | −0.35184 | 0.08624 | [−0.52088, −0.18280] |
| Larger model: \(\theta_1\) | −0.33234 | 0.24607 | [−0.81463, 0.14996] |
| Larger model: \(\phi_1\) | 0.02500 | 0.25989 | [−0.48439, 0.53438] |

![Parameter uncertainty and the strong correlation between AR and MA estimates](/examples/box-jenkins-diagnostics-20261119-v1/turn06_parsimony.png)

In the larger model, the estimated sampling correlation between \(\widehat\phi_1\) and \(\widehat\theta_1\) is **+0.936**. This is correlation between parameter estimators, not between observations. The elongated joint ellipse shows how the coefficients can move together while producing similar fits. Indeed,

\[
\frac{1-\theta_1B}{1-\phi_1B}
=1+(\phi_1-\theta_1)B+\cdots,
\]

so their effects can partly compensate. Allowing this tradeoff nearly triples the standard error of \(\widehat\theta_1\).

One default-start run issued a **failure-to-converge warning**. Three additional starts converged to essentially identical estimates and likelihoods; those converged results are reported here.

**Parsimony favors retaining the MA(1) candidate:** the extra flexibility buys almost no fit improvement while weakening parameter precision. This does not prove the simpler model forecasts better. These are approximate, model-dependent likelihood intervals and tests, not exact finite-sample t tests; residual adequacy and forecast performance remain unchecked.

[Code](/examples/box-jenkins-diagnostics-20261119-v1/turn06_parsimony.py) · [Numerical results and convergence records](/examples/box-jenkins-diagnostics-20261119-v1/turn06_parsimony.json)

**Instructor**: Parameter-estimate correlation differs from correlation between lagged predictors in an AR regression. Both can make individual coefficients hard to estimate. Residual SD measures unexplained variation; coefficient SE measures estimate uncertainty.

**Human**: Now check the smaller model’s residuals. Show the residual ACF and PACF, and look for changes in spread, unusual observations and departures from normality.

**Instructor**: Before revealing: What structure would be evidence that the model has left something unexplained?

<!-- checkpoint: 7. Residual plots -->
**Assistant**: **The residuals show little systematic linear dependence, but week 68 is unusual and the middle of the series has greater spread.** Normality looks broadly plausible, with a noticeable upper-tail departure.

I refitted the no-constant ARIMA(0,1,1), obtaining \(\widehat\theta_1=-0.35184\). The fit converged without warnings. These newly executed diagnostics use all **119 standardized one-step prediction errors**:

\[
r_t=\frac{z_t-\widehat z_{t\mid t-1}}{\sqrt{F_t}},
\]

where \(F_t\) is the model-implied prediction-error variance. The first error accounts for stationary initialization. Parameters were estimated using the full sample, so these are fitted diagnostics, not an out-of-sample forecasting evaluation.

![Paper-towel residuals: time plot, rolling spread, ACF, PACF, histogram and normal Q–Q plot](/examples/box-jenkins-diagnostics-20261119-v1/turn07_residuals.png)

**Dependence.** The ACF measures correlation between errors separated by a given lag; the PACF measures their linear association after accounting for intermediate lags. Neither shows a clear decay pattern or repeated structure. At lag 18, the empirical ACF is \(0.197\), slightly outside the plotted reference band of approximately \(\pm0.180\). All displayed PACF values fall inside.

These bands use \(\pm1.96/\sqrt{119}\). They are rough pointwise white-noise references, not simultaneous limits or exact fitted-residual confidence bands. With 24 lags inspected, one isolated exceedance is insufficient evidence for another model term. Conversely, small residual correlations do not prove independence.

**Spread.** The trailing 20-error standard deviation rises around the middle and declines later. Dividing the sample into consecutive thirds gives:

| Weeks | Standard deviation of standardized errors |
|---|---:|
| 2–41 | 0.983 |
| 42–81 | 1.177 |
| 82–120 | 0.836 |

This suggests a period of greater variability, partly influenced by the large error at week 68. Overlapping rolling windows can exaggerate the appearance of a sustained episode; the plot alone does not establish a variance change.

**Unusual observations.** Week 68 has the largest standardized error, \(+3.52\), corresponding to sales about **36,442 rolls above the fitted one-step prediction**. Week 59 is next at \(-2.46\). Week 68 merits checking against the original record or business context; its size alone does not justify deleting it.

**Normality.** Most Q–Q points follow the reference line, with week 68 standing above the upper tail. Skewness is \(0.265\), and excess kurtosis is \(0.530\). Jarque–Bera gives \(p=0.302\); Shapiro–Wilk gives \(p=0.658\). These approximate diagnostic results provide no strong evidence against normality, but cannot verify Gaussian innovations, especially in the tails that affect prediction intervals.

[Code](/examples/box-jenkins-diagnostics-20261119-v1/turn07_residuals.py) · [Numerical results](/examples/box-jenkins-diagnostics-20261119-v1/turn07_residuals.json)

**Instructor**: Inspect residual ACF and PACF, time order, spread and tails. A single unusual observation is a reason to investigate, not automatically to remove it.

**Human**: How do Box–Pierce and Ljung–Box combine the residual evidence? Work through one calculation and show how the statistic, degrees of freedom and p-value fit together.

**Instructor**: Before revealing: Why combine correlations over several lags? Which fitted parameters affect the reference degrees of freedom?

<!-- checkpoint: 8. Box–Pierce and Ljung–Box -->
**Assistant**: **Box–Pierce and Ljung–Box combine several residual autocorrelations into one test of remaining linear dependence.** Squaring the correlations prevents positive and negative values from cancelling.

Let \(\widehat r_k\) be the empirical residual autocorrelation at lag \(k\), \(n=119\) the number of residuals, and \(h\) the chosen maximum lag. The statistics are

\[
Q_{\mathrm{BP}}=n\sum_{k=1}^{h}\widehat r_k^2,
\qquad
Q_{\mathrm{LB}}=n(n+2)\sum_{k=1}^{h}
\frac{\widehat r_k^2}{n-k}.
\]

Ljung–Box applies a finite-sample correction, giving each squared correlation a slightly larger weight, especially at higher lags. Its chi-square calibration is still approximate.

Under an adequate fitted ARMA model, the usual reference distribution is

\[
Q\ \overset{\text{approx.}}{\sim}\ \chi^2_{h-p-q}.
\]

Our ARIMA(0,1,1) has \(p=0,q=1\), so **degrees of freedom \(=h-1\)**. This adjustment accounts for estimating the MA coefficient. We do not subtract the differencing order or the innovation variance parameter.

For a worked example, choose **\(h=12\)**. Using the unchanged standardized residuals from our previous check,

\[
\sum_{k=1}^{12}\widehat r_k^2=0.08092444,
\]

so

\[
Q_{\mathrm{BP}}=119(0.08092444)=9.6300.
\]

For Ljung–Box, lag 4 alone contributes

\[
\frac{119(121)}{119-4}(0.128149)^2=2.0562.
\]

Summing all 12 contributions gives \(Q_{\mathrm{LB}}=10.3642\). With \(12-1=11\) degrees of freedom,

\[
p=P\!\left(\chi^2_{11}\ge10.3642\right)=0.4980.
\]

The 5% rejection cutoff is \(19.675\), well above the observed statistic.

![Contributions to the two statistics and the chi-square reference distribution for the worked example](/examples/box-jenkins-diagnostics-20261119-v1/turn08_portmanteau.png)

The newly calculated results at several illustrative cutoffs are:

| Maximum lag \(h\) | df | Box–Pierce \(Q\) | \(p\) | Ljung–Box \(Q\) | \(p\) |
|---|---:|---:|---:|---:|---:|
| 6 | 5 | 3.883 | 0.566 | 4.080 | 0.538 |
| 12 | 11 | 9.630 | 0.564 | 10.364 | 0.498 |
| 18 | 17 | 15.665 | 0.548 | 17.555 | 0.417 |
| 24 | 23 | 16.556 | 0.830 | 18.663 | 0.721 |

These overlapping tests are sensitivity checks, not independent confirmations. Increasing \(h\) increases both the statistic and degrees of freedom, so the p-value need not decrease.

We find **no strong aggregate evidence of residual autocorrelation at these cutoffs**. This does not prove independence or model correctness. Neither test resolves the unusual week 68 observation, possible variation in spread, or normality. A p-value is a reference tail probability under the null assumptions, not the probability that the model is correct.

[Code](/examples/box-jenkins-diagnostics-20261119-v1/turn08_portmanteau.py) · [Full calculation](/examples/box-jenkins-diagnostics-20261119-v1/turn08_portmanteau.json)

**Instructor**: Use the reported residual sample length and parameter-count convention. Approximate non-rejection is limited evidence about the selected correlations, not proof of independence or future performance.

**Human**: What would a failed diagnostic look like here? Deliberately omit the MA term and compare the residual evidence with our candidate.

**Instructor**: Before revealing: If we deliberately omit the MA term, what should remain in the residuals?

<!-- checkpoint: 9. A model that misses dependence -->
**Assistant**: **Omitting the MA term leaves detectable lag-one dependence in the residuals.** This is a concrete diagnostic failure: the model has left useful linear structure unexplained.

The deliberately reduced model is the no-constant **ARIMA(0,1,0)**:

\[
z_t=y_t-y_{t-1}=a_t.
\]

It assumes weekly changes are white noise, so its prediction errors are simply the observed differences. Our candidate instead uses

\[
z_t=a_t-\theta_1a_{t-1},
\qquad \widehat\theta_1=-0.35184.
\]

I fitted the reduced model to the same 119 differences; it converged without warnings. The comparison below uses standardized one-step prediction errors for both models, retaining every observation.

![Residual ACF and PACF after omitting the MA term, compared with the fitted MA(1) candidate](/examples/box-jenkins-diagnostics-20261119-v1/turn09_omission.png)

The reduced model’s **lag-one residual ACF is \(0.30665\)**, clearly beyond the rough pointwise reference band of \(\pm0.17967\). Its lag-one PACF is also \(0.30665\). After fitting the MA term, both fall to **\(0.00724\)**.

This empirical improvement is consistent with the MA(1) interpretation: with our minus-sign convention, a negative \(\theta_1\) produces positive population lag-one correlation. The plotted sample correlations remain noisy estimates, not theoretical population curves.

The aggregate evidence agrees. Using

\[
Q_{\mathrm{LB}}
=n(n+2)\sum_{k=1}^{h}\frac{\widehat r_k^2}{n-k},
\]

we obtain:

| Maximum lag | Omit MA: \(Q\), df, \(p\) | Candidate: \(Q\), df, \(p\) |
|---|---|---|
| 6 | 14.958, 6, 0.021 | 4.080, 5, 0.538 |
| 12 | 25.273, 12, 0.014 | 10.364, 11, 0.498 |
| 18 | 34.953, 18, 0.010 | 17.555, 17, 0.417 |
| 24 | 37.217, 24, 0.042 | 18.663, 23, 0.721 |

The reduced model uses \(h\) degrees of freedom because it estimates no AR or MA coefficients. The candidate uses \(h-1\). These approximate chi-square tests reject the reduced model at 5% at all four displayed cutoffs. They are overlapping sensitivity checks, not four independent discoveries. Box–Pierce likewise rejects at lags 6, 12 and 18, though not at 24 (\(p=0.087\)).

**The failure is remaining autocorrelation, not failure to converge.** The MA term substantially removes that dependence. Non-rejection for the candidate does not prove independence, and it does not resolve the previously observed week 68 error or possible variation in spread. The evidence supports retaining the MA term without establishing that every forecasting assumption holds.

[Code](/examples/box-jenkins-diagnostics-20261119-v1/turn09_omission.py) · [Numerical comparison](/examples/box-jenkins-diagnostics-20261119-v1/turn09_omission.json)

**Instructor**: This deliberately underspecified candidate demonstrates the revise-and-check loop using the same observations. It is not an invented data error.

**Human**: With those diagnostic limits in mind, forecast future paper-towel sales and give prediction intervals. Explain why uncertainty changes with the horizon.

**Instructor**: Before revealing: Why can a point forecast remain flat while its prediction interval widens?

<!-- checkpoint: 10. Forecasts and uncertainty -->
**Assistant**: **The model forecasts about 258,873 rolls per week for the next 12 weeks, with widening prediction intervals.** A flat point forecast does not mean future sales are expected to be constant.

These newly computed forecasts use all 120 observations and the no-drift ARIMA(0,1,1):

\[
z_t=\Delta y_t=a_t-\theta_1a_{t-1},
\qquad
\widehat\theta_1=-0.35184,\quad
\widehat\sigma_a^2=1.07076.
\]

The refit converged without warnings. Week 120 sales were 256,453 rolls.

![Paper-towel sales forecasts with 80% and 95% pointwise prediction intervals, and forecast-error standard deviation by horizon](/examples/box-jenkins-diagnostics-20261119-v1/turn10_forecast.png)

Selected forecasts follow, rounded to the nearest 100 rolls. The saved results contain every horizon through week 132.

| Future week | Horizon | Forecast rolls | 80% prediction interval | 95% prediction interval |
|---|---:|---:|---:|---:|
| 121 | 1 | 258,900 | [245,600, 272,100] | [238,600, 279,200] |
| 122 | 2 | 258,900 | [236,600, 281,200] | [224,800, 293,000] |
| 124 | 4 | 258,900 | [225,100, 292,600] | [207,200, 310,500] |
| 128 | 8 | 258,900 | [209,600, 308,100] | [183,600, 334,200] |
| 132 | 12 | 258,900 | [198,000, 319,800] | [165,700, 352,000] |

**Why is the point forecast flat?** The last estimated innovation is \(\widehat a_{120}=0.68779\), in the recorded units. Therefore,

\[
\widehat z_{121\mid120}
=-\widehat\theta_1\widehat a_{120}
=0.24200.
\]

That adds about 2,420 rolls to the last sales level. Beyond this first step, future changes have forecast mean zero, so every subsequent level forecast stays at that adjusted value.

**Why do intervals widen?** Future changes contain new innovations, and these accumulate in the sales level. Treating parameters and the last innovation as known, the horizon-\(h\) level error is

\[
y_{T+h}-\widehat y_{T+h\mid T}
=a_{T+h}+(1-\theta_1)\sum_{j=1}^{h-1}a_{T+j}.
\]

Hence,

\[
\operatorname{Var}(y_{T+h}-\widehat y_{T+h\mid T})
=\sigma_a^2\left[1+(h-1)(1-\theta_1)^2\right].
\]

Variance grows linearly with horizon; interval width grows with its square root. The calculation includes filtered-state uncertainty and agrees with this formula to numerical precision.

The intervals assume independent Gaussian innovations, constant variance, and fixed fitted parameters. They exclude parameter and model-selection uncertainty or future structural changes. Our unusual week 68 error and varying residual spread therefore limit confidence in their coverage. These are **pointwise intervals for individual weeks**, not simultaneous coverage for the entire future path, and no future observations are available to validate them.

[Code](/examples/box-jenkins-diagnostics-20261119-v1/turn10_forecast.py) · [All forecasts and numerical results](/examples/box-jenkins-diagnostics-20261119-v1/turn10_forecast.json)

**Instructor**: Return from first differences to the sales level, then use actual rolls = 100000 + 10000 times the recorded sales scale. Intervals must identify what uncertainty is included.

**Human**: A made-up AI report says the large residual-test p-value proves independence, and the larger model must forecast better because it fits better. What would you correct, and what should we do next?

**Instructor**: Before revealing: Do nonsignificant residual tests and lower training error settle the forecasting decision?

<!-- checkpoint: 11. Checking an AI conclusion -->
**Assistant**: I would correct both claims.

**A large residual-test p-value does not prove independence.** For our ARIMA(0,1,1), the Ljung–Box statistic through lag 12 was \(Q=10.364\), compared with an approximate chi-square distribution with \(12-1=11\) degrees of freedom, giving \(p=0.498\).

That means the observed aggregate residual autocorrelation is not unusually large under the test’s null assumptions. It does not mean there is a 49.8% probability that the residuals are independent. The test examines linear autocorrelation through selected lags and can miss dependence, including dependence in variability. Its non-rejection also does not resolve the unusually large error at week 68 or the greater residual spread in the middle of the series.

**A better in-sample fit does not guarantee better forecasts.** ARIMA(1,1,1) contains ARIMA(0,1,1) as the special case with AR coefficient \(\phi_1=0\). Its optimized likelihood can therefore increase simply because it has another adjustable parameter.

Here, the increase was tiny: log likelihood rose from \(-172.98775\) to \(-172.97959\). Meanwhile:

- AIC increased from \(349.98\) to \(351.96\), favoring the smaller model.
- The added AR estimate was \(0.0250\), with standard error \(0.2599\).
- The estimated AR and MA coefficients had sampling correlation \(0.936\), and the MA standard error increased from \(0.0862\) to \(0.2461\).

Those results suggest extra flexibility with little supporting information. They do not establish which model forecasts better.

**Next, retain the smaller model provisionally and evaluate the unresolved issues.** Check week 68 against the original records or business context without automatically removing it. Investigate whether the apparent changes in spread persist beyond isolated large errors.

Then compare forecast accuracy using a time-ordered, rolling-origin evaluation: fit each model using only observations available at each origin, forecast relevant horizons, and compare errors and prediction-interval coverage. Include a simple benchmark such as the no-drift random walk.

Because these 120 observations already informed model selection, that evaluation would provide useful retrospective evidence, not a fully untouched validation. Newly arriving sales would provide the clearest prospective check. The appropriate conclusion is **“a reasonable candidate with stated limits,” not “a verified model.”**

**Instructor**: The constructed claim checks independent reasoning. Normality, stable spread, outliers, changing relationships and time-ordered forecast evidence still matter. Proceed to seasonal models after these checks.

**Instructor**: Continue to [Part IV: seasonal ARIMA](?transcript=./examples/seasonal-arima-20261119-replay.md). Further reading: [Residual diagnostics, Forecasting: Principles and Practice](https://otexts.com/fpp3/toolbox.html#residual-diagnostics).
