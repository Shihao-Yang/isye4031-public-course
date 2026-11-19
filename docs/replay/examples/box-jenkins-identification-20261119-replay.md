# Title: Box–Jenkins Part II: theory and model identification

**Instructor**: Continue from foundations. Compare model mechanisms and theoretical ACF/PACF with empirical evidence, using the original paper-towel story. Estimation and diagnostic checking continue in Part III.

**Instructor**: Learning goals. Explain MA, AR and mixed-model signatures; compare theory with sample evidence; propose a tentative candidate; trace residuals and forecasts; distinguish a constant from a series mean; and recognize ambiguous identification.

**Instructor**: Provenance. This replays an actual Codex CLI conversation. Human prompts are simulated instructor questions. Numerical analyses ran in saved scripts. Selected assistant responses retain their recorded wording, with only asset links made portable. Instructor notes are separately labeled. Constructed illustrations are identified. Playback displays saved results. Source: Bowerman, O’Connell and Koehler, Forecasting, Time Series, and Regression, fourth edition, Table 9.1 and Chapter 9, especially Example 9.6. Newly executed figures and numerical results accompany the original example. This classroom edition omits one trial response that introduced later concepts early; a fresh response replaces it. The full recording is retained privately.

**Human**: Remind us of the paper-towel forecasting problem. Show the sales and the working series we will use to identify a model.

**Instructor**: Before revealing: What do the recorded sales units mean? What changes when we difference them?

<!-- checkpoint: 1. The working series -->
**Assistant**: The **paper-towel forecasting problem** concerns the Olympia Paper Company’s weekly sales. We have 120 consecutive weekly observations and want to forecast future sales to help plan production, storage, and budgets. Calendar dates and future observed sales are not supplied.

Let \(y_t\) denote the recorded sales value for week \(t\). Its units are **10,000 rolls above a baseline of 100,000 rolls**, so

\[
\text{actual sales in week }t=100{,}000+10{,}000y_t.
\]

For example, \(y_1=15\) means sales of 250,000 rolls. A negative \(y_t\) means sales below the baseline; it does not mean negative sales.

![Observed paper-towel sales and the first-differenced working series](/examples/box-jenkins-identification-20261119-v1/turn01_sales_working_series.png)

The upper panel shows the observed sales. Their level moves through long rises and falls, making a constant mean questionable. Subtracting the 100,000-roll baseline only changes the vertical scale; it does not remove those movements.

For identification, we will use the **first-differenced working series**

\[
z_t=\Delta y_t=y_t-y_{t-1},\qquad t=2,\ldots,120.
\]

Thus, \(z_t\) measures the change in sales from the preceding week, in units of 10,000 rolls. For example,

\[
z_2=14.4064-15.0000=-0.5936,
\]

which represents a decrease of 5,936 rolls. Differencing leaves us with 119 observations because the first week has no preceding observation in the supplied data.

The lower panel shows that these changes fluctuate around approximately zero, with much less persistent movement in their level. This makes first differencing a reasonable starting point. We will provisionally treat \(z_t\) as having a stable mean, variance, and dependence structure over time. The plot supports that working assumption, but does not establish stationarity or identify a particular model. Those remain questions for the identification sequence.

These plots were computed from the supplied four-decimal data. [Python code](/examples/box-jenkins-identification-20261119-v1/turn01_sales_working_series.py) · [Numerical results](/examples/box-jenkins-identification-20261119-v1/turn01_sales_working_series.json)

**Instructor**: Keep the sales level and weekly change separate. Identification concerns the working series.

**Human**: Start with zero-mean AR(1) and MA(1). How does one surprise move through each series?

**Instructor**: Before revealing: Does the previous observation enter the equation, or the previous surprise? Predict how long one surprise lasts.

<!-- checkpoint: 2. AR and MA mechanisms -->
**Assistant**: Let \(z_t\) be our working series, the weekly change in paper-towel sales. Let \(a_t\) be a new surprise, or *innovation*. Assume the innovations are independent, with mean zero and common finite variance.

The **zero-mean AR(1)** equation is

\[
z_t=\phi_1z_{t-1}+a_t.
\]

Today’s value inherits a fraction \(\phi_1\) of yesterday’s value, then receives today’s surprise. For a causal stationary series, assume \(|\phi_1|<1\).

The **zero-mean MA(1)** equation, using the textbook’s minus-sign convention, is

\[
z_t=a_t-\theta_1a_{t-1}.
\]

Today’s value combines today’s surprise and the preceding surprise. It does not feed the preceding value \(z_{t-1}\) back into the equation.

To see the difference, introduce **one surprise of \(+1\)** at week \(t\), with all other surprises zero and the series initially zero. Use the **fixed illustrative values** \(\phi_1=0.7\) and \(\theta_1=0.7\).

![One surprise moving through constructed zero-mean AR(1) and MA(1) series](/examples/box-jenkins-identification-20261119-v1/turn03_zero_mean_surprise.png)

For the AR(1), the surprise becomes part of the value carried into the next week:

\[
\begin{aligned}
z_t&=1,\\
z_{t+1}&=0.7(1)=0.7,\\
z_{t+2}&=0.7(0.7)=0.49.
\end{aligned}
\]

At week \(t+h\), its effect is \(\phi_1^h\). The effect continues indefinitely, shrinking toward zero. With a negative \(\phi_1\), it would alternate in sign while shrinking.

For the MA(1), the surprise appears directly in only two weeks:

\[
\begin{aligned}
z_t&=1,\\
z_{t+1}&=-0.7(1)=-0.7,\\
z_{t+2}&=0.
\end{aligned}
\]

After one week, that surprise leaves the equation entirely. The negative second effect comes from our minus-sign convention and positive \(\theta_1\).

These figures show exact, constructed responses. In an observed series, new surprises arrive every week and their effects overlap. Also, the plotted effects concern **weekly sales changes \(z_t\)**, not the sales level \(y_t\).

[Code](/examples/box-jenkins-identification-20261119-v1/turn03_zero_mean_surprise.py) · [Numerical results](/examples/box-jenkins-identification-20261119-v1/turn03_zero_mean_surprise.json)

**Instructor**: The shock-response plot explains the mechanism. It is not an ACF plot; the next steps derive correlations.

**Human**: What theoretical ACF and PACF does MA(1) imply? Show the population curves beside what we might see in a finite sample.

**Instructor**: Before revealing: Which observations share a surprise? Would an exact population zero appear as an exact sample zero?

<!-- checkpoint: 3. MA(1): theory and sample -->
**Assistant**: For the **zero-mean MA(1)**,

\[
z_t=a_t-\theta_1a_{t-1},
\]

assume the innovations \(a_t\) are independent, with mean zero and variance \(\sigma_a^2\). Its identifying population pattern is: **the ACF cuts off after lag 1, while the PACF tails off.**

To see why, write the two adjacent values:

\[
z_t=a_t-\theta_1a_{t-1},\qquad
z_{t-1}=a_{t-1}-\theta_1a_{t-2}.
\]

They share exactly one innovation, \(a_{t-1}\), whose coefficients are \(-\theta_1\) and \(1\). Thus,

\[
\operatorname{Var}(z_t)=(1+\theta_1^2)\sigma_a^2,\qquad
\operatorname{Cov}(z_t,z_{t-1})=-\theta_1\sigma_a^2.
\]

Values separated by two or more weeks share no innovations. Consequently, the population ACF, denoted \(\rho_k=\operatorname{Corr}(z_t,z_{t-k})\), is

\[
\boxed{\rho_1=\frac{-\theta_1}{1+\theta_1^2},
\qquad \rho_k=0\quad(k\ge2).}
\]

The **PACF at lag \(k\)** measures the remaining linear relationship between \(z_t\) and \(z_{t-k}\) after accounting for the intervening values. A zero ordinary correlation does not require a zero partial correlation. For example,

\[
\operatorname{PACF}(2)
=\frac{\rho_2-\rho_1^2}{1-\rho_1^2}
=-\frac{\rho_1^2}{1-\rho_1^2}.
\]

So even though \(\rho_2=0\), the lag-2 PACF is negative when \(\theta_1\ne0\).

![Population and finite-sample ACF and PACF for an illustrative MA(1)](/examples/box-jenkins-identification-20261119-v1/turn04_ma1_correlations.png)

The figure uses the **fixed illustrative coefficient \(\theta_1=0.7\)**. The population ACF has just one nonzero positive-lag value, \(\rho_1\approx-0.470\). The population PACF begins

\[
-0.470,\;-0.283,\;-0.186,\;-0.126,\ldots
\]

and gradually approaches zero. With this positive coefficient and our minus-sign convention, the tail is negative.

The right panels show **one newly simulated sample of 119 observations**, matching our working-series length, with independent normal innovations and a fixed seed. These are constructed data, not paper-towel sales. The sample ACF beyond lag 1 fluctuates around zero, and the sample PACF only roughly resembles its population tail.

The dashed lines are the familiar approximate white-noise reference limits, \(\pm1.96/\sqrt{119}\approx\pm0.180\). They are pointwise guides, not exact MA(1) confidence bands or simultaneous bounds.

For identification, look for an ACF broadly consistent with a lag-1 cutoff and a PACF that fades. A finite sample will not reproduce the population curves exactly, and this pattern supplies a tentative candidate, not a verified model.

[Code](/examples/box-jenkins-identification-20261119-v1/turn04_ma1_correlations.py) · [Numerical results](/examples/box-jenkins-identification-20261119-v1/turn04_ma1_correlations.json)

**Instructor**: Population ACF and PACF are model properties. Sample ACF and PACF estimate them. These two pairs must be compared together.

**Human**: Now do the same for AR(1). Why does its ACF decay while its PACF cuts off?

**Instructor**: Before revealing: After accounting for the previous value, can an older value improve the linear prediction in this AR(1)?

<!-- checkpoint: 4. AR(1): theory and sample -->
**Assistant**: For the **zero-mean AR(1)**,

\[
z_t=\phi_1z_{t-1}+a_t,
\]

assume independent innovations \(a_t\), with mean zero and common finite variance, and \(|\phi_1|<1\) for causal stationarity. Its population pattern is: **the ACF decays geometrically, while the PACF cuts off after lag 1.**

Why does the ACF decay? Let \(\gamma_k=\operatorname{Cov}(z_t,z_{t-k})\). For \(k\ge1\), today’s innovation is independent of past values, so

\[
\begin{aligned}
\gamma_k
&=\operatorname{Cov}(\phi_1z_{t-1}+a_t,\ z_{t-k})\\
&=\phi_1\gamma_{k-1}.
\end{aligned}
\]

Dividing by the stationary variance \(\gamma_0\), and starting from \(\rho_0=1\), gives

\[
\boxed{\rho_k=\phi_1^k.}
\]

Dependence reaches distant weeks through repeated transmission: \(z_{t-2}\) influences \(z_{t-1}\), which influences \(z_t\). Each step multiplies the correlation by \(\phi_1\). A positive coefficient gives a same-sign decay; a negative coefficient gives an alternating decay.

Why does the PACF cut off? The PACF removes the linear contribution of intervening values. Once \(z_{t-1}\) is included, the equation leaves only \(a_t\) unexplained. Older values provide no additional linear predictive information. Thus,

\[
\boxed{\operatorname{PACF}(1)=\phi_1,\qquad
\operatorname{PACF}(k)=0\quad(k\ge2).}
\]

At lag 2, we can check this directly:

\[
\operatorname{PACF}(2)
=\frac{\rho_2-\rho_1^2}{1-\rho_1^2}
=\frac{\phi_1^2-\phi_1^2}{1-\phi_1^2}=0.
\]

![Population and finite-sample ACF and PACF for an illustrative AR(1)](/examples/box-jenkins-identification-20261119-v1/turn05_ar1_correlations.png)

The left panels use the **fixed illustrative coefficient \(\phi_1=0.7\)**. The population ACF begins \(0.7,\,0.49,\,0.343,\ldots\), while the PACF is exactly zero beyond lag 1.

The right panels show **one newly simulated sample of 119 observations**, after a 2,000-observation burn-in, using independent standard-normal innovations and a fixed seed. These are constructed data, not paper-towel observations. The sample ACF falls faster than its population counterpart here, and the sample PACF has small nonzero values beyond lag 1.

The dashed limits, approximately \(\pm0.180\), are pointwise white-noise reference guides, not exact AR(1) confidence bands or simultaneous bounds. We seek an overall pattern consistent with decay and cutoff; empirical estimates will not reproduce exact population zeros. Such a pattern supports tentative identification.

[Code](/examples/box-jenkins-identification-20261119-v1/turn05_ar1_correlations.py) · [Numerical results](/examples/box-jenkins-identification-20261119-v1/turn05_ar1_correlations.json)

**Instructor**: AR ACF decay and PACF cutoff reflect different questions about dependence.

**Human**: Return to the paper-towel differences. Compare their sample ACF and PACF with the theoretical patterns. Which model would you try first, and why is the choice tentative?

**Instructor**: Before revealing: Compare both empirical plots with both theoretical signatures. Which candidate would you try, and what is ambiguous?

<!-- checkpoint: 5. Identify the paper-towel model -->
**Assistant**: I would try an **MA(1) for the first differences** first. The main clue is a positive lag-1 ACF followed by mostly small correlations, suggesting a cutoff after lag 1.

Our working series is

\[
z_t=y_t-y_{t-1},
\]

with 119 observed weekly differences. The figure below was newly computed from the supplied paper-towel data.

![Sample ACF and PACF of the paper-towel first differences](/examples/box-jenkins-identification-20261119-v1/turn06_paper_towel_identification.png)

Compare the observed shapes with the theoretical population patterns:

| Model for \(z_t\) | Population ACF | Population PACF |
|---|---|---|
| AR(1) | Geometric decay | Zero after lag 1 |
| MA(1) | Zero after lag 1 | Tails off |

The sample ACF begins

\[
r_1=0.307,\qquad r_2=-0.065,\qquad r_3=-0.072.
\]

There is a clear positive lag-1 correlation, followed immediately by small values. That looks more like an MA(1) cutoff than a smooth geometric decay.

The sample PACF begins

\[
0.307,\qquad -0.175,\qquad 0.006,\qquad 0.133.
\]

This is less decisive. Its negative lag-2 value is compatible with the beginning of an MA(1) PACF tail, but the subsequent values do not trace a clean theoretical curve.

Using our textbook convention, the tentative candidate is

\[
\boxed{z_t=a_t-\theta_1a_{t-1}.}
\]

Because an MA(1) has \(\rho_1=-\theta_1/(1+\theta_1^2)\), the positive observed lag-1 correlation points toward a **negative** \(\theta_1\). Our earlier positive-coefficient illustration therefore had the opposite lag-1 sign.

The choice remains tentative for three reasons. First, these are noisy sample estimates, not population correlations. Second, an AR(1) with a modest positive coefficient can decay quickly enough to resemble a cutoff in a short sample. Third, the ACF has an isolated lag-18 value of about \(0.215\), which we should acknowledge without letting one spike dictate a more complicated model.

The dashed limits are approximate pointwise white-noise guides, \(\pm0.180\). Crossing them is not a model-selection rule, and staying inside them does not prove independence. MA(1) is a reasonable first candidate under our provisional stationarity assumption, not an established description of the differences.

[Code](/examples/box-jenkins-identification-20261119-v1/turn06_paper_towel_identification.py) · [Numerical results](/examples/box-jenkins-identification-20261119-v1/turn06_paper_towel_identification.json)

**Instructor**: The paper-towel example is deliberately tentative. A visually plausible pattern is the beginning of the Box–Jenkins method.

**Human**: Fit that MA(1) candidate by conditional least squares. Walk through the first few predictions and residuals so we can see what is being minimized.

**Instructor**: Before revealing: Why must changing the MA coefficient change every later reconstructed residual?

<!-- checkpoint: 6. Fit and reconstruct residuals -->
**Assistant**: Conditional least squares gives **\(\widehat\theta_1=-0.35339\)** for our zero-mean MA(1) candidate:

\[
z_t=a_t-\theta_1a_{t-1},
\qquad z_t=y_t-y_{t-1}.
\]

Assume the innovations have mean zero, common finite variance, and are independent. Because the innovations are unobserved, we reconstruct residuals recursively for each trial coefficient.

The first available difference is \(z_2\). **Condition on \(e_1=0\)**, setting the preceding unobserved innovation to zero. Then calculate

\[
\widehat z_{t\mid t-1}=-\theta_1e_{t-1},
\qquad
e_t=z_t-\widehat z_{t\mid t-1}
=z_t+\theta_1e_{t-1}.
\]

Here \(\widehat z_{t\mid t-1}\) is the one-step prediction of the working series, and \(e_t\) is its prediction error.

At the fitted coefficient, the prediction becomes \(0.35339e_{t-1}\):

| Week \(t\) | Observed \(z_t\) | Prediction | Residual \(e_t\) | \(e_t^2\) |
|---|---:|---:|---:|---:|
| 2 | −0.59360 | 0.00000 | −0.59360 | 0.35236 |
| 3 | 0.53190 | −0.20977 | 0.74167 | 0.55007 |
| 4 | 1.09910 | 0.26210 | 0.83700 | 0.70057 |
| 5 | −0.40540 | 0.29579 | −0.70119 | 0.49166 |
| 6 | −1.23450 | −0.24779 | −0.98671 | 0.97360 |

For example, week 3 uses the preceding residual:

\[
\widehat z_{3\mid2}=0.35339(-0.59360)=-0.20977,
\]
\[
e_3=0.53190-(-0.20977)=0.74167.
\]

That residual then enters the week-4 prediction. Changing \(\theta_1\) changes the entire sequence of predictions and residuals, so we must rerun the recursion for every trial value.

The quantity minimized is

\[
\boxed{S(\theta_1)=\sum_{t=2}^{120}e_t(\theta_1)^2.}
\]

![Conditional sum of squared residuals for the paper-towel MA(1)](/examples/box-jenkins-identification-20261119-v1/turn07_ma1_conditional_ls.png)

Using all 119 differences, the numerical optimizer converged without warnings, giving

\[
S(\widehat\theta_1)=127.47880.
\]

For comparison, \(\theta_1=0\) predicts every difference as zero and gives \(S(0)=143.84061\).

These are newly computed conditional least-squares results from the supplied rounded data. The table contains **in-sample predictions using a coefficient estimated from the full series**, not held-out forecasting results. The smaller sum of squares establishes the fitted coefficient under this initialization; it does not establish that MA(1) adequately describes the series.

[Code](/examples/box-jenkins-identification-20261119-v1/turn07_ma1_conditional_ls.py) · [Numerical results and all residuals](/examples/box-jenkins-identification-20261119-v1/turn07_ma1_conditional_ls.json)

**Instructor**: The initial unobserved innovation is set to zero for conditional least squares. Historical software can use a different initial treatment. In-sample fitted errors are not forecasts issued in the past.

**Human**: How do we turn this into forecasts of paper-towel sales? Walk through the next two weeks from the end of the observed data.

**Instructor**: Before revealing: What do we know at week 120? Which future innovations have conditional mean zero?

<!-- checkpoint: 7. First forecasts -->
**Assistant**: Our point forecasts are approximately **258,889 rolls for week 121 and 258,889 rolls for week 122**, both made at the end of week 120.

We first forecast the differences, then accumulate them to recover the sales level. Recall

\[
z_t=y_t-y_{t-1},\qquad
\text{sales in rolls}=100{,}000+10{,}000y_t.
\]

At the end of the observed data,

\[
y_{120}=15.6453,\qquad e_{120}=0.689264,
\]

where \(e_{120}\) is the final recursive residual from our conditional least-squares fit. The fitted working-series model is

\[
z_t=a_t+0.353387a_{t-1}.
\]

For forecasting, assume future innovations have conditional mean zero.

**Week 121.** The preceding innovation is represented by the available residual \(e_{120}\), so

\[
\widehat z_{121\mid120}
=0.353387e_{120}
=0.243577.
\]

The notation \(\widehat z_{121\mid120}\) means a forecast for week 121 using information available through week 120. Add this predicted change to the last observed level:

\[
\widehat y_{121\mid120}
=15.6453+0.243577
=15.888877.
\]

Converting to rolls gives

\[
100{,}000+10{,}000(15.888877)
\approx258{,}889.
\]

**Week 122, still forecasting from week 120.** Its working-series equation is

\[
z_{122}=a_{122}+0.353387a_{121}.
\]

Both innovations are future surprises at our forecast origin. Their conditional means are zero, giving

\[
\widehat z_{122\mid120}=0.
\]

Therefore,

\[
\widehat y_{122\mid120}
=y_{120}+\widehat z_{121\mid120}+\widehat z_{122\mid120}
=15.888877.
\]

![Observed paper-towel sales and the next two point forecasts](/examples/box-jenkins-identification-20261119-v1/turn08_sales_forecasts.png)

The predicted **change** becomes zero after the first forecast week, so the predicted **sales level** stays at the newly adjusted level. It does not return to zero or to the 100,000-roll baseline.

Equal point forecasts do not mean the two future sales values will be equal or equally uncertain. These forecasts use no future observations and remain conditional on our tentative model. Once week 121 is observed, its new residual will allow us to update the forecast for week 122.

[Code](/examples/box-jenkins-identification-20261119-v1/turn08_sales_forecasts.py) · [Numerical results](/examples/box-jenkins-identification-20261119-v1/turn08_sales_forecasts.json)

**Instructor**: These forecasts illustrate the recursion. Model adequacy and forecast intervals are the subjects of Part III, before operational use.

**Human**: What changes if the working series has a nonzero mean? Explain the constant in MA and AR models, and what it means for the sales forecasts.

**Instructor**: Before revealing: Does the constant equal the mean in both AR and MA models? What would a nonzero mean change imply for sales levels?

<!-- checkpoint: 8. Constant, mean and drift -->
**Assistant**: A nonzero mean in the working series means a **nonzero average weekly change in sales**. It adds a persistent upward or downward movement to sales forecasts.

Let

\[
z_t=y_t-y_{t-1},\qquad \mu=E(z_t),
\]

and let the innovations \(a_t\) have mean zero. The relationship between the model’s constant \(\delta\) and its mean \(\mu\) depends on the model.

For an **MA(1)**,

\[
z_t=\delta+a_t-\theta_1a_{t-1}.
\]

Taking expectations gives

\[
\boxed{\mu=\delta.}
\]

Equivalently, \(z_t-\mu=a_t-\theta_1a_{t-1}\): the same zero-mean mechanism now operates around \(\mu\).

For an **AR(1)**,

\[
z_t=\delta+\phi_1z_{t-1}+a_t.
\]

Under causal stationarity, \(|\phi_1|<1\), the mean satisfies

\[
\mu=\delta+\phi_1\mu,
\qquad
\boxed{\mu=\frac{\delta}{1-\phi_1}.}
\]

Thus, the AR constant is **not itself the mean**. The centered equation makes its role clearer:

\[
z_t-\mu=\phi_1(z_{t-1}-\mu)+a_t.
\]

In both models, the earlier zero-mean correlation patterns remain unchanged: adding a constant shifts the series without changing its population ACF or PACF.

For sales forecasting, consider the MA(1), using fitted quantities and the final residual \(e_T\). Assuming future innovations have conditional mean zero,

\[
\widehat z_{T+1\mid T}=\mu-\theta_1e_T,
\qquad
\widehat z_{T+h\mid T}=\mu\quad(h\ge2).
\]

Accumulating these changes gives

\[
\boxed{\widehat y_{T+h\mid T}=y_T+h\mu-\theta_1e_T,\qquad h\ge1.}
\]

So the next two sales-level forecasts are

\[
\widehat y_{T+1\mid T}=y_T+\mu-\theta_1e_T,
\]
\[
\widehat y_{T+2\mid T}=y_T+2\mu-\theta_1e_T.
\]

Previously, \(\mu=0\) produced a flat forecast after the initial adjustment. With nonzero \(\mu\), the sales forecast subsequently changes by \(\mu\) each week, or **\(10{,}000\mu\) rolls per week**.

For an AR(1),

\[
\widehat z_{T+h\mid T}
=\mu+\phi_1^h(z_T-\mu).
\]

Its predicted changes approach \(\mu\) gradually, so its sales forecasts also eventually follow that average weekly movement.

To obtain numerical forecasts with a nonzero mean, we would refit the model allowing the constant. We cannot simply add a mean to our previous forecasts: the fitted coefficient and recursive residuals can also change.

**Instructor**: A mean test must account for serial dependence. Dividing the sample SD by the square root of the sample size is not generally enough.

**Human**: How does the identification rule extend to MA(q)? Work through MA(2), including its theoretical and sample ACF and PACF.

**Instructor**: Before revealing: If the same surprise enters three consecutive observations, at what lag can the population ACF still be nonzero?

<!-- checkpoint: 9. MA(q), including MA(2) -->
**Assistant**: For a **zero-mean MA(q)**,

\[
z_t=a_t-\theta_1a_{t-1}-\cdots-\theta_qa_{t-q},
\]

assume independent innovations with mean zero and common finite variance \(\sigma_a^2\). The order \(q\) is the largest innovation lag with a nonzero coefficient.

The identification rule extends directly: **the population ACF cuts off after lag \(q\), while the PACF generally tails off.** Two values separated by more than \(q\) periods share no innovations, so their covariance is zero. This finite combination of innovations is stationary for any fixed finite coefficients.

For **MA(2)**,

\[
z_t=a_t-\theta_1a_{t-1}-\theta_2a_{t-2}.
\]

Its variance is

\[
\gamma_0=(1+\theta_1^2+\theta_2^2)\sigma_a^2.
\]

At lag 1, \(z_t\) and \(z_{t-1}\) share two innovations. Multiplying their corresponding coefficients gives

\[
\gamma_1=[(-\theta_1)(1)+(-\theta_2)(-\theta_1)]\sigma_a^2
=\theta_1(\theta_2-1)\sigma_a^2.
\]

At lag 2, only \(a_{t-2}\) is shared:

\[
\gamma_2=-\theta_2\sigma_a^2.
\]

Beyond lag 2 there is no overlap. Therefore,

\[
\boxed{
\rho_1=\frac{\theta_1(\theta_2-1)}{1+\theta_1^2+\theta_2^2},
\quad
\rho_2=\frac{-\theta_2}{1+\theta_1^2+\theta_2^2},
\quad
\rho_k=0\ (k>2).
}
\]

The PACF removes intervening values, which changes these relationships. For example,

\[
\operatorname{PACF}(2)=\frac{\rho_2-\rho_1^2}{1-\rho_1^2}.
\]

Even when ordinary correlations vanish beyond lag 2, the corresponding partial correlations generally remain nonzero and gradually decay.

![Population and finite-sample correlations for a constructed MA(2)](/examples/box-jenkins-identification-20261119-v1/turn09_ma2_correlations.png)

The figure uses **fixed illustrative values \(\theta_1=0.5,\theta_2=0.3\)**. Its population ACF begins

\[
-0.261,\;-0.224,\;0,\;0,\ldots,
\]

while its population PACF begins

\[
-0.261,\;-0.313,\;-0.189,\;-0.165,\ldots.
\]

Notice that “tails off” does not require a decrease in magnitude at every lag.

The right panels show **one newly simulated sample of 119 observations**, using independent standard-normal innovations, a fixed seed, and a 2,000-observation burn-in. These are constructed data, not paper-towel sales.

Here the sample lag-2 ACF is only \(-0.163\), inside the dashed reference limits, despite a nonzero population value. Conversely, lag 19 crosses those limits even though its population correlation is zero. The limits are approximate pointwise white-noise guides, not exact MA(2) or simultaneous bounds.

Thus, identifying MA(q) means looking for an overall ACF cutoff near \(q\), supported by a trailing PACF. It does **not** mean counting significant spikes. Weak correlations and sampling variation make the proposed order tentative.

[Code](/examples/box-jenkins-identification-20261119-v1/turn09_ma2_correlations.py) · [Numerical results](/examples/box-jenkins-identification-20261119-v1/turn09_ma2_correlations.json)

**Instructor**: For MA(q), ACF is zero beyond q. Interior correlations can vanish for special parameter choices; do not demand a significant sample spike at every early lag.

**Human**: Now extend the rule to AR(p). Show an AR(2) example and how its theoretical ACF and PACF compare with a finite sample.

**Instructor**: Before revealing: Which plot should cut off for a second-order AR model, and can its other plot oscillate?

<!-- checkpoint: 10. AR(p), including AR(2) -->
**Assistant**: For a **zero-mean AR(p)**,

\[
z_t=\phi_1z_{t-1}+\cdots+\phi_pz_{t-p}+a_t,
\]

assume independent innovations with mean zero and common finite variance, and coefficients that give a causal stationary series. The identification rule is: **the population PACF cuts off after lag \(p\), while the ACF generally tails off.**

The reason is in the equation. Once the preceding \(p\) values are included, older values add no further linear predictive information. But ordinary correlations can extend indefinitely because dependence passes through successive values.

Consider the **constructed AR(2) example**, with fixed illustrative coefficients,

\[
z_t=0.6z_{t-1}-0.3z_{t-2}+a_t.
\]

These coefficients give a causal stationary process. Its population ACF obeys

\[
\rho_1=\frac{\phi_1}{1-\phi_2},
\qquad
\rho_k=\phi_1\rho_{k-1}+\phi_2\rho_{k-2}\quad(k\ge2).
\]

Here, \(\rho_k\) denotes correlation at lag \(k\). Starting with \(\rho_0=1\),

\[
\rho_1=\frac{0.6}{1.3}=0.462,
\]
\[
\rho_2=0.6(0.462)-0.3=-0.023,
\]
\[
\rho_3=0.6(-0.023)-0.3(0.462)=-0.152.
\]

Thus, the ACF does not cut off at lag 2. It continues in a **damped oscillation**. “Tails off” allows changes in sign and does not require the magnitude to decrease at every successive lag.

For the PACF,

\[
\operatorname{PACF}(1)=\rho_1=0.462,
\]

and

\[
\operatorname{PACF}(2)
=\frac{\rho_2-\rho_1^2}{1-\rho_1^2}
=\phi_2=-0.300.
\]

At every lag above 2, the population PACF is zero: the two preceding values already capture the model’s linear predictive dependence.

![Population and finite-sample correlations for a constructed AR(2)](/examples/box-jenkins-identification-20261119-v1/turn10_ar2_correlations.png)

The right panels show **one newly simulated sample of 119 observations**, using independent standard-normal innovations, a fixed seed, and a 2,000-observation burn-in. These are constructed data, not paper-towel observations.

The sample PACF begins \(0.442,-0.320,-0.045,\ldots\), reasonably resembling the population cutoff. Its later values fluctuate around zero. The sample ACF likewise departs from the exact population oscillation.

The dashed limits, approximately \(\pm0.180\), are pointwise white-noise reference guides, not exact AR(2) confidence bands or simultaneous bounds.

For identification, look for a PACF that becomes small after approximately \(p\) lags, together with an ACF that fades. This example supports AR(2) as a candidate; finite-sample patterns alone do not verify a model.

[Code](/examples/box-jenkins-identification-20261119-v1/turn10_ar2_correlations.py) · [Numerical results](/examples/box-jenkins-identification-20261119-v1/turn10_ar2_correlations.json)

**Instructor**: The AR(2) illustration is constructed. The legacy chemical-viscosity Example 9.8 is further reading. Both explanations concern the same theoretical signature.

**Human**: What changes when AR and MA terms are mixed? Show an ARMA(1,1) example and bring the identification patterns together.

**Instructor**: Before revealing: If neither population plot has a finite cutoff, what model family becomes plausible?

<!-- checkpoint: 11. Mixed ARMA models -->
**Assistant**: When AR and MA terms are mixed, **neither the population ACF nor the population PACF generally has a sharp cutoff**. Both usually tail off, making the orders harder to identify visually.

A zero-mean **ARMA(1,1)** is

\[
z_t=\phi_1z_{t-1}+a_t-\theta_1a_{t-1}.
\]

Assume independent innovations with mean zero and common finite variance, and \(|\phi_1|<1\) for causal stationarity. The AR term carries previous values forward; the MA term adds a direct contribution from the preceding surprise.

Unlike an AR(1), today’s innovation combination is correlated with \(z_{t-1}\), because both contain \(a_{t-1}\). Consequently, the first autocorrelation changes to

\[
\rho_1=
\frac{(\phi_1-\theta_1)(1-\phi_1\theta_1)}
{1+\theta_1^2-2\phi_1\theta_1}.
\]

At longer lags, the AR propagation remains:

\[
\boxed{\rho_k=\phi_1\rho_{k-1}\quad(k\ge2).}
\]

So its ACF has a geometric tail, but generally does not start at \(\rho_1=\phi_1\).

Consider the **constructed example** with fixed illustrative values \(\phi_1=0.6\) and \(\theta_1=-0.4\):

\[
z_t=0.6z_{t-1}+a_t+0.4a_{t-1}.
\]

![Population and finite-sample correlations for a constructed ARMA(1,1)](/examples/box-jenkins-identification-20261119-v1/turn11_arma11_correlations.png)

Its population ACF begins

\[
0.756,\;0.454,\;0.272,\;0.163,\ldots,
\]

while its population PACF begins

\[
0.756,\;-0.276,\;0.109,\;-0.043,\ldots.
\]

Both continue beyond lag 1. Here the PACF alternates in sign while fading. Conditioning on the preceding observed value does not isolate today’s innovation as it does in a pure AR(1), because the preceding innovation also enters directly.

The right panels show **one newly simulated sample of 119 observations**, with independent standard-normal innovations, a fixed seed, and a 2,000-observation burn-in. These are constructed data, not paper-towel observations. Sampling noise makes the smaller tails difficult to distinguish from zero. The dashed limits are approximate pointwise white-noise guides, not exact ARMA confidence bands.

The population identification patterns come together as follows:

| Model | ACF | PACF |
|---|---|---|
| AR(p) | Tails off | Cuts off after \(p\) |
| MA(q) | Cuts off after \(q\) | Tails off |
| ARMA(p,q) | Generally tails off | Generally tails off |

These describe models without redundant terms; special parameter cancellations can produce simpler behavior.

For mixed models, the plots usually suggest a small set of plausible orders, not a unique answer. Rapidly fading tails can resemble cutoffs in finite samples. Our paper-towel MA(1) therefore remains a tentative candidate; introducing mixed models alone gives no reason to replace it.

[Code](/examples/box-jenkins-identification-20261119-v1/turn11_arma11_correlations.py) · [Numerical results](/examples/box-jenkins-identification-20261119-v1/turn11_arma11_correlations.json)

**Instructor**: Use ARMA(p,q) for a stationary working series and ARIMA(p,d,q) for the original series. The summary table comes after the individual mechanisms.

**Human**: A made-up AI summary says that because both sample plots appear to cut off after lag 1, the paper-towel model must be ARMA(1,1). How should we judge that claim and choose the next step?

**Instructor**: Before revealing: A made-up AI summary claims two apparent cutoffs prove a mixed model. What evidence is missing?

<!-- checkpoint: 12. Ambiguous sample patterns -->
**Assistant**: The claim is **not justified**. It confuses noisy sample patterns with exact population rules, and it assigns the wrong identifying pattern to ARMA(1,1).

For population correlations, the usual rules are:

| Model | ACF | PACF |
|---|---|---|
| AR(1) | Tails off | Cuts off after lag 1 |
| MA(1) | Cuts off after lag 1 | Tails off |
| ARMA(1,1) | Generally tails off | Generally tails off |

Thus, two apparent sample cutoffs do **not** imply ARMA(1,1). Small tails can become visually indistinguishable from zero in a finite sample. An AR(1), MA(1), or mixed model can consequently produce ambiguous plots.

![Sample ACF and PACF of the paper-towel differences](/examples/box-jenkins-identification-20261119-v1/turn06_paper_towel_identification.png)

For our 119 differences, the sample ACF begins \(0.307,-0.065,-0.072\), suggesting a lag-1 cutoff. The PACF begins \(0.307,-0.175,0.006\). Its lag-2 value is close to the approximate reference limit, so describing it as an unequivocal cutoff discards potentially useful information. The ACF also has an isolated lag-18 spike that the summary overlooks.

The correct judgment is: **MA(1) remains a reasonable first candidate, with AR(1) a plausible alternative. Neither plot establishes a unique model.** Crossing or staying inside pointwise reference limits is not an automatic order-selection rule.

We have already fitted the MA(1) by conditional least squares. The next step should be to **examine what dependence remains in its fitted residuals**. A residual is the observed difference minus its one-step prediction. If the model has captured the predictable serial structure, those residuals should have little remaining serial correlation.

That assessment gives a concrete reason to retain or revise the candidate. Remaining short-lag structure could motivate comparing another small model, including ARMA(1,1); the summary’s assertion alone does not justify adding an AR term. Conversely, small residual correlations would support adequacy without proving independence or establishing that the model is uniquely correct.

The summary is therefore a hypothesis to check against the equations and data, not evidence that the model “must” have a particular order.

[Computed sample correlations](/examples/box-jenkins-identification-20261119-v1/turn06_paper_towel_identification.json) · [Existing MA(1) fit and residuals](/examples/box-jenkins-identification-20261119-v1/turn07_ma1_conditional_ls.json)

**Instructor**: This constructed AI exercise practices independent interpretation. Compare candidate fits and residual diagnostics next; do not treat a sample signature as proof or a future exam promise.

**Instructor**: Continue to [Part III: estimation and diagnostics](?transcript=./examples/box-jenkins-diagnostics-20261119-replay.md). Further reading: [ARIMA models, Forecasting: Principles and Practice](https://otexts.com/fpp3/arima.html).
