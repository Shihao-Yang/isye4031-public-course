# Title: Lecture 13 | Regression pitfalls and multicollinearity

**Instructor**: The short Human questions are simulated instructor prompts. The Assistant responses and figures come from a recorded Codex CLI conversation with executed Python. Instructor notes were added for discussion. Playback shows the saved run.

Start with the setup reminder. For one meeting, follow checkpoints 1–13 and allow time for discussion. Checkpoints 14–19 are optional extensions for a later meeting or self-study. The main route covers predictor overlap, coefficient interpretation, proposed remedies, prediction, and diagnostic checks. Optional extensions go deeper into alternative methods and diagnostic calculations.

By the end, explain what a conditional coefficient compares, interpret VIF and uncertainty, evaluate a proposed remedy, and decide what evidence is needed before trusting a prediction or removing an observation.

**Human**: Remind us of the sales-territory question and show us the data.

**Instructor**: Before continuing, say what the response measures and what a row represents.

<!-- checkpoint: 1. Setup -->
**Assistant**: The question is: **How is annual territory sales related to the salesperson’s experience, territory characteristics, advertising, and performance measures?** We want to distinguish associations after accounting for the other predictors from useful predictions and causal claims.

The data contain **25 observations, eight predictors, and no missing values**. We will start with the full eight-predictor regression and retain all 25 observations.

![All 25 sales-territory observations](/examples/lecture13-v3/sales_data_overview.png)

The columns mean:

| Column | Meaning |
|---|---|
| **Sales** | Annual product sales, in hundreds of units |
| Time | Months with the company |
| MktPoten | Product and competing-product sales in the territory, in hundreds of units |
| Adver | Dollar advertising expenditure |
| MktShare | Weighted average of the prior four years’ company market share |
| Change | Change in that share over four years |
| Accts | Accounts measure; the supplied values include fractions |
| WkLoad | Weighted workload-per-account index |
| Rating | Sales manager’s aggregate performance score, on a 1–7 scale |

These are observational data from representatives considered effective. Advertising and experience were not randomly assigned, so their regression coefficients will describe conditional associations. For example, an advertising coefficient would compare sales across observations with the same values of the other seven predictors; it would not by itself establish what increasing advertising would cause.

[Data](/examples/lecture13-v3/sales_territory.csv) · [Code](/examples/lecture13-v3/sales_data_overview.py) · [Results](/examples/lecture13-v3/sales_data_overview_results.json)

**Human**: Would randomly sampling salespeople let us claim advertising causes sales? Show me why.

**Instructor**: Would a representative sample alone justify changing an advertising budget? Explain your reasoning.

<!-- checkpoint: 2. Sampling and causation -->
**Assistant**: **No. Randomly sampling salespeople helps make the sample representative of the population sampled. It does not make advertising independent of the factors that also influence sales.** Random assignment of advertising addresses that second problem.

Here is a **constructed experiment**, separate from our sales-territory data. I generated a population where territory potential influences both advertising and sales:

$$
\text{Advertising}=2\,\text{Potential}+\text{noise},
\qquad
\text{Sales}=3\,\text{Potential}+\text{noise}.
$$

Advertising has **zero causal effect** in this construction: changing advertising leaves the sales equation unchanged.

![Random sampling versus random assignment in constructed data](/examples/lecture13-v3/random_sampling_causality_demo.png)

The left panel uses a random sample of 1,000 observations. Its simple regression of sales on advertising has a slope of **1.179**, despite the zero causal effect. High-potential territories tend to have both more advertising and more sales. Random sampling preserves that relationship.

For the right panel, I randomly reassigned the advertising values within the same sample, leaving sales unchanged. This breaks advertising’s connection to territory potential. The simple regression slope is **0.001**, close to the true effect of zero. Random assignment does not guarantee an exactly zero sample slope; chance variation remains.

For our actual data, including `MktPoten` and the other predictors can adjust for measured differences. A causal interpretation still requires assumptions about unmeasured differences and measurement timing that random sampling alone cannot establish. Also, the stated sampling population was representatives considered effective, which limits whom the results represent.

[Code](/examples/lecture13-v3/random_sampling_causality_demo.py) · [Constructed data](/examples/lecture13-v3/constructed_random_sampling_demo.csv) · [Results](/examples/lecture13-v3/random_sampling_causality_demo_results.json)

**Human**: Can you show which predictors overlap in our actual data?

**Instructor**: Predict which pair overlaps most. Could pairwise plots miss overlap with several predictors together?

<!-- checkpoint: 3. Predictor overlap -->
**Assistant**: **The strongest pairwise overlap is between `Time` and `Accts`, with a correlation of 0.758.** In these 25 observations, longer tenure tends to accompany larger values of the accounts measure.

![Correlations among the eight predictors](/examples/lecture13-v3/predictor_overlap_correlations.png)

Each cell shows a Pearson correlation. Values near $+1$ mean two predictors tend to increase together; values near $-1$ mean they tend to move in opposite directions. Values near zero indicate little pairwise linear association. The diagonal is always 1 because each predictor is perfectly correlated with itself.

The largest correlations in absolute value are:

| Predictor pair | Correlation |
|---|---:|
| Time and Accts | 0.758 |
| Change and Rating | 0.549 |
| MktPoten and Accts | 0.479 |
| Time and MktPoten | 0.454 |
| Adver and Rating | 0.411 |

The `Time`, `Accts`, and `MktPoten` relationships suggest that these predictors share information. In the full eight-predictor sales regression, the coefficient for `Time` describes its association with sales **holding the other seven predictors fixed**, including accounts and market potential. That asks a more specific question than simply comparing salespeople with longer and shorter tenure. Overlap can make those separate coefficients less precisely estimated.

This plot is descriptive; it does not establish causal relationships or show that any predictor should be removed. It also checks only pairs. A predictor can overlap strongly with a **combination** of other predictors even when no individual correlation is especially large. Variance inflation factors would measure that additional overlap.

[Code](/examples/lecture13-v3/predictor_overlap_correlations.py) · [Correlation matrix](/examples/lecture13-v3/predictor_overlap_correlations.csv) · [Results](/examples/lecture13-v3/predictor_overlap_correlations_results.json)

**Human**: Show me what information is left for Accounts after adjustment.

**Instructor**: What comparison should remain after the other predictors have been accounted for?

<!-- checkpoint: 4. Information for Accounts -->
**Assistant**: **About 17.7% of the Accounts variation remains after adjusting linearly for the other seven predictors.** Their combined fit explains 82.3%, so Accounts overlaps substantially with them.

I fitted an auxiliary regression with **Accounts as the outcome** and `Time`, `MktPoten`, `Adver`, `MktShare`, `Change`, `WkLoad`, and `Rating` as predictors, retaining all 25 observations. For each observation, the information left is its residual:

$$
\text{Accounts residual}
=
\text{observed Accounts}
-
\text{Accounts fitted from the other predictors}.
$$

![Accounts variation remaining after adjustment](/examples/lecture13-v3/accounts_after_adjustment.png)

The left panel compares observed Accounts with the values fitted from the other predictors. The right panel shows what remains. A positive residual means the observation has a higher Accounts value than the linear model predicts from its other characteristics; a negative residual means it has a lower value. These differences range from **−26.62 to +42.59** in the supplied Accounts units.

The 17.7% refers to the fraction of total **squared variation around the mean** left in these residuals. It does not mean that only 17.7% of the observations remain useful.

In the full eight-predictor sales regression, this residual Accounts variation is what identifies the Accounts coefficient. More precisely, that coefficient compares residual sales with residual Accounts after removing the same seven predictors from both. The corresponding Accounts variance inflation factor is **5.64**, quantifying the loss of coefficient precision associated with this overlap.

This is an in-sample description of linear overlap, not predictive validation or evidence of a causal Accounts effect.

[Code](/examples/lecture13-v3/accounts_after_adjustment.py) · [Residual values](/examples/lecture13-v3/accounts_after_adjustment_values.csv) · [Results](/examples/lecture13-v3/accounts_after_adjustment_results.json)

**Instructor**: Quick calculation: $\mathrm{VIF}=1/(1-R^2_{\mathrm{aux}})=1/(1-0.8227)\approx5.64$. Only 17.7% of the centered Accounts variation remains. At fixed error variance and Accounts spread, the standard-error multiplier is $\sqrt{5.64}\approx2.37$. A VIF above 10 requires an auxiliary $R^2$ above 0.90; any positive overlap does not imply that threshold.

**Human**: Can you show how that affects the Accounts slope and uncertainty?

**Instructor**: What would make the adjusted Accounts slope precise or imprecise?

<!-- checkpoint: 5. Accounts slope and uncertainty -->
**Assistant**: **Adjustment reduces the Accounts slope from 21.72 to 5.55, while increasing its standard error from 3.95 to 4.78.** The adjusted estimate is therefore smaller and less precise.

| Sales regression | Accounts slope | Standard error | 95% confidence interval |
|---|---:|---:|---:|
| Accounts only | 21.72 | 3.95 | 13.56 to 29.88 |
| All eight predictors | 5.55 | 4.78 | −4.57 to 15.67 |

Slopes are in hundreds of sales units per unit of the supplied Accounts measure.

![Accounts slopes and uncertainty before and after adjustment](/examples/lecture13-v3/accounts_slope_uncertainty.png)

The left panel removes the other seven predictors from **both Accounts and Sales**. The fitted slope through these residuals is exactly the Accounts coefficient in the full model: **5.55**. The larger unadjusted slope includes associations that Accounts shares with the other predictors.

The uncertainty follows:

$$
\operatorname{SE}(\hat\beta_{\text{Accounts}})
=
\frac{s}{\sqrt{\sum_i(\text{Accounts residual}_i)^2}},
$$

where $s$ is the full sales model’s estimated residual standard deviation. Less Accounts variation remains in the denominator, making its separate slope harder to estimate.

With only 17.7% of Accounts variation remaining, overlap multiplies the standard error by $\sqrt{5.64}=2.37$, **holding the full model’s error variance fixed**. That is an algebraic comparison, not the actual ratio between the two models’ standard errors. Adjustment also reduces unexplained sales variation: the residual standard deviation falls from 881.09 to 449.03, partly offsetting the loss of Accounts information.

The full-model interval includes zero. These data do not estimate the conditional Accounts association precisely enough to rule out zero; they do not establish that Accounts has no relationship with sales. The intervals use classical OLS assumptions, including constant error variance and normal errors for exact small-sample coverage.

[Code](/examples/lecture13-v3/accounts_slope_uncertainty.py) · [Results](/examples/lecture13-v3/accounts_slope_uncertainty_results.json) · [Residual data](/examples/lecture13-v3/accounts_sales_partial_residuals.csv)

**Human**: What would happen with an exact copy of Accounts? Show me.

**Instructor**: Could two different coefficient pairs give the same predictions?

<!-- checkpoint: 6. Exact duplication -->
**Assistant**: **An exact copy leaves fitted sales unchanged, but makes the two Accounts coefficients impossible to identify separately.**

I added a **constructed column equal to Accounts in every row**, keeping all 25 observations and the original predictors.

![Exact-copy experiment: unchanged predictions and nonunique coefficients](/examples/lecture13-v3/accounts_exact_copy_demo.png)

Because the columns are identical, their contribution to fitted sales is

$$
\beta_A\,\text{Accounts}+\beta_C\,\text{Copy}
=(\beta_A+\beta_C)\,\text{Accounts}.
$$

The data can identify the **sum** of the coefficients, but cannot determine how to divide it between them. For example, all these choices produce the same fitted sales:

| Accounts coefficient | Copy coefficient | Sum |
|---:|---:|---:|
| −7.225 | 12.775 | 5.551 |
| 2.775 | 2.775 | 5.551 |
| 12.775 | −7.225 | 5.551 |

The software returned the equal split using a pseudoinverse and issued a warning that the parameters are not uniquely determined. That split is a computational convention, not evidence that the two columns have separate equal associations.

The design matrix now has **10 columns including the intercept, but rank 9**. The copy adds no independent information. Fitted sales differ from the original model by less than $3\times10^{-10}$, just numerical precision.

After adjustment for all the other columns, Accounts has **zero residual variation**, because its copy explains it perfectly. Its separate slope therefore has no identifiable standard error or confidence interval. The combined slope remains **5.55**, with the original standard error **4.78** and 95% interval **−4.57 to 15.67**.

[Code](/examples/lecture13-v3/accounts_exact_copy_demo.py) · [Constructed data](/examples/lecture13-v3/sales_with_constructed_accounts_exact_copy.csv) · [Results](/examples/lecture13-v3/accounts_exact_copy_demo_results.json)

**Human**: And if the copy is only almost identical? Show me.

**Instructor**: Predict what happens to the separate slopes when two predictors become almost identical.

<!-- checkpoint: 7. Near duplication -->
**Assistant**: **An almost identical copy makes the two slopes identifiable, but extremely imprecise and sensitive to tiny changes.** Large opposite coefficients can cancel, leaving fitted sales much more stable.

I constructed
$$
\text{Copy}=\text{Accounts}+\text{random noise},
$$
with noise standard deviation **0.1 Accounts units** and a fixed seed. All original data remain unchanged. Accounts and its constructed copy have correlation **0.9999965**.

| Model coefficient | Estimate | Standard error |
|---|---:|---:|
| Accounts, original eight-predictor model | 5.55 | 4.78 |
| Accounts, model including near copy | 1083.27 | 916.05 |
| Near copy, same model | −1078.14 | 916.40 |
| Sum of the two coefficients, same model | 5.13 | 4.73 |

The design now has full rank, so there is a unique least-squares solution. But almost no Accounts variation remains after adjusting for its near copy. The model struggles to distinguish their separate associations.

![Coefficient sensitivity and fitted sales with a constructed near copy](/examples/lecture13-v3/accounts_near_copy_demo.png)

For the left panel, I made **200 additional constructed perturbations**, each adding noise with standard deviation **0.01** to the copy. The Accounts coefficient ranged from **903 to 1294**, while the copy’s coefficient moved in the opposite direction. Their sum stayed between **4.79 and 5.37**.

The right panel compares fitted sales before and after adding the initial near copy. They are close, but no longer identical: the root mean squared difference is **104.41 hundreds of sales units**. The small noise supplies an extra direction that the model can fit.

This is a **sensitivity experiment**, not predictive validation. It shows why unstable individual coefficients can coexist with more stable fitted values.

[Code](/examples/lecture13-v3/accounts_near_copy_demo.py) · [Constructed data](/examples/lecture13-v3/constructed_accounts_near_copy_data.csv) · [Results](/examples/lecture13-v3/accounts_near_copy_demo_results.json) · [Perturbation results](/examples/lecture13-v3/accounts_near_copy_sensitivity_trials.csv)

**Human**: Would centering or adding an interaction help? Show the difference.

**Instructor**: What would centering change? What question would an interaction add?

<!-- checkpoint: 8. Proposed remedies -->
**Assistant**: **Centering does not fix the near-copy problem. Adding an interaction changes the model, allowing the Accounts slope to depend on another predictor.** I used Time for that comparison.

![Centering versus adding an Accounts–Time interaction](/examples/lecture13-v3/centering_and_interaction_comparison.png)

For the left panel, I subtracted each predictor’s mean in the constructed near-copy model.

| Near-copy model | Accounts slope | Standard error |
|---|---:|---:|
| Original predictors | 1083.27 | 916.05 |
| Centered predictors | 1083.27 | 916.05 |

Centering leaves the slopes, their uncertainty, and fitted sales unchanged to numerical precision. It moves the intercept’s reference point, but supplies no new information to distinguish Accounts from its copy. An exact duplicate would also remain an exact duplicate after centering.

For the right panel, I returned to the **actual eight predictors, without the near copy**, and added the constructed product
$$
(\text{Accounts}-\overline{\text{Accounts}})
(\text{Time}-\overline{\text{Time}}).
$$

In this interaction model, the estimated conditional Accounts slope is
$$
6.69-0.01475(\text{Time}-87.642).
$$

The estimate suggests a decreasing Accounts slope with tenure, but the interaction’s 95% interval is **−0.0627 to 0.0332**, including zero. These 25 observations provide weak evidence for that change. The shaded band shows pointwise uncertainty, not a simultaneous confidence band.

Centering helps interpret this interaction: the Accounts coefficient **6.69** applies at mean tenure, **87.642 months**. Without centering, its coefficient is **7.99**, applying at zero tenure. Both parameterizations give identical fitted sales when all main effects are retained.

Thus, centering helps interpretation; interaction adds flexibility. Neither automatically resolves overlapping predictors. This comparison describes in-sample associations, not validated prediction or causal effects.

[Code](/examples/lecture13-v3/centering_and_interaction_comparison.py) · [Results](/examples/lecture13-v3/centering_and_interaction_results.json) · [Slope curve](/examples/lecture13-v3/accounts_time_interaction_slopes.csv)

**Human**: Could we just drop Accounts? Show me what changes.

**Instructor**: How would removing Accounts change the meaning of the Time coefficient?

<!-- checkpoint: 9. Dropping Accounts -->
**Assistant**: **We can fit a model without Accounts, and the overall fit changes modestly. But dropping it changes what the other slopes mean.** This comparison uses the actual data, all 25 observations, and no constructed copy or interaction.

![Changes in fitted sales after dropping Accounts](/examples/lecture13-v3/drop_accounts_comparison.png)

| Measure | Full eight-predictor model | Without Accounts |
|---|---:|---:|
| $R^2$ | 0.9220 | 0.9155 |
| Adjusted $R^2$ | 0.8831 | 0.8806 |
| Residual standard deviation | 449.03 | 453.64 |

Residual standard deviations are in hundreds of sales units. The root mean squared change in fitted sales is **104.39**, with a largest absolute change of **236.42** at observation 12.

Some other coefficients move substantially:

| Predictor | Full-model slope | Without Accounts |
|---|---:|---:|
| Time | 2.01 | 3.70 |
| MktShare | 199.02 | 256.13 |
| WkLoad | 19.79 | 2.36 |

In the full model, Time’s slope holds Accounts fixed. After dropping Accounts, it no longer does. Because Accounts overlaps with Time and the other predictors, they absorb some of its association with sales. Time’s standard error also falls from **1.93 to 1.28**, but that greater precision describes a different conditional association.

The classical partial $F$ test for adding Accounts to the seven-predictor model gives $F(1,16)=1.35$, $p=0.262$. That is weak evidence for its additional linear contribution, not proof that Accounts is irrelevant.

Dropping Accounts is therefore a possible simplification, with a small in-sample fit cost. Whether it improves prediction requires validation of the selection procedure. Whether it is appropriate for interpretation depends on whether we need to compare territories while holding Accounts fixed.

[Code](/examples/lecture13-v3/drop_accounts_comparison.py) · [Results](/examples/lecture13-v3/drop_accounts_comparison_results.json) · [Coefficient comparison](/examples/lecture13-v3/drop_accounts_coefficient_comparison.csv) · [Fitted changes](/examples/lecture13-v3/drop_accounts_fitted_changes.csv)

**Human**: Do Time and Accounts matter together? Can you show the joint evidence?

**Instructor**: Can a pair of predictors matter jointly while each individual estimate is imprecise?

<!-- checkpoint: 10. Joint evidence -->
**Assistant**: **Yes, there is evidence that Time and Accounts contribute jointly to explaining sales after accounting for the other six predictors.** The joint test gives **$F(2,16)=4.93$, $p=0.0215$**, even though neither individual test is significant at 5%.

I compared the original eight-predictor model with a model omitting both Time and Accounts, retaining all 25 observations. The joint null hypothesis is

$$
H_0:\ \beta_{\text{Time}}=\beta_{\text{Accounts}}=0.
$$

| Test in the full model | $p$-value |
|---|---:|
| Time alone, holding Accounts and the others fixed | 0.3134 |
| Accounts alone, holding Time and the others fixed | 0.2621 |
| Time and Accounts jointly | **0.0215** |

Adding the pair reduces the residual sum of squares from **5,212,680 to 3,225,985**. They explain **38.1% of the variation left unexplained by the six-predictor model**, an in-sample partial $R^2$.

![Joint confidence region for Time and Accounts coefficients](/examples/lecture13-v3/time_accounts_joint_evidence.png)

The ellipse is the **95% joint confidence region** for the two coefficients. The point $(0,0)$ lies outside it, agreeing with the joint test. Yet the region crosses each zero axis: either coefficient can be zero when the other is allowed to compensate.

That is how overlap can produce weak individual evidence alongside stronger joint evidence. These data have difficulty dividing the association between Time and Accounts, but provide evidence against removing both contributions together.

The test supports **at least one nonzero conditional slope**, not a claim that both individually matter. It relies on the classical regression error assumptions and establishes neither causal effects nor predictive performance on new territories.

[Code](/examples/lecture13-v3/time_accounts_joint_evidence.py) · [Results](/examples/lecture13-v3/time_accounts_joint_evidence_results.json) · [Joint region](/examples/lecture13-v3/time_accounts_joint_confidence_region.csv)

**Human**: Could a territory be inside every range but still risky to predict? Show me.

**Instructor**: Is staying within every observed range enough to support a new prediction?

<!-- checkpoint: 11. A new territory -->
**Assistant**: **Yes. Being inside every predictor’s range does not guarantee that the combination resembles the observed territories.** Prediction can require extrapolation across combinations even when no individual value exceeds its range.

I constructed a hypothetical territory by examining 256 combinations at 5% or 95% of each observed range and choosing the one with highest prediction leverage. Every value is strictly inside its range.

![A constructed territory inside every range but unusual jointly](/examples/lecture13-v3/within_ranges_prediction_risk.png)

For example, it combines **347.19 months of tenure with an Accounts value of 83.92**. Both values occur within the observed ranges, but that pairing runs against the positive Time–Accounts relationship. The left panel shows this mismatch; the calculations use all eight predictors.

Under the original full eight-predictor model:

| Constructed territory | Prediction leverage $h_0$ | Predicted Sales | 95% prediction interval |
|---|---:|---:|---:|
| All predictors at their means | 0.040 | 3374.57 | 2403.82 to 4345.31 |
| Unusual combination | 9.567 | 4692.87 | 1598.53 to 7787.22 |

Sales values are in hundreds of units. Prediction leverage measures how far the combination lies from the data’s center, accounting for predictor overlap. For a new point, it can exceed 1.

The standard error for predicting one new observation is

$$
s\sqrt{1+h_0}.
$$

Thus, the unusual combination produces much greater uncertainty. More fundamentally, its prediction relies heavily on the assumed linear relationship extending into a sparsely supported combination of characteristics.

These are **model-based intervals for hypothetical territories**, not validation results. Even the wide interval does not account for a linear model failing in that unusual region.

[Code](/examples/lecture13-v3/within_ranges_prediction_risk.py) · [Constructed territories](/examples/lecture13-v3/constructed_within_ranges_territories.csv) · [Results](/examples/lecture13-v3/within_ranges_prediction_risk_results.json)

**Human**: Which observed territories are unusual, and which actually change the fit? Show me.

**Instructor**: Which matters more for influence: an unusual predictor profile, a large residual, or both?

<!-- checkpoint: 12. Influential territories -->
**Assistant**: **Row 7 has the most unusual predictor combination and the greatest influence on the full-model fit. Rows 17 and 16 also have unusual predictors, but change the fit much less.** Row numbers refer to CSV data order.

![Leverage, residuals, and measured influence for observed territories](/examples/lecture13-v3/observed_territory_influence.png)

Leverage measures unusual combinations of the eight predictors. The studentized residual measures how unusual observed sales are relative to the model. Cook’s distance combines these to summarize influence.

I also refitted the model omitting each row individually:

| Row | Leverage | Studentized residual | Cook’s distance | RMS fitted-sales change |
|---|---:|---:|---:|---:|
| 7 | 0.716 | −1.07 | 0.317 | 151.74 |
| 12 | 0.502 | 1.43 | 0.215 | 124.85 |
| 10 | 0.219 | 2.22 | 0.123 | 94.41 |
| 1 | 0.517 | 0.93 | 0.104 | 86.74 |
| 17 | 0.562 | −0.22 | 0.007 | 22.61 |
| 16 | 0.513 | −0.26 | 0.008 | 24.63 |

RMS changes are measured at **all 25 original predictor combinations**, in hundreds of sales units.

The distinction matters: rows 17 and 16 have high leverage but small residuals, so their deletion has little effect. Row 10 has lower leverage but a larger sales discrepancy, making it more influential.

Individual slopes are sensitive too. The full-model Accounts slope is **5.55**; omitting row 7 changes it to **7.68**, while omitting row 12 changes it to **1.30**.

These deletion refits are **sensitivity analysis**, not validation or evidence that a row is erroneous. The full model still retains every observation. Influence identifies rows whose measurements and circumstances deserve closer examination before any exclusion.

[Code](/examples/lecture13-v3/observed_territory_influence.py) · [Results](/examples/lecture13-v3/observed_territory_influence_results.json) · [All diagnostics](/examples/lecture13-v3/observed_territory_influence_values.csv) · [Selected original rows](/examples/lecture13-v3/unusual_territories_original_values.csv)

**Instructor**: Follow-up decisions depend on evidence. Verify flagged records against their source and investigate the territory circumstances. A documented group difference could motivate an indicator or interaction; a transformation needs a reason tied to model form or error spread. Compare diagnostics and conclusions after a justified change. A numerical flag alone does not justify deleting an observation.

**Human**: Do the error assumptions look reasonable? Show me the diagnostic plots.

**Instructor**: Which assumptions can these plots help check, and which need information about how the data were collected?

<!-- checkpoint: 13. Error assumptions -->
**Assistant**: **The diagnostics look broadly compatible with normal errors, but leave some concern about the mean pattern and constant variance. Independence cannot be established from these plots.** These results describe the original full eight-predictor model, retaining all 25 observations.

![Full-model error diagnostic plots](/examples/lecture13-v3/full_model_error_diagnostics.png)

**Residuals versus fitted sales:** The red smooth dips below zero in the middle and rises at both ends. That suggests possible curvature missing from the additive linear model. With only 25 observations and few points at the upper end, this is a clue to investigate, not decisive evidence of nonlinearity.

**Normal Q–Q plot:** The points follow the reference line reasonably closely, with some departures but no striking tail discrepancy. Row 10 has the largest absolute externally studentized residual, **2.22**. The plot does not suggest a severe normality problem, although this small sample cannot establish normality.

**Scale–location plot:** Standardized residual magnitudes tend to be larger at higher fitted sales than around the lower-middle range. There is no dramatic funnel, but constant variance is not beyond question. A Breusch–Pagan screen gives $p=0.193$; that does not rule out changing variance, especially with eight predictors and only 25 observations.

**Residuals in row order:** There is a stretch of mostly negative residuals later in the file. However, CSV order is not documented as time or collection order, so we cannot interpret this as temporal dependence. Assessing independence requires information about shared territories, timing, or other grouping.

The classical intervals and tests remain useful provisional summaries. These plots qualify their assumptions; they do not validate prediction or justify deleting observations.

[Code](/examples/lecture13-v3/full_model_error_diagnostics.py) · [Results](/examples/lecture13-v3/full_model_error_diagnostics_results.json) · [Diagnostic values](/examples/lecture13-v3/full_model_error_diagnostics_values.csv)

**Instructor**: End of the main route. Before moving on, state one supported conclusion, one claim the evidence cannot support, and the next check to run. For example, predictor overlap limits separate coefficient precision; it does not establish a causal advertising effect. The residual pattern motivates checking model form and collecting information about timing or groups. The following checkpoints are optional extensions.

**Human**: Could we combine the overlapping predictors? Show me what PCA does.

**Instructor**: What information might a combined predictor preserve, and what interpretation might it lose?

<!-- checkpoint: 14. Optional: combining predictors and PCA -->
**Assistant**: **Yes. PCA combines the predictors into uncorrelated components, but reducing their number can discard information relevant to sales.**

I standardized all eight predictors before PCA so their different units would not determine the result. **Sales was excluded from PCA.**

![PCA variance explained and component weights](/examples/lecture13-v3/predictor_pca_exploration.png)

Each component is a weighted combination of standardized predictors. For example, the largest weights in PC1 include

$$
\text{PC1}=0.41z_{\text{Time}}+0.44z_{\text{Accounts}}
+0.39z_{\text{MktPoten}}+\cdots.
$$

Thus, Time and Accounts contribute together to a direction of shared variation. PC1 also includes the other predictors, as the right panel shows. The components are uncorrelated in this sample; that does not make them independent or give them causal meanings.

| Components retained | Predictor variation retained | In-sample sales $R^2$ |
|---|---:|---:|
| First 2 | 56.2% | 0.8408 |
| First 3 | 72.2% | 0.8448 |
| First 6 | 94.6% | 0.8875 |
| All 8 | 100% | 0.9220 |

Keeping **all eight components reproduces the original regression’s fitted sales** to numerical precision. It changes coordinates without reducing information, so it does not solve the difficulty of estimating separate original-predictor associations.

Keeping fewer components imposes a restriction: sales cannot depend on the discarded directions. That can stabilize estimation, but PCA prioritizes **predictor variation**, not association with sales. Here, adding PC7, which carries only 4.3% of predictor variation, raises in-sample sales $R^2$ from 0.8875 to 0.9213.

These are exploratory fits, not validation. To choose a component count for prediction, standardization, PCA, and tuning must all occur inside training folds.

[Code](/examples/lecture13-v3/predictor_pca_exploration.py) · [Results](/examples/lecture13-v3/predictor_pca_exploration_results.json) · [Weights](/examples/lecture13-v3/predictor_pca_weights.csv) · [Fit comparison](/examples/lecture13-v3/predictor_pca_in_sample_fit_comparison.csv)

**Instructor**: Other combinations need interpretation too. A sum or average requires compatible units or a justified standardization; a ratio changes the question and can be unstable near a zero denominator. Factor analysis models shared latent factors and measurement errors, so it requires assumptions beyond the PCA rotation shown here. A fitted factor model is deferred.

**Human**: How would ridge help? Show me the tradeoff.

**Instructor**: What changes when we allow bias to reduce instability?

<!-- checkpoint: 15. Optional: ridge -->
**Assistant**: **Ridge limits the large, compensating coefficients that overlapping predictors can produce. The tradeoff is bias and a worse training fit in exchange for potentially more stable predictions.**

I standardized all eight actual predictors and fitted ridge by minimizing

$$
\sum_i(y_i-\hat y_i)^2+\alpha\sum_{j=1}^{8}\beta_j^2,
$$

with an unpenalized intercept. At $\alpha=0$, this is ordinary least squares.

![Ridge coefficient shrinkage and the tradeoff between fit and sensitivity](/examples/lecture13-v3/ridge_tradeoff_analysis.png)

The left panel shows coefficient shrinkage. Individual coefficients need not shrink monotonically, but their combined squared magnitude decreases. In PCA terms, ridge suppresses directions with little predictor variation most strongly. Those are often the directions where separating overlapping predictors is difficult. Unlike truncated PCA, ridge gradually shrinks directions without discarding them outright.

| Penalty $\alpha$ | Training RMSE | Mean deletion change | Largest deletion change |
|---|---:|---:|---:|
| 0 | 359.22 | 55.09 | 151.74 |
| 1 | 361.11 | 49.29 | 163.66 |
| 10 | 429.06 | 40.42 | 154.79 |
| 100 | 860.79 | 37.24 | 121.55 |

All values are in hundreds of sales units. Deletion changes measure how much fitted sales at the 25 original points change after omitting one observation and refitting, including standardization.

A small penalty reduces **average sensitivity** with little training-fit cost. However, the largest deletion change initially increases, so ridge does not guarantee improvement for every observation. A strong penalty eventually sacrifices substantial fit.

This is **sensitivity analysis, not predictive validation**. Choosing $\alpha$ requires validation with scaling and tuning inside training folds. Ridge also cannot resolve causal interpretation or supply information that the data lack.

[Code](/examples/lecture13-v3/ridge_tradeoff_analysis.py) · [Results](/examples/lecture13-v3/ridge_tradeoff_analysis_results.json) · [Coefficient paths](/examples/lecture13-v3/ridge_standardized_coefficient_path.csv) · [Tradeoff values](/examples/lecture13-v3/ridge_tradeoff_path_results.csv)

**Human**: Can a polynomial have the same identification problem? Show me.

**Instructor**: Would repeated observations at the same x values identify a more flexible polynomial?

<!-- checkpoint: 16. Optional: polynomial estimability -->
**Assistant**: **Yes. Polynomial terms can have the same exact or near identification problem as duplicated predictors.** What matters is the information in the observed $x$ values.

I constructed 30 observations at just two levels, $x=0$ and $x=1$, with
$$
y=2+3x+\text{random noise},
$$
then fitted a quadratic:
$$
\hat y=\beta_0+\beta_1x+\beta_2x^2.
$$

![Polynomial identification problems in constructed data](/examples/lecture13-v3/polynomial_identification_demo.png)

**At these two levels, $x^2=x$.** The linear and quadratic columns are exact copies. The design has three columns but rank 2, so only $\beta_1+\beta_2$ is identifiable.

The three curves in the left panel have different linear and quadratic coefficients, but identical fitted values at every observed point. Their coefficients sum to **2.959**. Between 0 and 1, their predictions differ substantially. Being inside the observed range does not supply information about curvature.

For the right panel, I added tiny random perturbations to $x$, with standard deviation **0.001**, keeping $y$ unchanged. The design now has full rank, but almost no information separates $x$ from $x^2$:

| Near-two-level quadratic term | Estimate | Classical standard error |
|---|---:|---:|
| Linear coefficient | −64.27 | 21.64 |
| Quadratic coefficient | 67.23 | 21.64 |

The large opposite coefficients largely cancel near the observed levels, yet produce a pronounced curve in the unsupported middle.

Centering or using an orthogonal polynomial basis can improve numerical conditioning, but cannot create missing information or repair exact rank deficiency. This is a **constructed illustration**, separate from the sales data, not evidence that their relationship is quadratic.

[Code](/examples/lecture13-v3/polynomial_identification_demo.py) · [Constructed data](/examples/lecture13-v3/constructed_polynomial_identification_data.csv) · [Results](/examples/lecture13-v3/polynomial_identification_demo_results.json)

**Instructor**: For a degree-$p$ polynomial in one predictor, identifying all $p+1$ coefficients requires at least $p+1$ distinct $x$ values. Repeated measurements at existing levels can help estimate noise, but do not add new polynomial directions. Distinct values that cluster tightly can still leave poor precision.

**Human**: How do PRESS and DFFITS relate to leaving observations out? Show me.

**Instructor**: How does evaluating an omitted point differ from evaluating a point used to fit the model?

<!-- checkpoint: 17. Optional: deleted residuals, PRESS and DFFITS -->
**Assistant**: **Both use leave-one-out refits, but measure different things: PRESS measures prediction error for the omitted observations; DFFITS measures how much each observation changes its own fitted value.**

I explicitly refitted the full eight-predictor model 25 times, each time omitting one observation. The results match the OLS shortcut formulas to numerical precision.

![PRESS residuals and DFFITS from leave-one-out refits](/examples/lecture13-v3/press_dffits_deletion_check.png)

Let $e_i=y_i-\hat y_i$ be the ordinary residual and $h_i$ its leverage. The prediction error when observation $i$ is omitted is

$$
y_i-\hat y_{i(-i)}=\frac{e_i}{1-h_i},
\qquad
\mathrm{PRESS}=\sum_i\left(\frac{e_i}{1-h_i}\right)^2.
$$

High leverage magnifies the deleted residual. For row 7, the ordinary residual is **−254.37**, but its held-out prediction error is **−896.49**.

DFFITS instead standardizes the change in fitted value:

$$
\mathrm{DFFITS}_i=
\frac{\hat y_i-\hat y_{i(-i)}}{s_{(-i)}\sqrt{h_i}}
=
t_{i,\mathrm{external}}\sqrt{\frac{h_i}{1-h_i}},
$$

where $s_{(-i)}$ is the residual standard deviation from the deletion fit.

| Row | Deleted prediction error | DFFITS |
|---|---:|---:|
| 7 | −896.49 | −1.697 |
| 12 | 881.47 | 1.435 |
| 10 | 1009.67 | 1.173 |

Row 10 contributes most to PRESS; row 7 has the largest absolute DFFITS. Prediction difficulty and influence are related, but their rankings can differ.

For this **fixed full-model specification**, PRESS is **7,551,189**, giving leave-one-out RMSE **549.59**, versus training RMSE **359.22**, in hundreds of sales units. This assesses held-out prediction for that specification. It does not validate the broader model exploration, resolve sampling dependence, or establish that predictors will be available when forecasting.

[Code](/examples/lecture13-v3/press_dffits_deletion_check.py) · [Results](/examples/lecture13-v3/press_dffits_deletion_results.json) · [Deletion calculations](/examples/lecture13-v3/press_dffits_deletion_values.csv)

**Instructor**: Terminology check: an internally studentized residual is $r_i=e_i/[s\sqrt{1-h_i}]$; an externally studentized residual replaces $s$ with $s_{(-i)}$. The deleted residual $e_i/(1-h_i)$ is in response units. Cook's distance summarizes the change across all fitted values, while DFFITS measures the standardized change at the omitted point.

Screening rules such as $|r_i|>2$, $h_i>2q/n$, Cook's $D_i>4/n$, or $|\mathrm{DFFITS}_i|>2\sqrt{q/n}$ use different conventions and are only prompts to investigate. Here $q=9$ includes the intercept. The leverage cutoff is 0.72, just above even row 7, which still has the largest deletion influence. A threshold does not replace the plots, source checks, or sensitivity analysis.

**Human**: Can you construct examples with the same summary but different plots?

**Instructor**: Predict whether identical regression summaries would guarantee similar data patterns.

<!-- checkpoint: 18. Optional: matching summaries -->
**Assistant**: **Yes. These three newly constructed datasets have the same regression summaries, but their plots tell very different stories.** They are separate from our sales-territory data.

![Constructed datasets with identical summaries but different patterns](/examples/lecture13-v3/same_summary_different_plots.png)

Each dataset has 30 observations and, to numerical precision, the following identical values:

| Summary | All three datasets |
|---|---:|
| Mean of $x$ | 0 |
| Mean of $y$ | 5 |
| Sample variance of $x$ | 1.4744 |
| Sample variance of $y$ | 6.9322 |
| Correlation | 0.9224 |
| Fitted regression | $\hat y=5+2x$ |
| $R^2$ | 0.8508 |
| Residual sum of squares | 30 |
| Classical slope standard error | 0.1583 |

The **left dataset** has scattered residuals without the clear systematic structure seen in the other panels. The **middle dataset** follows a curved relationship: the linear fit systematically underpredicts at the ends and overpredicts in the middle. The **right dataset** has one large residual while the remaining points closely follow a line. That observation would deserve investigation, not automatic deletion.

I constructed them using the same $x$ values and
$$
y=5+2x+e.
$$
Each residual vector has zero sum, is orthogonal to $x$, and has squared length 30. Those constraints preserve the listed summaries while allowing the residual pattern to change.

The identical standard errors are especially instructive: the classical formula produces the same number even when the plots raise different concerns about its assumptions. A strong correlation and high $R^2$ cannot establish that a straight-line model adequately describes the data.

[Code](/examples/lecture13-v3/same_summary_different_plots.py) · [Constructed data](/examples/lecture13-v3/constructed_same_summary_datasets.csv) · [Results](/examples/lecture13-v3/same_summary_different_plots_results.json)

**Human**: Would an Anderson–Darling test settle whether these models are okay? Show me.

**Instructor**: Would a large normality-test p-value establish that a regression model is adequate?

<!-- checkpoint: 19. Optional: normality tests -->
**Assistant**: **No. An Anderson–Darling test addresses normality, not whether the regression model is adequate.** A model can pass a normality screen while having a wrong mean relationship, changing variance, dependent errors, or unstable coefficients.

I applied the test to residuals from the full sales model and the three constructed examples:

| Residuals tested | AD statistic | Nominal $p$-value |
|---|---:|---:|
| Actual sales, full eight-predictor model | 0.288 | 0.588 |
| Constructed scattered example | 0.301 | 0.559 |
| Constructed curved example | 1.048 | 0.008 |
| Constructed one-large-residual example | 6.929 | $2.08\times10^{-17}$ |

![Normal Q–Q plots and Anderson–Darling residual screens](/examples/lecture13-v3/anderson_darling_model_limits.png)

The test compares the observed distribution with a normal distribution, giving particular weight to tail discrepancies. Here it flags the constructed curved and one-large-residual examples. But it does not explain *why* their residual distributions differ from normal. Their earlier residual-versus-$x$ plots reveal curvature and a concentrated discrepancy.

For the actual sales model, $p=0.588$ means the screen finds little evidence against normality. It does **not** remove the curvature or spread concerns we saw in its diagnostic plots. The test ignores where residuals occur relative to predictors.

There is also a calibration limitation: these are **nominal normality-test $p$-values applied to fitted residuals**. Even under independent normal errors, regression residuals are correlated and have variances depending on leverage. Thus, these values are descriptive screens, not exact tests of regression adequacy.

Normality matters for exact small-sample classical inference. Checking it complements plots and influence analysis; it cannot settle predictive reliability, independence, identification, or causal interpretation.

[Code](/examples/lecture13-v3/anderson_darling_model_limits.py) · [Results](/examples/lecture13-v3/anderson_darling_model_limits_results.json)

**Instructor**: Reference: Bowerman, O'Connell and Koehler, Forecasting, Time Series, and Regression: An Applied Approach, fourth edition, Example 4.2 and Sections 5.1 and 5.4. The sales-territory data are the same course example used previously. Additional teaching datasets are explicitly labeled as constructed.

[Sales-territory data](/examples/lecture13-v3/sales_territory.csv) · [Python package versions](/examples/lecture13-v3/requirements.txt)
