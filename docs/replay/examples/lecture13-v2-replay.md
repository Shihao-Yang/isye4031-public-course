# Title: Lecture 13 | Regression pitfalls and multicollinearity

**Instructor**: Short simulated instructor questions guide this recorded Codex CLI conversation. The responses, calculations, and linked code come from the actual run. Playback shows saved execution. These labeled Instructor notes were added for discussion.

By the end, explain what identifies a conditional regression coefficient, interpret VIF and coefficient uncertainty, evaluate a proposed model change, and check whether a new prediction has support in the data.

We continue with the same 25 sales territories as Lecture 12. Sales is annual sales in hundreds of units, and Time is months with the company. These observational data do not establish causal effects.

The reported tests and intervals assume a linear mean with independent, constant-variance errors; exact small-sample inference also assumes normal errors.

**Human**: I'm worried these predictors overlap. Is multicollinearity a problem in the sales data?

**Instructor**: Before continuing, predict which pair of predictors overlaps most. What evidence would change your view?

<!-- checkpoint: 1. Predictor overlap -->
**Assistant**: **There is meaningful overlap, especially for Accts, but these diagnostics do not suggest extreme multicollinearity.** The main concern is how precisely we can separate individual predictor associations in the full regression of Sales on all eight predictors.

The strongest pairwise correlation is between **Time and Accts**, at $r=0.758$. Pairwise correlations alone can miss overlap involving several predictors, so I also calculated variance inflation factors (VIFs).

| Predictor | VIF |
|---|---:|
| Accts | 5.64 |
| Time | 3.34 |
| MktShare | 3.24 |
| MktPoten | 1.98 |
| Adver | 1.91 |
| WkLoad | 1.82 |
| Rating | 1.81 |
| Change | 1.60 |

A VIF of 1 means no linear overlap with the other predictors. For Accts, the other seven predictors explain **82.3% of its variation**. Its VIF means that, holding residual variance and its own spread fixed, its coefficient variance is inflated by a factor of 5.64 compared with an orthogonal predictor design. The corresponding standard-error multiplier is $\sqrt{5.64}\approx2.37$.

That makes Accts’ separate association with Sales harder to estimate. Time and MktShare also have appreciable overlap. With only 25 observations and 16 residual degrees of freedom, this matters for coefficient precision.

The full model explains 92.2% of the observed Sales variation, but that fit does not establish accurate future predictions or precise individual coefficients. **I would not delete a predictor solely because of these VIFs.** They support caution when interpreting “holding the other predictors fixed,” especially for Accts and Time; they do not establish causal effects.

[Code](/examples/lecture13-v2/sales_multicollinearity_diagnostics.py) · [Results](/examples/lecture13-v2/sales_multicollinearity_results.json) · [Correlation plot](/examples/lecture13-v2/sales_predictor_overlap.png)

**Human**: Can you show me what information is left for the Accounts effect?

**Instructor**: Recall $\mathrm{VIF}_j=1/(1-R_j^2)$. With Accounts VIF about 5.64, what fraction of its variation remains after adjustment?

<!-- checkpoint: 2. Information for Accounts -->
**Assistant**: **Only 17.7% of the observed variation in Accts remains after accounting for the other seven predictors.** That remaining variation supplies the information for the Accounts coefficient in the full model.

To see it, I fitted Accts using the other predictors and kept its residuals. A positive residual means a territory has a higher Accts value than those predictors would suggest. I also adjusted Sales for those same predictors.

![Accounts added-variable plot](/examples/lecture13-v2/accounts_added_variable_plot.png)

Each point represents one of the 25 observations. The horizontal axis shows the remaining Accts variation; the vertical axis shows the remaining Sales variation. **The fitted line has exactly the same slope as the Accts coefficient in the full eight-predictor regression.**

| Full-model Accounts result | Value |
|---|---:|
| Coefficient | 5.55 |
| Standard error | 4.78 |
| 95% confidence interval | −4.57 to 15.67 |
| $p$-value | 0.262 |

The slope is positive: a one-unit increase in Accts, holding the other predictors fixed, is associated with an estimated increase of 5.55 hundreds of sales units. But the residual points are scattered, and the confidence interval includes both negative and positive slopes.

Accts explains **7.8% of the Sales variation left after adjusting for the other seven predictors**. This is a partial $R^2$, not 7.8% of total Sales variation.

So there is information left, but it gives an imprecise estimate of the separate Accounts association. The result does not establish no association, and it does not identify a causal effect.

[Code](/examples/lecture13-v2/accounts_partial_regression.py) · [Results](/examples/lecture13-v2/accounts_partial_regression_results.json) · [Residual data](/examples/lecture13-v2/accounts_partial_regression_residuals.csv)

**Human**: What if two predictors contain exactly the same information?

**Instructor**: Could two different pairs of coefficients give identical fitted sales when a predictor is duplicated?

<!-- checkpoint: 3. Exact duplication -->
**Assistant**: **Then their separate coefficients cannot be uniquely estimated.** This is perfect multicollinearity: the data can identify a combined association, but cannot tell how to divide it between the two predictors.

For a **constructed demonstration**, I added an exact copy of Accts to the full sales model. Their contribution becomes

$$
\beta_1\,\text{Accts}+\beta_2\,\text{Accts}
=(\beta_1+\beta_2)\,\text{Accts}.
$$

Only the sum matters. These coefficient pairs produce the same fitted sales, up to numerical rounding:

| Coefficient choice | Accts | Constructed copy | Sum |
|---|---:|---:|---:|
| Software’s pseudoinverse solution | 2.775 | 2.775 | 5.551 |
| Equally valid alternative | 12.775 | −7.225 | 5.551 |

The software issued a rank-deficiency warning. Its equal split is a computational convention, not evidence that each predictor has half the association. Adding 10 to one coefficient and subtracting 10 from the other leaves every fitted value unchanged.

In the residual view we just used, adjusting Accts for its exact copy leaves **zero variation**. There is no independent Accounts information left with which to estimate its separate coefficient; its VIF is infinite.

For exact duplicates, keeping one copy preserves all the model’s predictive information and gives an identifiable coefficient. Exact proportional copies have the same problem after allowing for their scaling.

The original sales predictors are not exactly redundant: their design matrix has full rank. Their overlap reduces precision, but does not make the coefficients unidentified.

[Code](/examples/lecture13-v2/exact_overlap_demonstration.py) · [Results](/examples/lecture13-v2/exact_overlap_demonstration_results.json)

**Human**: And if they're almost the same?

**Instructor**: Predict how almost duplicating Accounts will affect its standard error. The added predictor is a constructed experiment.

<!-- checkpoint: 4. Near duplication -->
**Assistant**: **If they are almost the same, their separate coefficients can be estimated, but with large uncertainty.** The model has to use the small difference between them to distinguish their associations with Sales.

I demonstrated this with **constructed near-copies of Accts**, adding fixed random noise at two scales. Each model includes all eight original predictors plus the near-copy; all 25 observations remain.

| Constructed noise SD, relative to Accts SD | Predictor correlation | Accts coefficient | Near-copy coefficient | Accts standard error |
|---|---:|---:|---:|---:|
| 10% | 0.9956 | 28.11 | −22.77 | 25.70 |
| 1% | 0.99996 | 233.08 | −227.75 | 254.67 |

As the predictors become closer, the coefficients grow in opposite directions and their standard errors increase sharply. Yet **their coefficient sum stays at 5.34**, and both constructed models have the same fitted sales, up to rounding.

The algebra explains why. If the near-copy is $x+\epsilon$, then

$$
\beta_1x+\beta_2(x+\epsilon)
=(\beta_1+\beta_2)x+\beta_2\epsilon.
$$

The shared variation estimates the sum. Only the tiny difference $\epsilon$ separates the two coefficients.

This also explains why large, opposing coefficients can coexist with a good fit. Predictions for territories following the observed overlap can remain useful, while predictions where the two predictors diverge may be much less precise.

This is a constructed illustration, not validation of forecasting performance. The actual sales data have milder overlap: Accts’ VIF is 5.64, compared with about 161 and 15,835 in these constructed models.

[Code](/examples/lecture13-v2/near_overlap_demonstration.py) · [Results](/examples/lecture13-v2/near_overlap_demonstration_results.json)

**Human**: Would centering the variables fix this?

**Instructor**: Which quantities should change when zero moves to each predictor's mean?

<!-- checkpoint: 5. Centering -->
**Assistant**: **No. Centering does not fix the overlap in this sales model.** Subtracting each predictor’s mean changes where zero is located, but leaves the relationships among predictors intact.

I refitted the original regression of Sales on all eight predictors, keeping the intercept and all 25 observations:

| Accounts result | Original predictors | Centered predictors |
|---|---:|---:|
| Coefficient | 5.551 | 5.551 |
| Standard error | 4.776 | 4.776 |
| VIF | 5.639 | 5.639 |

All eight slopes, their standard errors, and the fitted sales remain unchanged apart from numerical rounding. Centering also leaves predictor correlations unchanged. Consequently, it cannot create additional independent Accounts information.

What **does** change is the intercept. It moves from −1507.81 to 3374.57 hundreds of sales units. The centered intercept describes fitted Sales when every predictor equals its sample mean, which is usually easier to interpret than setting every original predictor to zero.

For the constructed duplicates we discussed, centering would preserve exact duplication. For near-duplicates, it would preserve their tiny differences and the resulting difficulty separating coefficients.

Centering can help interpretation and numerical computation. In models containing squares or interactions, it can also change correlations among the model’s columns, though it does not add information to an equivalent model. Our current sales regression contains only additive linear terms, so the Accounts uncertainty remains exactly the same.

[Code](/examples/lecture13-v2/centering_overlap_check.py) · [Results](/examples/lecture13-v2/centering_overlap_results.json)

**Human**: What about adding an interaction?

**Instructor**: What question would a Time by Accounts interaction answer?

<!-- checkpoint: 6. An interaction -->
**Assistant**: **An interaction asks a new question; it does not fix overlapping predictors.** It allows the association between Accounts and Sales to depend on another predictor.

As an exploratory example, I added **Time × Accts** to the full sales regression, retaining all eight main effects and all 25 observations. Using centered predictors, the estimated Accounts slope is

$$
6.693-0.01475(\text{Time}-87.642).
$$

Thus, at the sample mean experience of 87.642 months, the estimated Accounts association is 6.693 hundreds of sales units per Accts unit. The interaction estimates how that slope changes for each additional month.

| Interaction-model result | Value |
|---|---:|
| Time × Accts coefficient | −0.01475 |
| 95% confidence interval | −0.06271 to 0.03321 |
| $p$-value | 0.522 |
| Adjusted $R^2$, additive model | 0.8831 |
| Adjusted $R^2$, interaction model | 0.8787 |

**These data provide little evidence for this particular interaction.** Its confidence interval permits both increasing and decreasing Accounts slopes, and adjusted $R^2$ falls after adding it. This exploratory comparison does not validate future predictions or establish causation.

Here centering does help the parameterization: the interaction VIF falls from **22.47 to 3.19**, while fitted sales and the interaction coefficient’s uncertainty remain unchanged. It also makes the Accounts coefficient refer to average experience instead of zero experience.

So centering makes an interaction model easier to interpret, but adding an interaction does not supply the independent predictor variation missing because of overlap.

[Code](/examples/lecture13-v2/accounts_time_interaction.py) · [Results](/examples/lecture13-v2/accounts_time_interaction_results.json)

**Human**: Could we just drop Accounts?

**Instructor**: What would the Time coefficient mean after Accounts is removed?

<!-- checkpoint: 7. Dropping Accounts -->
**Assistant**: **Yes, dropping Accounts is a reasonable simplification to consider, but it changes what the remaining coefficients mean.** I compared the original additive model with all eight predictors against the same model without Accts, retaining all 25 observations.

| Measure | Full model | Without Accts |
|---|---:|---:|
| $R^2$ | 0.9220 | 0.9155 |
| Adjusted $R^2$ | 0.8831 | 0.8806 |
| Residual standard deviation | 449.03 | 453.64 |
| Time coefficient | 2.01 | 3.70 |
| Time standard error | 1.93 | 1.28 |

Removing Accts causes only a modest loss of fit in this sample. The test of its additional contribution gives $F(1,16)=1.35$, $p=0.262$. That provides limited evidence for retaining it, but does not establish that its association is zero. Adjusted $R^2$ still slightly favors the full model.

Overlap falls substantially: all remaining VIFs are below 1.75. However, **the Time coefficient increases from 2.01 to 3.70**. In the full model, it compares territories with the same Accts value and other predictors. Without Accts, that adjustment disappears, allowing Time to capture some of their shared association with Sales. Its smaller standard error therefore accompanies a different interpretation.

For prediction, the reduced model is a plausible candidate, but this comparison measures fit to the observed data, not future accuracy. For estimating the Accounts association, dropping it removes the question entirely. I would not drop it solely to obtain lower VIFs or sharper coefficients.

[Code](/examples/lecture13-v2/drop_accounts_comparison.py) · [Results](/examples/lecture13-v2/drop_accounts_comparison_results.json)

**Human**: Do Time and Accounts tell us anything together?

**Instructor**: Could the pair matter jointly even if neither individual p-value is small? Explain your prediction.

<!-- checkpoint: 8. Joint evidence -->
**Assistant**: **Yes. Time and Accounts jointly add evidence of an association with Sales, even though their separate contributions are difficult to distinguish.**

I compared the original eight-predictor additive model with a model omitting **both Time and Accts**, keeping the other six predictors and all 25 observations. The joint test asks

$$
H_0:\ \beta_{\text{Time}}=\beta_{\text{Accts}}=0.
$$

| Joint comparison | Result |
|---|---:|
| $R^2$ without Time and Accts | 0.8740 |
| $R^2$ with both | 0.9220 |
| Joint $F$ statistic | $F(2,16)=4.93$ |
| $p$-value | 0.0215 |
| Partial $R^2$ for the pair | 0.3811 |

Together, they explain **38.1% of the Sales variation remaining after adjustment for the other six predictors**. This is not 38.1% of total Sales variation.

The joint result can coexist with the individual full-model $p$-values of 0.313 for Time and 0.262 for Accts. Each individual test asks whether one predictor adds information when the other is already included. Because they overlap, either can account for some shared information, making their separate coefficients imprecise. The joint test asks whether removing both loses information, and here it does.

So the data support retaining information represented by this pair, but do not clearly establish that both predictors are needed individually. Our earlier comparison showed that removing Accts alone caused a much smaller loss of fit.

This concerns their **joint additive association**, not a Time × Accounts interaction, causal effects, or validated forecasting performance.

[Code](/examples/lecture13-v2/time_accounts_joint_test.py) · [Results](/examples/lecture13-v2/time_accounts_joint_test_results.json)

**Human**: If a new territory is inside all the observed ranges, is predicting there safe?

**Instructor**: Does checking each predictor's range tell us whether their combination is supported by the observed data?

<!-- checkpoint: 9. A new territory -->
**Assistant**: **No. Being inside every observed range does not guarantee that the combination of predictors is supported by the data.** A territory can be familiar on each variable separately and still be far from the territories observed jointly.

For a constructed illustration, I examined all 256 combinations of predictor minima and maxima and selected the one with the highest prediction leverage. Every value is within its observed range. It combines **Time = 365.04 months with Accts = 74.86**, along with boundary values for the other six predictors. Time and Accts tend to increase together in the observed data, so this combination goes against that pattern.

Using the original eight-predictor additive model:

| Location | Prediction leverage |
|---|---:|
| All predictors at their sample means | 0.040 |
| Highest among observed territories | 0.716 |
| Constructed territory within all ranges | 11.774 |

Prediction leverage measures how much coefficient uncertainty contributes to a prediction. Unlike leverage for an observation used to fit the model, it can exceed 1 at a new location.

At this constructed territory, fitted Sales is **4746.6 hundreds of units**, but the model-based 95% prediction interval for one new territory is **1344.5 to 8148.8**. That wide interval reflects weak support for the predictor combination, even though every separate range check passes.

This is a constructed stress test, not forecasting validation. For an actual new territory, we should examine its joint location and nearby observations, as well as whether the fitted relationship applies there. Individual range checks alone are insufficient.

[Code](/examples/lecture13-v2/within_ranges_prediction_check.py) · [Results](/examples/lecture13-v2/within_ranges_prediction_results.json)

**Human**: How can we tell whether one territory is driving the results?

**Instructor**: Does a large residual necessarily mean a point controls the fit? What other evidence would you inspect?

<!-- checkpoint: 10. Influential observations -->
**Assistant**: **Check both unusual observations and how the conclusions change when each is omitted.** In the full eight-predictor additive model, territory 7 has the greatest overall influence, while territory 12 has the largest influence on the Accounts coefficient.

Three diagnostics answer different questions: **leverage** measures an unusual predictor combination; a **studentized residual** measures an unusual Sales outcome given those predictors; **Cook’s distance** measures their combined influence on the fitted model.

| Territory (CSV row) | Leverage | Studentized residual | Cook’s distance |
|---|---:|---:|---:|
| 7 | 0.716 | −1.07 | 0.317 |
| 12 | 0.502 | 1.43 | 0.215 |
| 10 | 0.219 | 2.22 | 0.123 |

Territory 7 has the maximum observed Time and Accts values: 365.04 months and 256.10. Its influence comes mainly from its predictor position, not an especially large residual.

Leaving out each territory in turn shows:

- Omitting territory 12 changes the Accounts coefficient from **5.55 to 1.30**.
- Omitting territory 7 changes the joint Time–Accounts $p$-value from **0.0215 to 0.0510**.
- Across all omissions, Accounts coefficients range from **1.30 to 8.45**, and joint $p$-values range from **0.0083 to 0.0510**.

Thus, individual coefficient magnitudes are sensitive, and the joint conclusion is close enough to the conventional 0.05 threshold to change with one omission. That threshold crossing does not erase the association, but it argues against a firm binary conclusion.

These are **sensitivity checks, not validation or grounds for deletion**. No observations were removed from the primary analysis. The supplied file has no missing or duplicate rows; verifying these territories’ measurements would require original records.

[Code](/examples/lecture13-v2/territory_influence_check.py) · [Results](/examples/lecture13-v2/territory_influence_results.json) · [Plot](/examples/lecture13-v2/territory_influence_plot.png)

**Instructor**: Reference: Bowerman, O'Connell and Koehler, Forecasting, Time Series, and Regression: An Applied Approach, fourth edition, Example 4.2 (pp. 146–147) and Section 5.1 (pp. 222–226). The worked example uses the course sales-territory data. Constructed predictor changes and new territory profiles are labeled in the conversation.

[Download the data](/examples/lecture13-v2/sales_territory.csv) · [Python package versions](/examples/lecture13-v2/requirements.txt)
