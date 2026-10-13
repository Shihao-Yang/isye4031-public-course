# Title: Lecture 13 | Regression pitfalls and multicollinearity

**Instructor**: This is a simulated instructor-agent conversation adapted from Shihao's legacy Regression Pitfalls slides. The agent responses were recorded during real Python execution; the calculations are already complete during playback. Instructor notes are added separately. Constructed experiments are labeled.

By the end, explain what identifies a conditional regression coefficient, calculate and interpret a VIF, distinguish nonunique estimates from imprecise estimates, and assess a proposed remedy or new prediction using evidence.

We continue with the same 25 sales territories as Lecture 12. The textbook identifies Sales as annual sales in hundreds of units, Time as months with the company, and Advertising as dollar expenditure. Random sampling in this observational example does not establish causal effects.

**Human**: Let's continue with regression pitfalls using the same sales-territory example. My old slides ask us to look at the predictor scatterplots and correlation matrix before interpreting coefficients. Fit Sales on all eight predictors with an intercept and compute the correlations and VIFs. Show the three largest VIFs alongside the coefficients, standard errors and p-values for those predictors, plus a readable Time-versus-Accts plot. What can this tell us about overlap among predictors? Does the software returning an answer mean every coefficient is well determined? Stop after the diagnosis. Please write and execute the code.

**Instructor**: Before revealing the results, predict which pair of predictors is likely to overlap. What evidence would distinguish a computable model from precise individual coefficients?

<!-- checkpoint: 1. Diagnose predictor overlap -->
**Assistant**: I fit ordinary least squares with an intercept and all eight predictors to the 25 sales territories. I computed Pearson correlations and each predictor’s VIF by regressing it on the other seven predictors with an intercept:
$$\mathrm{VIF}_j=\frac{1}{1-R_j^2}.$$

The three largest VIFs in this **full model** are:

| Predictor | VIF | Coefficient | Standard error | Two-sided $p$ |
|---|---:|---:|---:|---:|
| Accts | 5.639 | 5.551 | 4.776 | 0.2621 |
| Time | 3.343 | 2.010 | 1.931 | 0.3134 |
| MktShare | 3.236 | 199.024 | 67.028 | 0.0090 |

Coefficients describe changes in fitted Sales, measured in hundreds of units, per unit of the named predictor, holding all other predictors fixed.

![Time versus Accts for the observed sales territories](/examples/lecture13/time_vs_accts.png)

Time and Accts have a Pearson correlation of **0.758**. The plot shows their positive association: territories with longer company tenure tend to have larger Accts values. Consequently, the model has limited information for separating their individual associations with Sales while holding the other predictors fixed.

Pairwise correlations reveal overlap between pairs. VIF captures overlap with a combination of all other predictors, so a modest pairwise correlation does not rule out multicollinearity. The Accts VIF indicates substantial variance inflation under the classical OLS assumptions. Its standard error, like Time’s, is large relative to its estimate. MktShare also has overlap, yet its estimate is more precise relative to its magnitude.

**Software returning coefficients does not mean every coefficient is well determined.** The design matrix has full rank, so the fitted coefficients are unique, but uniqueness does not ensure precision. These standard errors and exact finite-sample $t$ tests assume a correctly specified linear mean and independent, homoskedastic, normally distributed errors. Nonsignificance does not establish no association, and these observational coefficients do not identify causal interventions.

[Runnable code](/examples/lecture13/diagnosis.py) · [Numerical results](/examples/lecture13/diagnosis_results.json) · [Correlation matrix](/examples/lecture13/correlations.csv) · [All VIFs and predictor coefficients](/examples/lecture13/vif_coefficients.csv)

**Human**: Accounts has VIF 5.639, even though it is below the slide's cutoff of 10. Walk me through that one calculation. Regress Accts on the other seven predictors, show the resulting R-squared, and calculate VIF and the corresponding standard-error inflation factor by hand from those numbers. What is being held fixed in that comparison? Show a plot of the Accounts variation left after adjustment, then residualize Sales on the same seven predictors and check that the residual-versus-residual slope equals the full-model Accounts coefficient. I want to understand what information is left for that coefficient. Does VIF below 10 settle whether it is precise enough for our purpose? Run the calculations and stop there.

**Instructor**: Use $\mathrm{VIF}=1/(1-R_j^2)$. If the other predictors explain about 82% of Accounts variation, roughly how much variation remains? Explain what the Accounts coefficient compares before continuing.

<!-- checkpoint: 2. What information identifies the Accounts slope? -->
**Assistant**: I regressed Accts on an intercept plus Time, MktPoten, Adver, MktShare, Change, WkLoad, and Rating. This auxiliary regression gives $R^2=0.822674$: those predictors explain about 82.27% of the observed, mean-centered Accts variation.

The calculation is:
$$
1-R^2=0.177326,\qquad
\mathrm{VIF}=\frac{1}{0.177326}=5.639317,
$$
$$
\text{standard-error inflation factor}
=\sqrt{5.639317}=2.374725.
$$

**What is held fixed?** Under a correctly specified linear model with independent, homoskedastic errors, this compares coefficient variance at the same error variance and the same total centered Accts variation. The hypothetical reference makes Accts orthogonal to the other predictors. It is not a comparison with simply dropping those predictors, which can change the error variance and coefficient meaning.

![Accts variation remaining after adjustment](/examples/lecture13/accounts_remaining_variation.png)

The Accts residual is observed Accts minus its fitted value from the other seven predictors. Its sum of squares is **8,840.882**, compared with **49,856.539** before adjustment. Thus, only 17.73% of the original variation remains to distinguish Accts from the other predictors.

I also residualized Sales using the same intercept and seven predictors, then regressed its residuals on the Accts residuals through the origin:

| Calculation | Accts slope |
|---|---:|
| Residual versus residual regression | 5.550961 |
| Original full model | 5.550961 |

They agree to numerical precision, as the Frisch–Waugh–Lovell identity requires.

![Residualized Sales versus residualized Accts](/examples/lecture13/sales_accounts_residuals.png)

For inference, I retained the full model’s error variance estimate and residual degrees of freedom. Its Accts standard error is **4.775550**, with a classical 95% confidence interval of **[−4.573, 15.675]**; exact coverage additionally assumes normal errors.

**VIF below 10 does not settle adequacy.** Precision depends on remaining predictor variation, error variance, and the accuracy needed for the decision. This interval permits substantially different conditional associations. Without a purpose-specific tolerance, the cutoff cannot establish sufficient precision or causal meaning.

[Runnable code](/examples/lecture13/accounts_adjustment.py) · [Numerical results](/examples/lecture13/accounts_adjustment_results.json) · [Residual data](/examples/lecture13/accounts_residuals.csv)

**Human**: So the real Accounts coefficient is unique, but its uncertainty is large. My slides also warn that very high collinearity can make a model impossible to estimate or make coefficients change wildly. Can we separate those two situations? Make a clearly labeled constructed stress test: add an exact copy of Accts to the full model, and then a separate near-copy made with small fixed-seed noise (seed 4031; noise standard deviation 1% of the sample standard deviation of Accts). Compare rank, uniqueness and the two Accounts coefficients. Then increase only the first Sales value by 1% of the sample standard deviation of Sales, refit the near-copy model, and compare both coefficients and fitted values on the original 25 rows. Show a readable plot and numerical changes. Keep the original data untouched. Is this sensitivity experiment evidence of good prediction on new territories? Please run it.

**Instructor**: If Accounts appears twice, can the data distinguish two separate slopes? Predict what could change in the coefficients while leaving the fitted values unchanged. The added columns and response perturbation are constructed teaching experiments.

<!-- checkpoint: 3. Exact and near collinearity -->
**Assistant**: **This constructed stress test separates nonuniqueness from sensitivity.** I kept the original CSV unchanged and fitted each model with an intercept.

The exact copy equals Accts. The separate near-copy equals Accts plus fixed-seed normal noise, using seed 4031 and noise standard deviation **0.455780**, which is 1% of the sample standard deviation of Accts.

| Model | Rank / columns | Unique coefficients? | Accts coefficient | Copy coefficient |
|---|---:|---|---:|---:|
| Original full model | 9 / 9 | Yes | 5.550961 | N/A |
| Exact copy added | 9 / 10 | No | 2.775480 | 2.775480 |
| Near-copy added | 10 / 10 | Yes | −141.424791 | 147.369867 |
| Near-copy, Sales perturbed | 10 / 10 | Yes | −140.410049 | 146.316103 |

With the **exact copy**, only the sum of the two Accounts coefficients is identified. Adding any amount to one and subtracting it from the other leaves fitted values unchanged. Software returned an equal split using its pseudoinverse solution and warned about rank deficiency. That split is not uniquely determined by the data.

With the **near-copy**, the coefficients are unique but large and opposing. Their separation depends on the small constructed differences between the columns.

I then increased only the first Sales value by **13.130681 hundreds of units**, or 1% of Sales’ sample standard deviation, holding every predictor fixed.

| Change after refitting near-copy model | Value |
|---|---:|
| Accts coefficient | +1.014742 |
| Near-copy coefficient | −1.053765 |
| Maximum absolute fitted-value change | 7.011888 |
| RMS fitted-value change across original rows | 1.919071 |

![Fitted-value changes on the original territories](/examples/lecture13/stress_fitted_changes.png)

The opposing coefficient changes largely cancel on these rows. This particular perturbation demonstrates sensitivity, but does not produce an enormous coefficient swing.

**It provides no evidence of good prediction on new territories.** It is an in-sample sensitivity experiment; no new outcomes or independent validation were used.

[Runnable code](/examples/lecture13/collinearity_stress.py) · [Numerical results](/examples/lecture13/stress_results.json) · [Row-level results](/examples/lecture13/stress_row_results.csv)

**Human**: The exact-copy result makes sense: two separate coefficients cannot be identified. The near-copy fit has large opposing coefficients, but this particular perturbation did not create an enormous swing, so let's keep that finding as it is. What should we actually do about collinearity? My old slides mention adding an interaction, removing a predictor, or combining predictors. Return to the unmodified eight-predictor data. Test whether centering and scaling the original predictors changes their VIFs or fitted values; add a centered Time-by-Accts interaction and recompute the primary-predictor VIFs; then remove Accts and compare the Time coefficient, standard error and p-value with the original full model. Explain which operations just change coordinates and which change the model or its question. Would removing Accounts establish that it has no useful association? Would adding an interaction be a general remedy? Keep lasso and a full ridge lesson for later, but explain briefly when collecting more independent predictor variation or regularization could help. Execute the comparisons.

**Instructor**: Decide which operation changes only units and which changes the model: centering, rescaling, adding an interaction, or removing Accounts. What would you need to justify the model change?

<!-- checkpoint: 4. Check the proposed remedies -->
**Assistant**: I executed all comparisons on the **unmodified data**, retaining an intercept.

Centering, and then scaling by each predictor’s sample standard deviation, left VIFs and fitted values unchanged to numerical precision. Maximum VIF differences were below $1.4\times10^{-15}$; maximum fitted-value differences were below $1.3\times10^{-9}$ hundreds of units. These operations **change coordinates**: centering changes the intercept’s reference point, and scaling changes coefficient units. They do not remove predictor overlap.

Next, I added $(Time-\overline{Time})(Accts-\overline{Accts})$ while retaining all eight centered predictors:

| Predictor | Original VIF | With interaction |
|---|---:|---:|
| Time | 3.343 | 4.156 |
| MktPoten | 1.978 | 2.190 |
| Adver | 1.910 | 2.430 |
| MktShare | 3.236 | 3.281 |
| Change | 1.602 | 1.602 |
| Accts | 5.639 | 6.363 |
| WkLoad | 1.818 | 1.932 |
| Rating | 1.809 | 2.127 |

Adding this interaction **changes the model**: the Time association can vary with Accts, and conversely. Its Time main coefficient now describes the slope at mean Accts. An interaction is not a general collinearity remedy; here, primary-predictor VIFs increased.

Removing Accts produced:

| Model | Time coefficient | Standard error | $p$-value |
|---|---:|---:|---:|
| Original full model | 2.0096 | 1.9307 | 0.3134 |
| Without Accts | 3.7007 | 1.2822 | 0.0103 |

These classical tests assume correct linear means and independent, homoskedastic, normally distributed errors. The reduced model no longer holds Accts fixed, so its Time coefficient answers a different question. **Removing Accts does not establish that it lacks useful association.**

Combining predictors also changes the question unless the transformation preserves all information. Collecting observations with more independent predictor variation can help separate associations, provided sampling and measurement remain suitable. Regularization can stabilize estimates by accepting bias and imposing constraints, but prediction benefits require validation. Neither approach alone establishes causal effects.

[Runnable code](/examples/lecture13/collinearity_options.py) · [Numerical results](/examples/lecture13/options_results.json) · [VIF comparisons](/examples/lecture13/options_vifs.csv) · [Fitted values](/examples/lecture13/options_fitted_values.csv)

**Human**: Now let's follow my slides from multicollinearity to hidden extrapolation. Suppose someone says, "Every predictor is within its observed minimum and maximum, so this prediction is safe." Construct two hypothetical profiles for the ORIGINAL eight-predictor full model: one at all predictor means, and one with Time at its observed minimum and Accts at its observed maximum, keeping the other six at their means. These are teaching examples, not observed new territories. Plot the two profiles against the observed Time-Accts cloud. Compute h0 for both profiles and compare with the largest training leverage; show their fitted Sales and classical 95% prediction intervals for individual new territories. Explain why the rectangle of separate ranges is insufficient. Is h0 below the training maximum a guarantee of adequate joint support or valid prediction? Does the wider interval account for all possible model failure outside the observed cloud? Write and run the calculation.

**Instructor**: Picture a new representative with very short tenure and the largest Accounts value. Each number is in range. Is that combination supported by the observed cloud? Explain your reasoning before seeing the plot.

<!-- checkpoint: 5. Hidden extrapolation -->
**Assistant**: I fitted the **original eight-predictor model with an intercept** and constructed two teaching profiles. Neither represents an observed new territory. Both satisfy every predictor’s separate minimum–maximum range.

For each profile, I computed
$$
h_0=x_0^\mathsf{T}(X^\mathsf{T}X)^{-1}x_0,
$$
where $x_0$ includes the intercept. This measures the profile’s position relative to the full predictor design, not just Time and Accts.

| Constructed profile | Time (months) | Accts | $h_0$ | Fitted Sales | 95% individual prediction interval |
|---|---:|---:|---:|---:|---:|
| All predictor means | 87.642 | 122.699 | 0.0400 | 3374.57 | [2403.82, 4345.31] |
| Minimum Time, maximum Accts | 8.130 | 256.100 | 2.9008 | 3955.29 | [2075.26, 5835.31] |

The other six predictors remain at their means. Sales and interval endpoints are in **hundreds of units**. The largest training leverage is **0.7163**, at territory row **7**. The second profile’s $h_0$ substantially exceeds it.

![Constructed profiles against observed Time–Accts combinations](/examples/lecture13/hypothetical_time_accts.png)

The rectangle of separate ranges permits combinations that the data do not contain. Here, low Time paired with high Accts falls far from the observed positive association. Marginal range checks miss this joint extrapolation; the plot displays only part of the full predictor geometry.

The classical individual prediction interval is
$$
\widehat{Sales}_0\ \pm\ t_{0.975,16}\,s\sqrt{1+h_0}.
$$
It assumes the linear mean remains correct at the profile, independent normal errors with constant variance, an independent new territory’s error, and accurately measured predictors. The $1$ accounts for individual outcome noise; $h_0$ accounts for fitted-mean uncertainty.

**Being below the largest training leverage would not guarantee adequate joint support or valid prediction.** Leverage summarizes global geometry and can miss local gaps or unsupported combinations. A training maximum is not a safety threshold.

The wider interval also does **not** cover every possible model failure. Curvature, changed error variance, measurement problems, or a different territory population can invalidate its coverage.

[Runnable code](/examples/lecture13/hidden_extrapolation.py) · [Numerical results](/examples/lecture13/hidden_extrapolation_results.json) · [Profiles and predictions](/examples/lecture13/hypothetical_profiles.csv) · [Training leverages](/examples/lecture13/training_leverages.csv)

**Human**: Before accepting that model, we should also check the other pitfalls in my slides. For the ORIGINAL eight-predictor model, inspect residuals versus fitted values and compare leverage, externally studentized residuals and Cook's distance. Show two clear plots and a small table for the two rows with largest leverage and the row with largest absolute studentized residual; label row numbers as positions in the CSV. Does large leverage necessarily mean large influence? Which records would you investigate, and what information about sampling, measurements and error assumptions is still missing? Don't delete a row just because of a cutoff. Run a sensitivity refit excluding the largest-Cook-distance row, explicitly as a sensitivity check, and report how the Time and Accounts coefficients change. End with the next investigation, not a declaration that assumptions passed.

**Instructor**: For the original full model, explain how an unusual predictor profile differs from an unusual residual. Predict whether the points with the largest leverage must have the largest influence.

<!-- checkpoint: 6. Leverage, residuals and influence -->
**Assistant**: I computed residuals, leverage, externally studentized residuals, and Cook’s distance for the **original eight-predictor model with an intercept**. Row numbers below are **positions in the CSV, excluding its header**.

![Residuals versus fitted Sales](/examples/lecture13/residuals_vs_fitted.png)

The residual plot has substantial scatter and several large deviations. It does not establish constant variance or a correct linear mean; with this small dataset, curvature and changing spread remain uncertain.

![Leverage and externally studentized residuals, with Cook’s distance](/examples/lecture13/leverage_residual_cook.png)

| CSV row position | Reason selected | Leverage | External studentized residual | Cook’s distance |
|---|---|---:|---:|---:|
| 7 | Largest leverage | 0.7163 | −1.0682 | 0.3172 |
| 17 | Second-largest leverage | 0.5615 | −0.2158 | 0.0070 |
| 10 | Largest absolute studentized residual | 0.2186 | 2.2178 | 0.1228 |

Leverage measures unusual predictor combinations. External studentization scales a residual using an error variance estimate obtained without that row. Cook’s distance measures the overall fitted-model change associated with omitting a row.

**Large leverage does not necessarily mean large influence.** Row 17 has high leverage but a small residual and small Cook’s distance. Row 7 has the largest Cook’s distance.

As a **sensitivity check only**, I excluded row 7 and refitted:

| Predictor | Original coefficient | Sensitivity coefficient | Change |
|---|---:|---:|---:|
| Time | 2.0096 | 3.1712 | +1.1616 |
| Accts | 5.5510 | 7.6849 | +2.1339 |

This demonstrates dependence on that record, not grounds for deleting it. Classical studentized-residual inference assumes a correct linear mean and independent, homoskedastic normal errors.

I would investigate rows 7, 17, and 10. The source describes random selection among effective representatives, but leaves the selection process, territory dependence, measurement timing and accuracy, and applicability to future territories unresolved.

[Runnable code](/examples/lecture13/influence_diagnosis.py) · [Numerical results](/examples/lecture13/influence_results.json) · [All row diagnostics](/examples/lecture13/influence_all_rows.csv)

The next investigation is to verify those records against source measurements and collection details, then examine residuals against individual predictors and any documented time or territory grouping.

**Human**: One last question before the next step. The full-model Time and Accounts p-values were both fairly large. My slides say overlapping predictors can be jointly useful even when individual tests look weak. Test Time and Accts TOGETHER in the original model, keeping the other six predictors and the intercept in both models. Compute the reduced and full SSEs, give the partial F calculation with its degrees of freedom, and compare its p-value with the two individual p-values. Explain the different null hypotheses and the exploratory nature of this comparison after looking at the data. Then make a provisional recommendation: what can we say about these two associations, what prediction or causal claim should we withhold, and what evidence should we get next? Keep it open-ended. Leave lasso and a full regularization demonstration for the next lesson.

**Instructor**: Time and Accounts each have a large individual p-value. Does that mean we could remove both without losing information? State the joint null hypothesis, predict what the comparison could reveal, and identify one next step that would still be needed.

<!-- checkpoint: 7. Joint evidence and the next decision -->
**Assistant**: The **joint test provides stronger evidence than either individual test** in the original full model. Both models retain the intercept, MktPoten, Adver, MktShare, Change, WkLoad, and Rating; the full model additionally includes Time and Accts.

| Model | SSE | Residual degrees of freedom |
|---|---:|---:|
| Reduced: neither Time nor Accts | 5,212,679.763 | 18 |
| Original full model | 3,225,984.675 | 16 |

SSE is the sum of squared Sales residuals, in squared hundreds-of-units. The partial F calculation is
$$
F=\frac{(SSE_R-SSE_F)/2}{SSE_F/16}
=\frac{1,986,695.088/2}{3,225,984.675/16}
=4.9267.
$$
Using an $F_{2,16}$ distribution gives **$p=0.02152$**.

| Test in the original full model | Null hypothesis | $p$-value |
|---|---|---:|
| Individual Time | $\beta_{Time}=0$, allowing Accts | 0.31341 |
| Individual Accts | $\beta_{Accts}=0$, allowing Time | 0.26213 |
| Joint | $\beta_{Time}=\beta_{Accts}=0$ | 0.02152 |

The individual tests ask whether each predictor adds information after accounting for the other. The joint test asks whether both can be omitted together. Overlap can make their separate contributions difficult to distinguish even when their combined contribution is detectable. The joint result does not establish that both coefficients are nonzero.

Exact F and t inference assumes a correct linear conditional mean and independent, homoskedastic normal errors. Because we chose this comparison after examining correlations, coefficients, and diagnostics, its p-value is **exploratory**, not confirmation from a prespecified analysis.

Provisionally, I would keep Time and Accts under consideration as a pair: this model shows evidence of a combined conditional association, with uncertain separation. We should withhold claims of accurate prediction on new territories and causal effects of changing tenure or accounts.

[Runnable code](/examples/lecture13/joint_time_accounts.py) · [Numerical results](/examples/lecture13/joint_test_results.json) · [Model comparison](/examples/lecture13/joint_test_model_comparison.csv)

Next, verify influential records and measurement timing, clarify sampling and dependence, and obtain independent outcomes from relevant territories, ideally with more distinct Time–Accts combinations. Those findings can guide whether retaining, simplifying, or regularizing the pair serves the intended decision.

**Instructor**: The human prompts follow legacy L5.1, especially pages 6-16 on estimability, multicollinearity and extrapolation, with a short return to diagnostic checks. Textbook reference: Bowerman, O'Connell and Koehler, Forecasting, Time Series, and Regression: An Applied Approach, fourth edition, Example 4.2 (pp. 146-147) and Section 5.1 (pp. 222-226). Numerical results and figures were recomputed from the course data. Open-ended pauses practice interpreting and checking analysis. Lasso and a full regularization lesson remain a continuation.

[Download the data](/examples/lecture13/sales_territory.csv) · [Python package versions](/examples/lecture13/requirements.txt)
