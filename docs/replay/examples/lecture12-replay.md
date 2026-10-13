# Title: Lecture 12 | Sales-territory model selection

**Instructor**: This is a recorded Codex CLI rehearsal prepared on October 8, 2026. The orchestrator wrote short human prompts using Shihao's lecture conversation as a guide; Codex produced the answers during recorded analysis. Human prompts and final answers are preserved, with file links adjusted for this local player. Instructor notes are added separately and labeled. Computation is already complete during replay.

By the end, you should be able to interpret an adjusted coefficient, perform a model-selection step, explain what validation must exclude, investigate a diagnostic flag, and choose a useful follow-up when an AI claim exceeds the evidence.

The example has 25 territories and eight predictors. The source does not document units, timing or collection design. Use recorded sales units. Predictions and causal claims need evidence beyond this file.

**Human**: Let's use this sales-territory example. We have Sales and eight predictors. Can we fit the whole multiple linear regression first? Before fitting, check the data and tell me what question this model can answer. Show me the equation and the important output, but let's stop there for now.

**Instructor**: Before revealing the answer, name the response, the unit represented by one row, and one thing this observational regression cannot establish. Take 30 seconds.

<!-- checkpoint: 1. Question, data and full regression -->
**Assistant**: The data contain **25 territories**, Sales, and eight numeric predictors. There are no missing or nonfinite values, duplicate rows, or constant columns. The design matrix has full rank, so all nine coefficients, including the intercept, can be estimated. Sales ranges from 1,578.00 to 6,519.45 in recorded units; substantive plausibility cannot be verified without measurement definitions.

This model asks: **How is mean Sales associated with each predictor, holding the other seven fixed?** It does not establish causal effects. Prediction for new territories would require validation and confirmation that these predictors are available when predictions are made.

We fit ordinary least squares:
$$
Sales_i=\beta_0+\sum_{j=1}^{8}\beta_jX_{ij}+\varepsilon_i.
$$

The fitted equation is:
$$
\begin{aligned}
\widehat{Sales}={}&-1507.814+2.010\,Time+0.037205\,MktPoten\\
&+0.150989\,Adver+199.024\,MktShare\\
&+290.855\,Change+5.551\,Accts\\
&+19.794\,WkLoad+8.189\,Rating.
\end{aligned}
$$

| Term | Standard error | $p$-value |
|---|---:|---:|
| Intercept | 778.635 | 0.0707 |
| Time | 1.931 | 0.3134 |
| MktPoten | 0.008202 | 0.000338 |
| Adver | 0.047109 | 0.00552 |
| MktShare | 67.028 | 0.00904 |
| Change | 186.782 | 0.1390 |
| Accts | 4.776 | 0.2621 |
| WkLoad | 33.677 | 0.5649 |
| Rating | 128.506 | 0.9500 |

**Overall fit:** $R^2=0.9220$, adjusted $R^2=0.8831$, residual standard error $=449.03$, and 16 residual degrees of freedom. The overall test of eight zero slopes gives $F(8,16)=23.654$, $p=1.82\times10^{-7}$.

These are conventional OLS inference results, conditional on assumptions we have not checked. The next decision is whether diagnostics support interpreting this model; we stop before that step.

[Runnable code](/examples/lecture12/full_regression.py) · [Coefficients and confidence intervals](/examples/lecture12/full_regression_coefficients.csv) · [Data summary](/examples/lecture12/data_summary.csv)

**Human**: Visualize it. I want to see how the full model fits and why accounts can look strongly related to sales by itself but be nonsignificant after we include the other predictors. Show the relevant evidence, and explain what we should ask next.

**Instructor**: Accounts looks useful on its own. If its full-model p-value exceeds 0.10, does that show no association with sales? Predict what evidence the next answer should show. Take 45 seconds.

<!-- checkpoint: 2. Marginal and adjusted associations -->
**Assistant**: ![Full model fit and the Accounts relationship before and after adjustment](/examples/lecture12/full_model_visualization.png)

**The full model fits these 25 territories closely**, with $R^2=0.922$. The left panel compares observed Sales with fitted Sales; perfect predictions fall on the dashed line. Training RMSE is 359.2 recorded Sales units. This measures fit to the data used for estimation, not performance on new territories. The earlier residual standard error, 449.0, uses 16 residual degrees of freedom instead of 25.

The middle and right panels answer different questions:

| Accounts association | Slope | Standard error | 95% confidence interval | $p$ |
|---|---:|---:|---:|---:|
| Alone: `Sales ~ Accts` | 21.72 | 3.95 | [13.56, 29.88] | 0.0000135 |
| Full eight-predictor model | 5.55 | 4.78 | [−4.57, 15.67] | 0.262 |

The right panel removes the linear associations with the other seven predictors from **both** Sales and Accounts. Its fitted slope equals the Accounts coefficient in the full model.

Accounts shares substantial information with those predictors: collectively, they explain **82.3% of its variation**, giving an Accounts variance inflation factor of **5.64**. After adjustment, the slope shrinks and its uncertainty increases. Adding Accounts to the other seven increases $R^2$ only from 0.9155 to 0.9220. Nonsignificance does not establish no association or no causal effect.

**What should we ask next?** Do residual patterns or influential territories undermine the full model? If prediction is the goal, does retaining Accounts improve validated prediction performance? Its $p$-value alone cannot settle that decision.

[Runnable code](/examples/lecture12/visualize_full_model.py) · [Numerical evidence](/examples/lecture12/visualization_evidence.json)

**Human**: We have many non-significant predictors. What can we do? Try backward elimination with a removal threshold of 0.10. Show each removal and the refitted p-values. Why not drop all the non-significant predictors at once?

**Instructor**: From the full-model table, choose the first term to remove at 0.10. Should all terms with p above 0.10 leave together? Explain your prediction before advancing. Take 60 seconds.

<!-- checkpoint: 3. Backward elimination -->
**Assistant**: Using backward elimination at **0.10**, we remove **Rating, then WkLoad, then Accts**. At each step, we remove only the predictor with the largest $p>0.10$ and refit. The intercept stays in every model.

Each column below reports the p-values from a freshly fitted model:

| Predictor | Full model | Remove Rating | Remove WkLoad | Remove Accts |
|---|---:|---:|---:|---:|
| Time | 0.3134 | 0.2869 | 0.1983 | **0.0065** |
| MktPoten | 0.000338 | 0.000186 | 0.000079 | 0.000005 |
| Adver | 0.0055 | 0.0027 | 0.0018 | 0.0025 |
| MktShare | 0.0090 | 0.0066 | 0.000360 | 0.000003 |
| Change | 0.1390 | **0.0897** | 0.0927 | **0.0530** |
| Accts | 0.2621 | 0.2339 | **0.2881** | Removed |
| WkLoad | 0.5649 | **0.5501** | Removed | Removed |
| Rating | **0.9500** | Removed | Removed | Removed |

We stop with:
$$
\begin{aligned}
\widehat{Sales}={}&-1113.788+3.612\,Time\\
&+0.042088\,MktPoten+0.128857\,Adver\\
&+256.956\,MktShare+324.533\,Change.
\end{aligned}
$$

The final model has $R^2=0.9150$ and adjusted $R^2=0.8926$, compared with 0.9220 and 0.8831 for the full model.

**Why not drop everything nonsignificant at once?** Each p-value measures an association conditional on the predictors currently included. Removing variables changes both coefficients and standard errors. Here, simultaneous deletion would discard Time and Change, although both meet the threshold after sequential refitting. Shared information between Accounts and Time helps explain Time’s change.

This produces a candidate model, not proof that the retained predictors are the uniquely correct set. Selection on the same data makes ordinary final p-values too optimistic as confirmatory evidence. Next, we should examine diagnostics and, for prediction, validate the entire selection procedure.

[Code](/examples/lecture12/backward_elimination.py) · [All refitted p-values, including intercepts](/examples/lecture12/backward_elimination_pvalues.csv)

**Human**: Now try forward selection with an entry threshold of 0.10. Show the path and the stopping decision. Do we get the same model? Also show one step by hand: adding advertising after accounts, using the partial F calculation. Can we use that F test to compare the final forward and backward models?

**Instructor**: Will forward selection necessarily reach the backward model? For one added predictor, recall how the partial F statistic compares the SSE reduction with the remaining residual variance. Take 60 seconds.

<!-- checkpoint: 4. Forward selection and a manual check -->
**Assistant**: **No, forward selection gives a different model.** Start with an intercept only. At each step, test every excluded predictor conditional on those already selected, and add the one with the smallest $p<0.10$. Previously entered predictors stay.

| Step | Best candidate | Partial $F$ | Entry $p$ | Decision |
|---|---|---:|---:|---|
| 1 | Accts | 30.3018 | 0.0000135 | Add |
| 2 | Adver | 20.2105 | 0.000180 | Add |
| 3 | MktPoten | 6.4144 | 0.01936 | Add |
| 4 | MktShare | 14.6111 | 0.001065 | Add |
| 5 | Change | 2.5958 | 0.12363 | Stop |

At step 5, even the smallest candidate $p$ exceeds 0.10, so none qualifies.

| Procedure | Final predictors |
|---|---|
| Forward | Accts, Adver, MktPoten, MktShare |
| Backward | Time, MktPoten, Adver, MktShare, Change |

For the hand calculation, compare reduced model `Sales ~ Accts` with expanded model `Sales ~ Accts + Adver`. Both include an intercept. Their residual sums of squares are 17,855,474.735 and 9,306,228.875. Adding advertising uses one parameter; the expanded model has $25-3=22$ residual degrees of freedom:

$$
F=\frac{(SSE_R-SSE_E)/(3-2)}{SSE_E/(25-3)}
=\frac{17{,}855{,}474.735-9{,}306{,}228.875}
{9{,}306{,}228.875/22}
=20.2105.
$$

The nominal $F_{1,22}$ tail probability is $0.000180$, so advertising enters.

**This partial F test cannot directly compare the final models:** they are not nested. Forward includes Accts; backward excludes it and includes Time and Change. Selection also limits confirmatory interpretation of these nominal p-values. Predictive comparison would require repeating each selection procedure within training data only.

[Reproducible code](/examples/lecture12/forward_selection.py) · [All candidate tests](/examples/lecture12/forward_candidate_tests.csv) · [Numeric results](/examples/lecture12/forward_results.json)

**Human**: Which model predicts better? Let's check on each territory while holding it out. Compare the full model and the forward and backward procedures. I could select variables using all 25 rows and then cross-validate that formula. Is that a fair check of model selection? Show both calculations so we can see what goes wrong. Make a plot.

**Instructor**: Predict whether variable selection must be repeated using only the 24 training territories. Which response would otherwise influence selection? Commit to an answer before advancing. Take 60 seconds.

<!-- checkpoint: 5. Validation of the entire procedure -->
**Assistant**: **Forward selection predicts best in this leave-one-territory-out comparison.** The full model and backward procedure have almost identical RMSE.

For each territory, I held out its row, repeated selection at threshold 0.10 using only the other 24 rows, fitted coefficients there, and predicted the held-out Sales. All models include an intercept.

$$
e_i=Sales_i-\widehat{Sales}_{(-i)},\qquad
RMSE=\sqrt{\frac{1}{25}\sum_{i=1}^{25}e_i^2},
\qquad MAE=\frac{1}{25}\sum_{i=1}^{25}|e_i|.
$$

Errors below are in recorded Sales units.

| Model or procedure | Proper LOOCV RMSE | Proper LOOCV MAE | Select on all 25, then CV: RMSE | Select on all 25, then CV: MAE |
|---|---:|---:|---:|---:|
| Full eight predictors | 549.6 | 464.3 | 549.6 | 464.3 |
| Forward | **506.7** | **423.3** | 481.8 | 388.9 |
| Backward | 549.7 | 452.1 | **465.7** | 390.3 |

**Selecting on all 25 rows first is not a fair check of model selection.** Although coefficients are refitted on 24 rows, the held-out outcome already influenced which predictors enter the formula. Here, that leakage lowers reported RMSE by 24.9 for forward and 84.0 for backward, reversing their ranking.

Selection actually changes across folds: forward returns its original formula in 22 of 25 folds; backward does so in 20. The full formula has no selection step, so its two calculations agree.

![LOOCV comparison and held-out errors](/examples/lecture12/loo_comparison.png)

These results favor forward provisionally. With only 25 territories and undocumented sampling, they do not establish superiority for future territories. Checks passed without computational errors.

[Code](/examples/lecture12/loo_comparison.py) · [Every held-out prediction and error](/examples/lecture12/loo_predictions.csv) · [Metrics](/examples/lecture12/loo_metrics.csv) · [Selected formulas](/examples/lecture12/loo_selected_formulas.csv)

**Human**: Use the forward model as our working example for now. Run diagnostics and show me the residual, Q-Q, leverage and Cook-distance plots. Which territory should we investigate? Compare the fit with and without it. Does crossing a diagnostic threshold tell us to delete that row? Keep the model being diagnosed clear.

**Instructor**: Before the diagnostic answer, distinguish an unusual predictor combination, a large conditional residual, and influence on the fitted model. Would a screening threshold alone justify deletion? Take 60 seconds.

<!-- checkpoint: 6. Diagnostics and sensitivity -->
**Assistant**: We are diagnosing the **fixed forward-selected model**, fitted to all 25 territories:
$$
Sales=\beta_0+\beta_1Accts+\beta_2Adver+\beta_3MktPoten+\beta_4MktShare+\varepsilon.
$$
The sensitivity refit keeps these same predictors; selection is not repeated.

![Forward-model diagnostics](/examples/lecture12/forward_diagnostics.png)

The residual smoother suggests possible curvature, although 25 observations provide limited evidence. The Q-Q plot is approximately straight. Neither plot establishes that the assumptions hold; independence also depends on the undocumented collection design.

**Investigate territory 1 first**, numbering territories by CSV row order. Its leverage is $0.495$, above $2p/n=0.40$, and its Cook’s distance is $0.184$, the largest and the only value above $4/n=0.16$. Here $p=5$ includes the intercept. Its externally studentized residual is only $0.97$: influence comes mainly from its predictor configuration. Territory 7 also has high leverage; territory 10 has the largest studentized residual, $2.38$.

| Quantity | All 25 | Without territory 1 |
|---|---:|---:|
| Intercept | −1441.932 | −1451.917 |
| Accts coefficient | 9.21390 | 10.65073 |
| Adver coefficient | 0.174990 | 0.174050 |
| MktPoten coefficient | 0.038218 | 0.032695 |
| MktShare coefficient | 190.144 | 193.796 |
| $R^2$ | 0.90045 | 0.90491 |
| Residual standard error | 453.84 | 454.57 |

Accounts and market-potential coefficients shift noticeably, but all slope signs remain unchanged. Territory 1’s prediction falls from 3357.63 to 3051.04 when excluded; observed Sales is 3669.88. The two $R^2$ values use different datasets and do not establish improvement.

**Thresholds prompt investigation, not deletion.** Check source accuracy and whether the territory belongs to the intended population. Retain a valid unusual row and report sensitivity. No computational errors occurred.

[Code](/examples/lecture12/forward_diagnostics.py) · [Diagnostics](/examples/lecture12/forward_diagnostics.csv) · [Coefficient comparison](/examples/lecture12/forward_sensitivity_coefficients.csv)

**Human**: Let's test our judgment. Here is a claim I constructed for teaching: "All eight predictors give the largest R-squared, so that is our best prediction model. A nonsignificant normality test confirms the assumptions. Delete any territory above the Cook-distance guide, and use the significant coefficients to decide how to increase sales."

Which follow-up is best, and why?
A. Accept the claim because the software completed successfully.
B. Accept the claim after removing flagged territories.
C. Compare validation of the entire procedure, investigate flagged records and sensitivity, and check the collection design before making causal claims.
D. Keep only predictors with small p-values and skip other checks.

Explain the answer and why the other choices fail.

**Instructor**: Choose A, B, C or D from the human prompt and explain one reason. This misleading claim was constructed for this activity. It is not an error attributed to the recorded model. Take 90 seconds. This excerpt-and-judgment activity follows the verified recent practice and administered midterm approach; it does not promise future exam coverage.

<!-- checkpoint: 7. Independent assessment judgment -->
**Assistant**: **C is best.** It checks prediction, model sensitivity, and whether the data support the intended interpretation.

The constructed claim makes four mistakes:

- **Largest training $R^2$ does not mean best prediction.** Adding predictors cannot decrease ordinary training $R^2$. In our leave-one-territory-out check, with selection repeated within each training set, forward selection had RMSE **506.7**, compared with **549.6** for the full model and **549.7** for backward selection. This provisionally favors forward, without establishing a universal winner.
- **A nonsignificant normality test does not confirm assumptions.** It means the test did not detect departure from normality. With 25 observations, detection may be weak. It also does not establish linearity, constant error variance, or independence. We have not run a normality test here.
- **Cook’s distance is an investigation guide, not a deletion rule.** For the fixed four-predictor forward model, territory 1 crossed the guide. We should check its accuracy and examine sensitivity while retaining it if valid.
- **Significant coefficients do not identify interventions that increase sales.** They describe conditional associations. Undocumented timing, measurement, sampling, and possible confounding prevent causal conclusions.

**A fails** because successful computation establishes neither statistical validity nor suitability for the question.

**B fails** because automatic deletion can distort the sample and leaves the prediction and causal errors unresolved.

**D fails** because small p-values neither guarantee predictive performance nor replace diagnostics or design checks. Data-driven selection also complicates ordinary coefficient inference.

**Human**: What would you recommend doing next? Give me a provisional model decision, what the students should now be able to explain without AI, and the next human questions before anyone uses this for a new territory. Keep prediction, association, and causation separate. We can leave lasso for a separate follow-up chapter.

**Instructor**: Write one provisional decision and one missing piece of evidence that would change it. Then compare your next question with the recorded answer. Take 60 seconds.

<!-- checkpoint: 8. Human decision and next questions -->
**Assistant**: **Provisionally use the forward procedure for prediction**, with the current working formula
$$
Sales\sim Accts+Adver+MktPoten+MktShare
$$
including an intercept. Its properly evaluated leave-one-territory-out RMSE was **506.7 recorded Sales units**, versus 549.6 for the full model and 549.7 for backward selection. This is preliminary evidence from 25 territories. Retain valid unusual rows, investigate territory 1, and seek independent validation before operational use.

Students should now be able to explain without AI:

- Why adding predictors raises training $R^2$ without guaranteeing better prediction.
- How forward and backward selection work, why their paths differ, and how to calculate a partial F test for nested models.
- Why selection must occur inside each validation training set.
- How residuals, Q-Q plots, leverage, and Cook’s distance answer different questions, and why diagnostic guides do not mandate deletion.
- Why conditional association, statistical significance, and causal effects are distinct.

Before predicting for a new territory, ask:

- **Prediction:** What exactly are Sales and each predictor, over what period? Will all predictors be available at prediction time? Is the new territory comparable in predictor range and collection conditions? What error is acceptable, and what prediction interval would support the decision?
- **Association:** Are measurements accurate and territories independent? Does the apparent curvature warrant revising the formula? How stable are conditional associations under reasonable sensitivity checks?
- **Causation:** What action is being considered? Were predictors measured before Sales, and what confounding or reverse causation is plausible? Could a randomized intervention or credible causal design evaluate that action?

Keep lasso for the next chapter.

