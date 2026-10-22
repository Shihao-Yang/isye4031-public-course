# Title: Lecture 14: Lasso diagnostics and Elastic Net

**Instructor**: This replay presents a completed Codex CLI run. Human prompts were simulated for teaching; the assistant chose and executed the analysis. Its responses are preserved, with portable asset links. Instructor notes and discussion pauses were added separately. Playback shows saved calculations.

By the end, explain the Lasso penalty, check a fitted model, interpret selection stability, and explain how Elastic Net combines L1 and L2. Predict or discuss briefly at each checkpoint before revealing the response.

This is a new, explicitly constructed sales example with its own generation code. The topic sequence draws on [Jiecheng Lu's shrinkage labs](https://ljc-fvnr.github.io/isye4031-summer2026-labs/); no lab data or notebook files are redistributed. The example is separate from earlier sales-territory work.

**Human**: Remind us of the sales prediction question. Show the data and visualize the relationships.

**Instructor**: What is the prediction target? Which visible relationship might make individual coefficients difficult to interpret?

<!-- checkpoint: 1. Visual setup -->
**Assistant**: Our question is: **Can we predict a store’s weekly sales from its advertising, size, staffing, and local market characteristics?** We also want to see which inputs help when several carry similar information.

I created an original synthetic dataset of **240 fictional stores**, with weekly sales in dollars and 12 candidate predictors. A fixed random split assigns 168 stores to training and reserves 72 for a final comparison. Everything shown here uses only training data.

Here are five training rows, with selected columns:

| Store | Digital ads ($/week) | Floor area (sq. ft.) | Demand index | Weekly sales ($) |
|---|---:|---:|---:|---:|
| 1 | 1,143 | 1,944 | 39.7 | 29,669 |
| 2 | 1,302 | 870 | 54.9 | 19,083 |
| 5 | 1,363 | 2,109 | 32.4 | 21,885 |
| 6 | 1,238 | 1,980 | 65.9 | 23,189 |
| 7 | 873 | 3,121 | 48.7 | 35,814 |

![Relationships in the constructed sales training data](/examples/lecture14-v2/constructed_sales_training_relationships.png)

Larger stores tend to have higher sales: floor area has a correlation of **0.74** with sales. Local demand and digital advertising also show positive associations, at **0.43** and **0.34**. The arbitrary survey-noise input shows little relationship.

The correlation matrix reveals overlapping information within the advertising, store-size, and demand groups. Consequently, a predictor’s individual relationship with sales may not tell us how much it adds once other predictors enter a model. These are descriptive associations in constructed data; we have not fitted a prediction model or measured its accuracy.

[Dataset](/examples/lecture14-v2/constructed_sales_seed4031.csv) · [Dictionary](/examples/lecture14-v2/sales_data_dictionary.md) · [Generation code](/examples/lecture14-v2/generate_sales_data.py) · [Analysis code](/examples/lecture14-v2/explore_sales_training.py) · [Results](/examples/lecture14-v2/sales_training_exploration_results.json)

**Human**: Why might Lasso help here? Show us how its penalty works.

**Instructor**: Why might we accept a worse training fit? Explain what the penalty changes before revealing the calculation.

<!-- checkpoint: 2. Lasso penalty -->
**Assistant**: **Lasso could help because our sales predictors contain overlapping information and irrelevant inputs.** It shrinks coefficients and can set some exactly to zero, potentially reducing prediction variability and simplifying the model.

Lasso minimizes

$$
\frac{1}{2n}\sum_{i=1}^{n}
\left(y_i-\beta_0-\sum_j z_{ij}\beta_j\right)^2
+\lambda\sum_j|\beta_j|.
$$

The first term rewards fitting sales closely. The second charges for coefficient size. Here, $z_{ij}$ are predictors standardized using the training means and standard deviations, so different measurement units do not determine the penalty. Sales are expressed in thousands of dollars. The intercept $\beta_0$ is unpenalized.

At $\lambda=0$, this is ordinary least squares. Increasing $\lambda$ makes larger coefficients more costly. The absolute-value penalty has a corner at zero, allowing coefficients to become exactly zero.

![Lasso penalty and training coefficient paths](/examples/lecture14-v2/lasso_penalty_and_training_paths.png)

For our **constructed sales data**, these fits use the same 168 training stores:

| Penalty $\lambda$ | Nonzero coefficients | $\sum_j|\beta_j|$ |
|---|---:|---:|
| 0: ordinary least squares | 12 | 11.44 |
| 0.1: Lasso | 9 | 9.18 |
| 0.5: Lasso | 7 | 7.27 |
| 1.0: Lasso | 5 | 5.84 |

These are **penalty sensitivity results**, not evidence that prediction improves. Stronger penalties sacrifice training fit, and correlated predictors can compete for selection. A zero coefficient therefore does not establish that a variable is unimportant or has no causal effect. We would choose the penalty using training-only cross-validation; the reserved test set remains unused.

[Code](/examples/lecture14-v2/demonstrate_lasso_penalty.py) · [Results](/examples/lecture14-v2/lasso_penalty_demonstration_results.json)

**Human**: How does Ridge differ from Lasso? Show us the coefficient paths.

**Instructor**: Predict the model at a very large penalty. Does every individual coefficient have to shrink monotonically?

<!-- checkpoint: 3. Lasso and Ridge paths -->
**Assistant**: **Ridge shrinks coefficients toward zero; Lasso can also set them exactly to zero.** Their difference is the penalty:

$$
\begin{aligned}
\text{Lasso:}\quad&
\frac{\mathrm{SSE}}{2n}+\lambda\sum_j|\beta_j|,\\
\text{Ridge:}\quad&
\frac{\mathrm{SSE}}{2n}+\frac{\lambda}{2}\sum_j\beta_j^2.
\end{aligned}
$$

Both leave the intercept unpenalized. Ridge’s squared penalty charges increasingly for large coefficients. Its smooth shape generally keeps coefficients nonzero at finite penalties. Lasso’s absolute-value penalty has a corner at zero, which permits exact zeros.

![Lasso and Ridge coefficient paths on constructed training data](/examples/lecture14-v2/ridge_lasso_training_paths.png)

Each colored line follows one predictor’s coefficient across penalty strengths. Both plots use the same **168 training stores from our constructed dataset**, standardized predictors, and sales measured in thousands of dollars.

As the penalty increases, Lasso’s paths reach zero. Here, all coefficients become zero at $\lambda\approx4.81$, leaving an intercept-only prediction. Ridge’s coefficients approach zero gradually. Individual paths can bend or change direction because correlated predictors redistribute their contributions.

For these sales predictors, Ridge can spread weight across related measures such as floor area, staffing, and shelf capacity. Lasso can produce a smaller selected set, although which member of a correlated group survives may be sensitive to the sample.

The horizontal ranges differ to make both patterns visible. **Equal $\lambda$ values do not imply equal shrinkage across methods.** These are training coefficient paths and penalty sensitivity results, not validation evidence that either method predicts better. The reserved test set remains unused.

[Code](/examples/lecture14-v2/compare_ridge_lasso_paths.py) · [Results](/examples/lecture14-v2/ridge_lasso_path_results.json)

**Human**: How should we choose the Lasso penalty? Show the validation evidence.

**Instructor**: Which observations may influence scaling and the choice of penalty? Sketch the training, validation and test boundary.

<!-- checkpoint: 4. Training-only tuning -->
**Assistant**: **Choose the Lasso penalty using cross-validation prediction error.** For our constructed sales data, the lowest mean validation error occurs at **$\lambda\approx0.093$**.

I divided the 168 training stores into five fixed folds and evaluated 121 penalty values. For each fold, I fitted standardization and Lasso using the other four folds, then predicted the held-out stores. Thus, each validation store was excluded from both scaling and coefficient fitting.

![Five-fold validation evidence for choosing the Lasso penalty](/examples/lecture14-v2/lasso_fivefold_validation_evidence.png)

| Choice | $\lambda$ | Mean fold RMSE | Nonzero coefficients after training refit |
|---|---:|---:|---:|
| Minimum CV error | 0.093 | $3,455 | 9 |
| One-standard-error alternative | 0.584 | $3,590 | 6 |

For context, predicting each training fold’s mean sales gives a mean validation RMSE of **$6,523**.

Small penalties allow more flexibility; large penalties eventually remove useful sales information. The curve shows the tradeoff. I would use **0.093** when selecting by minimum cross-validation error.

The **one-standard-error rule** offers a simpler alternative: choose the largest penalty whose error lies within one estimated standard error of the minimum. Here, it retains six predictors, with somewhat higher validation error. This rule is a simplicity heuristic, not proof that the two choices perform equally.

The shaded band describes variability across folds; overlapping training sets prevent interpreting it as a formal confidence interval. Also, we selected the minimum using these same results, so its reported error can be optimistic for the tuned procedure. The **72 reserved test stores remain unused**.

[Code](/examples/lecture14-v2/select_lasso_penalty_cv.py) · [Results](/examples/lecture14-v2/lasso_fivefold_validation_results.json) · [Validation curve data](/examples/lecture14-v2/lasso_fivefold_validation_curve.csv)

**Human**: After fitting Lasso, how do we diagnose problems? Show us the checks.

**Instructor**: Constructed claim for discussion: “The residuals average zero, so this Lasso model is adequate.” What evidence would you request?

<!-- checkpoint: 5. Post-Lasso diagnostics -->
**Assistant**: **Check whether prediction errors show systematic patterns, changing spread, or unusually large misses.** Here, the tuned Lasso shows no strong curvature or widening error spread, although some individual predictions miss by nearly $9,000.

These checks use **nested out-of-fold predictions** for the 168 constructed training stores. Within each outer training fold, I retuned the penalty using five-fold cross-validation, including scaling inside every fitting fold. Each plotted store was excluded from tuning and fitting for its prediction.

![Diagnostic checks for the tuned Lasso procedure](/examples/lecture14-v2/lasso_nested_cv_diagnostic_checks.png)

A residual is observed sales minus predicted sales:

- **Top left:** Curvature would suggest missing structure. The prediction-quartile averages stay fairly close to zero; overall mean residual is **−$2**. That average alone could hide offsetting errors.
- **Top right:** A widening funnel would suggest increasing error variability. There is no clear funnel here.
- **Bottom left:** The Q–Q plot is approximately straight, with some tail departures. This is a descriptive check, not proof of normal errors. Lasso does not require normality to make predictions.
- **Bottom right:** Predictions broadly track sales. The nested out-of-fold **RMSE is $3,471**, and mean absolute error is **$2,808**.

I investigated the largest miss, store 157. Its observed sales were **$22,150**, versus a prediction of **$31,015**. The known synthetic mechanism assigned it approximately **−$8,949** in random noise. Its digital advertising also falls outside that fold’s training range. The row was retained.

These diagnose the tuning procedure, not the single full-training fit. Ordinary least-squares t-tests and residual degrees of freedom do not automatically apply after Lasso selection. The reserved test set remains unused.

[Code](/examples/lecture14-v2/diagnose_tuned_lasso_nested_cv.py) · [Results](/examples/lecture14-v2/lasso_nested_cv_diagnostic_results.json) · [Large-error investigation](/examples/lecture14-v2/lasso_large_error_investigation.csv)

**Instructor**: With an unpenalized intercept, the fitted training residuals average approximately zero because of the fitting criterion. This does not establish a correct mean function, constant variance or independence. Diagnostic patterns motivate further checks; a clean plot does not prove adequacy. Ordinary OLS coefficient tests do not automatically apply after shrinkage and selection. These fictional stores are independent; repeated store-week data would also require checks for dependence and validation that respects groups or time.

**Human**: Would Lasso keep the same variables in another sample? Does zero mean no effect?

**Instructor**: If a coefficient is zero, what can we conclude? Would stable selection establish a causal effect?

<!-- checkpoint: 6. Selection stability -->
**Assistant**: **Lasso may select different variables in another sample, and a zero coefficient does not mean “no effect.”** Selection depends on the sample, penalty, and which other predictors are available.

I tested stability using **60 overlapping subsamples of 134 training stores**, retuning the penalty with five-fold cross-validation each time. Scaling stayed inside the fitting folds.

![Lasso selection frequencies across training subsamples](/examples/lecture14-v2/lasso_selection_stability_clear_legend.png)

Some selections were consistent, while others varied:

- Digital advertising, promotion, floor area, and local demand were selected in **60/60** fits.
- Print advertising was selected in **23/60**.
- Staff hours was selected in only **5/60**, even though it has a nonzero coefficient in our known synthetic equation.

Staff hours shares information with floor area and shelf capacity. Lasso can suppress its coefficient while using related predictors. **Zero means this particular penalized fit assigns no additional linear contribution to that variable.** It does not establish absence of association, predictive information under another model, or a causal effect.

The reverse also matters: display color and survey noise were selected in **60/60** fits, despite being independent noise inputs in the generating mechanism. Overlapping subsamples can preserve accidental relationships already present in this finite dataset. Consistent selection therefore does not prove a real signal.

This is **subsampling sensitivity**, not independent new samples or validation. The frequencies describe stability within these training data; they are not probabilities that effects exist. The reserved test set remains unused.

[Code](/examples/lecture14-v2/lasso_selection_subsample_stability.py) · [Results](/examples/lecture14-v2/lasso_subsample_stability_results.json) · [Selection frequencies](/examples/lecture14-v2/lasso_subsample_selection_frequencies.csv)

**Human**: Can we combine L1 and L2? Show how Elastic Net works.

**Instructor**: How might a penalty keep useful information from correlated inputs while still permitting a sparse model?

<!-- checkpoint: 7. Elastic Net -->
**Assistant**: **Yes. Elastic Net combines Lasso’s variable selection with Ridge’s coefficient shrinkage.** It minimizes

$$
\frac{\mathrm{SSE}}{2n}
+\lambda\rho\sum_j|\beta_j|
+\frac{\lambda(1-\rho)}{2}\sum_j\beta_j^2.
$$

Here, $\lambda$ controls overall penalty strength, while $\rho$ controls the mix:

- $\rho=1$ gives Lasso.
- $\rho=0$ gives Ridge.
- $0<\rho<1$ combines both.

The intercept remains unpenalized, and predictors are standardized. The L1 term permits exact zeros; the L2 term encourages correlated predictors to share weight. That can reduce competition among related sales measures, although improved selection stability must be checked separately.

![Elastic Net coefficient paths and joint tuning](/examples/lecture14-v2/elastic_net_paths_and_joint_tuning.png)

The **left panel** shows coefficient paths with a fixed mix, $\rho=0.5$. Increasing $\lambda$ shrinks coefficients and eventually sets them to zero. This is coefficient sensitivity, not validation.

The **right panel** tunes both parameters using the same five training folds as before, with scaling fitted inside each fold:

| Model | Selected $\lambda$ | Selected $\rho$ | Mean validation RMSE |
|---|---:|---:|---:|
| Lasso | 0.093 | 1 | $3,455 |
| Elastic Net | 0.093 | 0.90 | $3,449 |

For these **constructed sales data**, the grid favors mostly L1 with a small L2 component. The roughly **$5 difference** provides little practical reason to declare a winner. These errors guided tuning; they do not independently evaluate the selected procedures.

Elastic Net still does not turn selected coefficients into causal effects or guarantee removal of noise inputs. The 72 reserved test stores remain unused.

[Code](/examples/lecture14-v2/demonstrate_elastic_net.py) · [Results](/examples/lecture14-v2/elastic_net_demonstration_results.json) · [CV data](/examples/lecture14-v2/elastic_net_joint_cv_results.csv)

**Human**: How does tuned Ridge compare, and which choice should we freeze?

**Instructor**: Elastic Net has two tuning parameters. What must change in the validation workflow, and what must remain untouched?

<!-- checkpoint: 8. Freezing the comparison -->
**Assistant**: **Tuned Ridge performs similarly, but I would freeze Lasso at $\lambda=0.0926$.** It gives a smaller model with nearly the same cross-validation error as Elastic Net.

All three methods used the same five training folds, with scaling fitted separately inside each fold. Ridge’s penalty follows $\mathrm{SSE}/(2n)+\lambda\sum_j\beta_j^2/2$.

| Tuned model | $\lambda$ | L1 fraction $\rho$ | Mean validation RMSE | Nonzero coefficients* |
|---|---:|---:|---:|---:|
| Ridge | 0.0501 | 0 | $3,480 | 12 |
| Lasso | 0.0926 | 1 | $3,455 | 9 |
| Elastic Net | 0.0926 | 0.90 | $3,449 | 10 |

*After refitting on all 168 training stores.*

![Comparison of tuned models on the same training folds](/examples/lecture14-v2/tuned_models_training_cv_comparison.png)

Elastic Net has the smallest tuning error, but its advantage over Lasso is about **$5 per store-week**. Ridge trails Lasso by about **$26**. These differences are small relative to the variation across folds. That supports a practical preference for Lasso’s smaller model; it does not establish statistical equivalence.

I have **frozen Lasso as the primary choice**, including its penalty, predictor list, training standardization, and fitted coefficients. This choice targets prediction, not recovery of the true variables: our stability checks showed that Lasso can omit a direct equation term and retain noise inputs.

I also saved the three fixed training fits for one eventual final comparison. The **72 reserved test stores remain unused**. Cross-validation guided these choices, so its selected minima are not independent final performance estimates.

[Code](/examples/lecture14-v2/tune_ridge_same_training_folds.py) · [Comparison results](/examples/lecture14-v2/ridge_lasso_enet_training_comparison.json) · [Frozen decision](/examples/lecture14-v2/frozen_sales_model_decision.json) · [Freeze code](/examples/lecture14-v2/freeze_sales_model_choices.py)

**Human**: Now evaluate the frozen models on the reserved stores. What should we conclude?

**Instructor**: Predict the comparison before revealing it. What difference would be large enough to change your decision?

<!-- checkpoint: 9. Final prediction comparison -->
**Assistant**: **The frozen Lasso achieved a test RMSE of $2,947 on the 72 reserved stores. All three models performed closely, and we retain Lasso as the preselected primary model.**

| Frozen model | Test RMSE | Test MAE |
|---|---:|---:|
| **Lasso** | **$2,947** | **$2,326** |
| Ridge | $2,993 | $2,354 |
| Elastic Net | $2,957 | $2,338 |

I loaded the saved training fits and verified their file hashes. No coefficients, penalties, or standardization parameters were updated during evaluation.

![Final comparison on the reserved synthetic stores](/examples/lecture14-v2/frozen_sales_final_test_comparison.png)

For Lasso, the average absolute prediction error was about **$2,326 per store-week**. RMSE is higher because it gives larger misses more weight. Lasso also overpredicted sales by about **$330 on average**, although that average can conceal offsetting errors.

Lasso’s RMSE was approximately **$11 lower than Elastic Net’s** and **$47 lower than Ridge’s**. Those are small observed differences. One holdout of 72 stores does not establish a reliable population ranking or prove that the methods are equivalent.

The practical conclusion is that **our frozen Lasso predicts new stores from this constructed population with an RMSE of roughly $3,000**. Its smaller fitted model remains a reasonable choice. That predictive performance does not establish that its selected variables are the true drivers of sales, and it provides no causal evidence.

The test set has now served its final evaluation role. Using these outcomes to revise the model would require fresh independent data for another final assessment.

[Code](/examples/lecture14-v2/evaluate_frozen_sales_test_once.py) · [Results](/examples/lecture14-v2/frozen_sales_final_test_results.json) · [Predictions](/examples/lecture14-v2/frozen_sales_final_test_predictions.csv)

**Instructor**: Explain your final choice in your own words. State the evidence, one limitation, and what you would check next. These interpretation questions do not specify future assessment coverage.

**Instructor**: Additional executed reproduction code: [plot lasso stability clear legend](/examples/lecture14-v2/plot_lasso_stability_clear_legend.py)

**Instructor**: To reproduce the run, use a fresh folder with the data, split and Python files, install the listed environment versions, and run the scripts in conversation order. Run the model-freezing script before the final evaluation script; it creates the local fitted-model file. Do not copy the saved final-test results into that folder, because the evaluation script deliberately refuses to overwrite them.

**Instructor**: [Environment versions](/examples/lecture14-v2/requirements.txt) · [Fixed split](/examples/lecture14-v2/fixed_sales_split.csv) · [Synthetic generation details](/examples/lecture14-v2/sales_generation_metadata.json). All data and code in this package were generated for this example.

