# Title: Lecture 17: Nonlinear regression, PCA and model comparison

**Instructor**: This replay shows a completed Codex CLI conversation. Human prompts were simulated for teaching. The assistant chose and executed the analyses; its wording is preserved with portable file links. Instructor notes and discussion pauses were added separately. Playback displays saved computations.

**Instructor**: By the end, you should be able to do the following independently: Distinguish curvature from nonlinearity in parameters; compute a PCA score; explain what explained predictor variance means; choose a complete predictive procedure without leakage; distinguish regression predictions, logistic probabilities and SVM scores. Pause at each question before revealing the response.

**Human**: Remind us of the store-sales question and show us the data.

**Instructor**: Before revealing the data, identify the response, the observational unit and what a new prediction would be used for. What would make random validation reasonable here?

<!-- checkpoint: 1. The sales question -->
**Assistant**: The store-sales question is: **Can we predict a store’s monthly dollar sales from its advertising spending, floor area, staffing, neighborhood income, and nearby competition?**

Each row represents one store. The response is continuous monthly sales, so this is a regression problem. The goal is to predict sales for a new store, with several characteristics considered together.

These are **300 constructed stores**, not observed business records. I generated them with seed `403117`. The construction includes correlated inputs: larger stores tend to advertise more and employ more staff. It also includes a moderately curved advertising relationship and random variation, so stores with similar characteristics can have different sales.

Before examining sales, I reserved **75 stores for final testing**. We will work with the other **225 training stores** for now.

Here are six training observations. Advertising and sales are in **thousands of dollars per month**; income is in **thousands of dollars per year**.

| Store | Area (sq ft) | Advertising | Staff | Area income | Competitors | Sales |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 3,534 | 52.0 | 13 | 62.7 | 7 | 360.7 |
| 3 | 3,378 | 34.3 | 11 | 72.7 | 0 | 371.5 |
| 4 | 2,484 | 30.7 | 13 | 68.8 | 6 | 378.4 |
| 6 | 3,169 | 32.1 | 10 | 56.3 | 4 | 312.9 |
| 7 | 2,105 | 24.1 | 10 | 46.1 | 3 | 197.6 |
| 8 | 1,873 | 18.7 | 9 | 43.1 | 2 | 210.1 |

![Monthly sales versus advertising for the 225 training stores](/examples/regression-forecasting-v1/store_sales_01_training_scatter.png)

The plot shows an association between advertising and sales, but advertising alone does not determine sales. Store size and the other predictors also vary. **This is a prediction question; the scatterplot does not establish the causal effect of increasing advertising.**

Files: [full dataset](/examples/regression-forecasting-v1/store_sales_01_data.csv), [generation and plotting code](/examples/regression-forecasting-v1/store_sales_01_create_and_display.py), and [construction details and split record](/examples/regression-forecasting-v1/store_sales_01_design.json).

**Instructor**: These are independent constructed stores. Random folds are appropriate for this teaching mechanism; repeated stores or time-dependent deployment would require a different split.

**Human**: Would allowing a curved advertising effect improve our regression? Show us what changes.

**Instructor**: Predict what adding a squared advertising term must do to training error. Does the same guarantee hold for validation error?

<!-- checkpoint: 2. Curved regression -->
**Assistant**: **Allowing curvature helps slightly in these data.** The improvement in validation prediction error is small, even though the constructed sales relationship includes curvature.

We kept store size, staffing, area income, and competition in both models. The only change was the advertising term. Writing \(A\) for monthly advertising in thousands of dollars:

\[
\begin{aligned}
\text{Linear:}\quad &\beta_A(A-28),\\
\text{Quadratic:}\quad &\beta_A(A-28)+\beta_{A^2}(A-28)^2.
\end{aligned}
\]

Centering advertising at 28 makes the coefficients easier to interpret; it does not change which quadratic curves the model can fit. This is still **linear regression in the coefficients**, with one additional predictor.

Using the same five validation folds within the 225 training stores:

| Model | Training RMSE | Validation RMSE |
|---|---:|---:|
| Linear advertising | $34,447 | $35,690 |
| Quadratic advertising | $34,074 | $35,271 |

Validation RMSE combines predictions made when each store was excluded from fitting. It falls by about **$419 per store, or 1.17%**. The quadratic model improves in all five folds, but the gains are modest. The 75 reserved test stores remain untouched.

![Linear and quadratic advertising relationships, with validation residuals](/examples/regression-forecasting-v1/store_sales_02_advertising_curvature.png)

The left panel holds the other predictors at their training medians. The fitted quadratic advertising contribution is

\[
1.485(A-28)-0.0345(A-28)^2.
\]

The negative squared coefficient lets the curve flatten as advertising increases. Consequently, an extra $1,000 of advertising has a different fitted association with sales at different spending levels. These are conditional associations, not causal effects.

The right panel groups validation residuals by advertising spending. Curvature changes their pattern, although substantial prediction error remains.

**The useful distinction is between fitting better and predicting better.** Adding the squared term must improve or preserve the training fit in ordinary least squares. Its validation improvement is the evidence supporting it here, and that evidence favors a small gain.

[Code](/examples/regression-forecasting-v1/store_sales_02_advertising_curvature.py) · [Validation results](/examples/regression-forecasting-v1/store_sales_02_curvature_metrics.csv) · [Saved shared folds](/examples/regression-forecasting-v1/store_sales_02_shared_five_folds.csv)

**Instructor**: A nested least-squares model cannot have larger training SSE after adding a column. Validation error can increase. A polynomial is nonlinear in advertising and linear in its coefficients. For the recorded fit, at advertising A=28 (thousand dollars), the local slope is 1.485 thousand sales dollars per thousand advertising dollars. At A=38 it is 1.485 - 2(0.0345)(10) = 0.795. This is an association, not an intervention estimate.

**Human**: Several inputs move together. Can PCA summarize them? Show us how a component score is calculated.

**Instructor**: Which inputs would dominate PCA if we used their original units? Before revealing the calculation, explain the difference between a loading and a store score.

<!-- checkpoint: 3. PCA scores and loadings -->
**Assistant**: **Yes. PCA summarizes much of the shared movement, although one component does not capture everything.** Using the five original inputs from the 225 training stores, the first component captures **44.2%** of their standardized variation. The first two together capture **66.0%**, and the first three capture **83.8%**.

![Individual and cumulative variance explained by PCA](/examples/regression-forecasting-v1/store_sales_03_pca_variance.png)

We first standardize each input because square feet, dollars, and employee counts have different scales:

\[
z_j=\frac{x_j-\text{training mean}_j}{\text{training standard deviation}_j}.
\]

PCA then finds a weighted sum with the largest possible variance, subject to the weights having squared sum one. Our first component is approximately

\[
\begin{aligned}
\mathrm{PC1}={}&0.620z_{\text{area}}
+0.523z_{\text{advertising}}
+0.563z_{\text{staff}}\\
&+0.029z_{\text{income}}
-0.158z_{\text{competitors}}.
\end{aligned}
\]

The large positive weights on area, advertising, and staffing make this mainly a **store size and spending component**. Its sign is arbitrary: reversing every weight would describe the same axis.

For **store 2**, the area calculation is

\[
z_{\text{area}}
=\frac{3534.12-2500.94}{710.45}
=1.454.
\]

Applying the same calculation to every input gives:

| Input | Standardized value | PC1 weight | Contribution |
|---|---:|---:|---:|
| Store area | 1.454 | 0.620 | 0.901 |
| Advertising | 2.330 | 0.523 | 1.219 |
| Staff | 1.063 | 0.563 | 0.598 |
| Area income | −0.088 | 0.029 | −0.003 |
| Competitors | 2.248 | −0.158 | −0.356 |

The component score is the sum of these contributions. Using unrounded values,

\[
\boxed{\mathrm{PC1}_{\text{store 2}}=2.361.}
\]

That score places this store on the positive side of the component, largely because its area, advertising, and staffing exceed their training means. The score is a dimensionless coordinate, not predicted sales.

**PCA summarizes inputs without using sales.** A direction that explains substantial input variation need not be the best direction for predicting sales. This calculation leaves the test stores untouched; predictive validation would refit both standardization and PCA inside each training fold.

[Code](/examples/regression-forecasting-v1/store_sales_03_pca_component_scores.py) · [Score calculation](/examples/regression-forecasting-v1/store_sales_03_score_calculation.csv) · [Variance results](/examples/regression-forecasting-v1/store_sales_03_explained_variance.csv)

**Instructor**: A loading is an axis weight; a score is a particular store’s coordinate on that axis. Standardization gives each input equal marginal variance before PCA, but whether equal weighting is scientifically sensible still needs judgment. Constructed claim: “83.8% explained predictor variance means 83.8% of sales is explained.” Identify the error. Answer: sales did not enter this PCA, so the percentage describes the standardized predictors only.

**Human**: Does using principal components predict sales better than Ridge, Lasso, or our curved fit? How should we choose?

**Instructor**: If PCA uses no sales values, is it safe to fit PCA on all rows before cross-validation? Explain which rows may determine scaling, loadings and the component count.

<!-- checkpoint: 4. Choosing predictive complexity -->
**Assistant**: **Principal component regression does not predict better here. All four methods perform almost identically, so I would keep the curved ordinary least squares fit as our provisional choice.**

For a fair comparison, every method received the same six candidate predictors: the five original inputs plus squared, centered advertising. Thus PCA now summarizes six columns, extending our earlier five-input illustration.

We used the saved five validation folds. Within each fold’s training portion, another five-fold comparison selected the number of principal components or the Ridge/Lasso penalty. Scaling and PCA were fitted inside those training folds. This **nested validation** keeps tuning separate from the observations used to assess prediction.

| Method | Validation RMSE |
|---|---:|
| Curved ordinary least squares | $35,271 |
| Principal component regression (PCR) | $35,271 |
| Ridge | $35,251 |
| Lasso | $35,277 |

Ridge has the lowest RMSE, but its advantage over the curved fit is only about **$20**, against a typical prediction error of roughly **$35,000**. That gives little practical reason to prefer it.

![Validation errors for the four methods on the same five folds](/examples/regression-forecasting-v1/store_sales_04_nested_validation_comparison.png)

PCR selected **all six components in every outer fold**. Keeping every component simply rotates the predictor space: ordinary least squares then gives the same predictions as fitting the original six columns. Here, validation did not favor dimension reduction.

This illustrates an important limitation of PCA. It finds directions that explain **input variation**, without looking at sales. A component with little input variation can still carry useful predictive information.

The methods address correlated inputs differently:

- **PCR** discards entire component directions when fewer components are retained.
- **Ridge** shrinks coefficients continuously while retaining every predictor.
- **Lasso** can set coefficients to zero. Its full-training selected fit retained all six here.

We should choose the feature set and tuning settings using training validation, then consider prediction error alongside interpretability and simplicity. **Correlation alone does not establish that PCA or penalization will improve prediction.** With these nearly tied results, the curved fit is a reasonable choice. The 75 reserved test stores remain untouched for final evaluation after we settle the model.

[Code](/examples/regression-forecasting-v1/store_sales_04_pcr_ridge_lasso_comparison.py) · [Results](/examples/regression-forecasting-v1/store_sales_04_nested_validation_results.csv) · [Tuning and comparison design](/examples/regression-forecasting-v1/store_sales_04_comparison_design.json)

**Instructor**: Within each fold, fit scaling and PCA on that fold’s training rows. Use validation outcomes to choose predictive complexity. After model selection, performance estimated on the same folds can be optimistic; the reserved test is needed for the final comparison. PCA and PCR remain linear transformations and fits unless nonlinear features are explicitly supplied. The nested comparison conditions on the candidate feature set. We already used development results to consider curvature, so this is not a fully untouched estimate of the entire feature-discovery process.

**Human**: If the outcome were whether a store misses its sales target, what would change? Use a small constructed example to explain logistic probabilities, thresholds, and SVM scores.

**Instructor**: Constructed claim: “An SVM score of 0.1 means a 10% chance of missing the target.” Explain what is wrong and what a probability threshold changes.

<!-- checkpoint: 5. Regression and classification -->
**Assistant**: **The task would become binary classification:** predict whether a store misses its target, instead of predicting its dollar sales.

Define
\[
Y=\begin{cases}
1,&\text{sales below target},\\
0,&\text{sales at or above target}.
\end{cases}
\]
The target must be specified before observing sales. Predictors must also be available when we make the prediction.

**Logistic regression produces a probability.** It combines predictors into a score \(\eta=\beta_0+\beta^\top x\), potentially including curved terms, then applies
\[
p=P(Y=1\mid x)=\frac{1}{1+e^{-\eta}}.
\]

For example, if \(\eta=-0.4\),
\[
p=\frac{1}{1+e^{0.4}}\approx0.401.
\]
The model estimates a **40.1% chance of missing the target**.

Here are four hypothetical sets of model outputs. These scores were assigned for illustration; no classifier was fitted.

| Store | Logistic score \(\eta\) | Miss probability | Predict miss at \(p\ge0.5\)? | At \(p\ge0.3\)? | SVM score |
|---|---:|---:|:---:|:---:|---:|
| A | −2.0 | 11.9% | No | No | −1.2 |
| B | −0.4 | 40.1% | No | Yes | +0.1 |
| C | +0.4 | 59.9% | Yes | Yes | +0.6 |
| D | +1.4 | 80.2% | Yes | Yes | +1.3 |

**A probability and a decision are different outputs.** Lowering the threshold changes store B’s classification without changing its probability. A lower threshold flags more stores, potentially catching more misses while creating more false alarms. The threshold should reflect the costs of those errors and be selected using training validation.

**An SVM produces a decision score.** With positive scores assigned to the “miss” class, the usual zero threshold flags B, C, and D. Store B lies close to that SVM’s decision boundary.

An SVM score of \(+0.1\) is **not a 10% probability**. Its magnitude depends on the fitted model and feature scaling. Converting SVM scores into probabilities requires a separately validated calibration step. Logistic and SVM predictions can disagree because they fit different decision rules.

Evaluation would also change: examine missed cases, false alarms, precision and recall, and probability quality when probabilities matter. Without observed outcomes in this illustration, we cannot calculate accuracy or identify a better classifier.

[Calculation code](/examples/regression-forecasting-v1/store_sales_05_classification_score_illustration.py) · [Illustrative outputs](/examples/regression-forecasting-v1/store_sales_05_hypothetical_model_outputs.csv)

**Instructor**: A logistic coefficient describes a change in log odds: holding other inputs fixed, a one-unit increase in a predictor multiplies the odds by exp(beta). The odds are p/(1-p). It does not add beta to the probability. Optional constructed metric exercise: assign hypothetical outcomes A=0, B=1, C=0, D=1 to the displayed illustration. Pause and compute precision and recall at each logistic threshold. Answer: at 0.5, TP=1, FP=1, FN=1 and TN=1, so precision=1/2 and recall=1/2. At 0.3, TP=2, FP=1, FN=0 and TN=1, so precision=2/3 and recall=1. These four invented outcomes are a calculation exercise, not validation evidence.

**Human**: Let’s choose the curved regression for simplicity. Freeze the comparison and evaluate the reserved stores. What can we conclude, and would you trust a prediction far beyond the observed advertising range?

**Instructor**: Write down the chosen model and what would count as a useful improvement before revealing the held-out results. Why might a prediction far outside the advertising range fail even with a good test score?

<!-- checkpoint: 6. Final evaluation and extrapolation -->
**Assistant**: **The frozen curved regression performs about as well on the reserved stores as validation suggested. I would use it cautiously for similar stores, but I would not trust predictions far beyond the observed advertising range.**

We fixed the model choice before evaluating the 75 reserved stores, then fitted its coefficients using only the 225 training stores. No tuning or model switching followed the test results.

| Measure | Reserved-store result |
|---|---:|
| RMSE | **$34,472** |
| Mean absolute error | **$27,281** |
| \(R^2\) | **0.675** |
| Mean prediction minus actual sales | **−$1,427** |

The test RMSE is close to the earlier validation RMSE of **$35,271**. For comparison, predicting the training-set mean sales for every reserved store gives an RMSE of **$60,935**. The predictors therefore provide useful information, although individual predictions still have substantial error.

The mean absolute error means predictions were about $27,281 away from actual monthly sales on average. It is not an error bound for every store.

![Reserved-store predictions and the quadratic curve extended beyond training advertising values](/examples/regression-forecasting-v1/store_sales_06_test_and_extrapolation.png)

The left panel compares predictions with actual sales; points on the dashed line would be exact predictions.

The right panel shows why extrapolation is risky. Training advertising ranged from **$5,000 to approximately $55,205 per month**, shown by the shaded region. Holding the other inputs at their training medians, the fitted curve predicts:

| Monthly advertising | Predicted monthly sales |
|---|---:|
| $100,000 | $232,158 |
| $150,000 | **−$28,250** |

The negative prediction exposes the limitation of extending this quadratic indefinitely. A squared term can describe local curvature without providing sensible behavior at distant inputs. Also, extremely high advertising combined with median store size may be unlike the combinations represented in training.

**Our conclusion is limited to prediction for stores similar to this constructed sample.** The reserved results support the chosen model’s predictive usefulness; they do not establish superiority over the nearly tied alternatives, validate real business performance, or show that changing advertising causes sales to change. Predictions far outside the supported range need new evidence.

[Code](/examples/regression-forecasting-v1/store_sales_06_frozen_test_evaluation.py) · [Frozen selection](/examples/regression-forecasting-v1/store_sales_06_frozen_selection.json) · [Test results](/examples/regression-forecasting-v1/store_sales_06_test_results.json) · [Store predictions](/examples/regression-forecasting-v1/store_sales_06_reserved_store_predictions.csv)

**Instructor**: For polynomial degree, compare complete candidate procedures using development validation; keep lower-order terms with higher powers. A small training error cannot select the degree. A reserved test measures the frozen procedure under its sampling conditions; it does not validate distant extrapolation or causal advertising decisions. End by writing a recommendation with one numerical comparison, one limitation and one next check.

**Instructor**: [Data attribution and reproduction](/examples/regression-forecasting-v1/DATA.md) · [Hotel data](/examples/regression-forecasting-v1/hotel_monthly.csv) · [Environment](/examples/regression-forecasting-v1/requirements.txt) · [Script order](/examples/regression-forecasting-v1/reproduction-order.txt). Each lesson uses its own explicitly reserved evaluation data.
