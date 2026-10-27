# Title: Lecture 15: Support vector machines: margins and linear classification

**Instructor**: This replay shows a completed Codex CLI conversation. The human prompts were simulated for teaching. The assistant chose and executed the analyses; its responses are preserved with portable asset links. Instructor notes and pauses were added separately. Playback displays saved results.

The real dataset is [UCI Banknote Authentication](https://archive.ics.uci.edu/dataset/267/banknote+authentication), by Volker Lohweg, shared under CC BY 4.0. The lessons build on [Jiecheng Lu's evaluation, classification and scaling labs](https://ljc-fvnr.github.io/isye4031-summer2026-labs/) and the SVM chapter of [An Introduction to Statistical Learning](https://www.statlearning.com/). The banknote analysis is newly executed for these lessons. Any constructed algebra examples or claims are labeled explicitly.

**Instructor**: By the end, you should be able to do the following independently: Explain a signed classification score and its geometric margin; calculate hinge loss; interpret C and support vectors; check scaling and grouped validation; compare a linear SVM with a logistic baseline. Pause at each question before revealing the response.

**Human**: Remind us of the banknote question and show us the data.

**Instructor**: Identify the response and one predictor. What information would you need before calling class 1 a forged banknote?

<!-- checkpoint: 1. The real data and prediction question -->
**Assistant**: The banknote question is: **Can we predict a specimen’s class from four numerical summaries of its image?**

This is a binary classification problem. Each row represents a digitized banknote-like specimen, and the response is **class 0 or class 1**. The source describes authentic and forged specimens but does not specify which numeric label means which category, so we will keep the numeric labels.

The four predictors are **variance, skewness, kurtosis, and entropy**, computed from wavelet-transformed images. These are image summaries, not currency amounts. The supplied documentation gives no physical units.

The dataset contains **1,372 rows**, with no missing values. Here are six observations from the development data, rounded for display:

| Row | Variance | Skewness | Kurtosis | Entropy | Class |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.622 | 8.666 | −2.807 | −0.447 | 0 |
| 2 | 4.546 | 8.167 | −2.459 | −1.462 | 0 |
| 3 | 3.866 | −2.638 | 1.924 | 0.106 | 0 |
| 4 | 3.457 | 9.523 | −4.011 | −3.594 | 0 |
| 5 | 0.329 | −4.455 | 4.572 | −0.989 | 0 |
| 6 | 4.368 | 9.672 | −3.961 | −3.163 | 0 |

Before plotting the classes, I reserved **343 rows for the final test**. The remaining **1,029 development rows** contain 572 observations of class 0 and 457 of class 1. All displays below use development observations.

![Banknote development data shown through two pairs of image summaries, colored by class](/examples/svm-week-v1/banknote_development_scatter_v1.png)

Each point is one observation; color and shape identify its class. The panels show two different projections of the same four-dimensional predictor space. They help us see how predictor values vary across the classes, but neither panel contains all the information available to a classifier. Overlap in a projection does not establish that the classes overlap when all four predictors are used.

One data-quality detail matters: **11 feature vectors occur more than once**, producing 24 extra duplicate rows. None has conflicting class labels. I kept every row and assigned identical feature vectors to the same split group, preventing an exact copy from appearing in both development and final-test data. Unknown dependence between specimens remains a limitation because no specimen identifiers or acquisition batches were supplied.

Our question is therefore about **prediction on unseen specimens**, not simply drawing a boundary through the displayed points. These plots describe the development data; they do not measure predictive performance.

Reproducible files: [Python code](/examples/svm-week-v1/banknote_data_overview_v1.py), [data audit](/examples/svm-week-v1/banknote_data_audit_v1.json), [split assignments](/examples/svm-week-v1/banknote_split_roles_v1.csv), and [displayed observations](/examples/svm-week-v1/banknote_development_preview_v1.csv).

**Instructor**: Source detail: UCI describes variance, skewness and kurtosis as wavelet-image features and entropy as an image feature. It does not supply physical units or a mapping from the numeric class labels to authentic and forged specimens.

**Human**: Can you show what a separating line and its margin mean?

**Instructor**: For the constructed score f(z) = 3z1 + 4z2 - 5, predict the class at z = (1, 1). What is its distance from the decision boundary?

<!-- checkpoint: 2. A line and its margin -->
**Assistant**: A **separating line** divides the feature plane into two prediction regions. Its **margin** measures a perpendicular distance around that boundary.

Here is an illustration using only **variance and skewness** from the banknote development data. Both features are standardized using development means and standard deviations, so the geometry uses comparable scales.

![A linear SVM decision boundary and its margin in two standardized banknote features](/examples/svm-week-v1/banknote_margin_geometry_v1.png)

The solid line is the **decision boundary**. A point’s predicted class depends on which side it falls:

- The side containing most orange triangles predicts class 1.
- The side containing most blue circles predicts class 0.
- Crossing the solid line changes the prediction.

Write the score as

\[
f(z)=w^\top z+b=w_1z_1+w_2z_2+b,
\]

where \(z_1,z_2\) are the standardized features. The boundary is \(f(z)=0\); positive scores predict class 1 and negative scores predict class 0. This score is not a probability.

For the fitted illustration,

\[
f(z)\approx -2.028z_1-1.180z_2-0.255.
\]

The dashed lines mark

\[
f(z)=-1 \qquad\text{and}\qquad f(z)=+1.
\]

They are parallel to the decision boundary. The perpendicular distance from the solid line to either dashed line is

\[
\frac{1}{\|w\|},
\]

so the **full margin width** is

\[
\boxed{\frac{2}{\|w\|}}.
\]

Here, those distances are approximately **0.426 on each side**, giving a full width of **0.853** in standardized feature space. The purple arrow shows that width. We measure perpendicular to the line because that is the shortest distance.

Why choose a wide margin? If several lines correctly separate the training classes, a wider margin provides more space between the boundary and the closest observations. In the perfectly separable case, the maximum-margin SVM chooses the line with the widest such gap. The observations touching its margin boundaries are support vectors.

**These two banknote features do not give perfect separation.** The displayed fit makes 127 training classification errors among 1,029 observations. It is a **soft-margin SVM**, which allows points inside the margin and even across the decision boundary. Consequently, the shaded margin is not an empty strip. A correctly classified point can still fall inside it.

This fit uses a fixed \(C=1\) only to demonstrate geometry. It is a two-feature training illustration, distinct from a predictor using all four features, and it provides no final-test performance estimate.

[Code](/examples/svm-week-v1/banknote_margin_geometry_v1.py) · [Numerical results](/examples/svm-week-v1/banknote_margin_geometry_results_v1.json) · [Plotted data and scores](/examples/svm-week-v1/banknote_margin_geometry_points_v1.csv)

**Instructor**: Constructed calculation: f(1, 1) = 2, so the prediction is class 1. The perpendicular distance is |2|/sqrt(3^2 + 4^2) = 0.4. Rescaling w and b together preserves the boundary and this distance. A score of 2 is not a probability.

**Human**: How does the soft margin decide which mistakes to tolerate?

**Instructor**: A point has signed score 0.5. Is its classification correct? Does it contribute to hinge loss?

<!-- checkpoint: 3. Hinge loss and the soft margin -->
**Assistant**: The soft-margin SVM tolerates mistakes when correcting them would cost too much in margin width. It makes that tradeoff through a single objective:

\[
\min_{w,b}\;
\underbrace{\frac12\|w\|^2}_{\text{encourages a wider margin}}
+
C\underbrace{\sum_i\max\{0,1-t_i f(x_i)\}}_{\text{penalizes margin violations}},
\]

where \(t_i=+1\) for class 1 and \(t_i=-1\) for class 0.

The key quantity is the **signed score**

\[
m_i=t_i f(x_i).
\]

A positive signed score means the prediction is correct; a negative one means it is wrong. Its magnitude tells us how far the score lies from the decision threshold. The penalty, called **hinge loss**, is

\[
\xi_i=\max(0,1-m_i).
\]

These constructed examples show how it works:

| Signed score \(m_i\) | Position | Penalty \(\xi_i\) |
|---:|---|---:|
| 1.5 | Correct, beyond its margin line | 0 |
| 1.0 | Correct, on its margin line | 0 |
| 0.5 | Correct, inside the margin | 0.5 |
| 0 | On the decision boundary | 1 |
| −0.5 | Misclassified | 1.5 |
| −2.0 | Further onto the wrong side | 3 |

Thus, **the SVM penalizes more than classification errors**. A correct prediction inside the margin also incurs a penalty. A point far onto the wrong side incurs more loss than a point just across the boundary.

There is no separate rule declaring particular observations acceptable mistakes. The algorithm moves the line to minimize the combined objective across all observations. Moving it to help one point can hurt several others or narrow the margin substantially.

**\(C\) controls how expensive violations are.** In the convention used here, increasing \(C\) puts more weight on the total hinge loss. Decreasing \(C\) gives more weight to keeping a wide margin.

Here is what happens when we refit the same standardized variance-and-skewness illustration:

| \(C\) | Training errors | Total hinge loss | Full margin width |
|---:|---:|---:|---:|
| 0.01 | 129 | 335.81 | 1.685 |
| 0.1 | 127 | 291.98 | 1.038 |
| 1 | 127 | 287.47 | 0.853 |
| 10 | 127 | 287.44 | 0.841 |

Increasing \(C\) narrowed the margin and reduced hinge loss, but the error count stayed at 127 across the last three fits. That illustrates an important distinction: **minimizing hinge loss is not the same as minimizing the number of mistakes**.

These are development training results from a two-feature model. They explain the tradeoff; they do not establish which \(C\) predicts unseen observations best.

[Code](/examples/svm-week-v1/banknote_soft_margin_tradeoff_v1.py) · [Banknote results](/examples/svm-week-v1/banknote_soft_margin_tradeoff_v1.csv) · [Constructed examples](/examples/svm-week-v1/soft_margin_constructed_score_examples_v1.csv)

**Instructor**: It is correctly classified and contributes 1 - 0.5 = 0.5 to hinge loss. In sklearn, C multiplies the total loss. ISLP also presents a constrained formulation whose C is a slack budget; increasing that budget permits more violations. Check the objective before interpreting the symbol.

**Human**: Show us how C changes the boundary and which points support it.

**Instructor**: Predict which points will matter for the fitted boundary. Must every support vector be a classification error?

<!-- checkpoint: 4. Support vectors and changing C -->
**Assistant**: **Increasing \(C\) narrows the margin here, shifts the boundary slightly, and changes which observations support it.** The largest visible change occurs between \(C=0.01\) and \(C=0.1\); the boundaries for \(C=1\) and \(C=10\) are very similar.

Each panel uses exactly the same development observations and standardized variance and skewness. Black rings identify the support vectors returned by the fitted SVM.

![Linear SVM boundaries, margin lines, and circled support vectors for four values of C](/examples/svm-week-v1/banknote_C_support_vectors_v1.png)

Read each panel as follows:

- The **solid line** is the decision boundary, \(f(z)=0\).
- The **dashed lines** are \(f(z)=-1\) and \(f(z)=+1\).
- The **rings** mark observations with nonzero support-vector weights.

At small \(C\), violations are relatively inexpensive. The model accepts a wide margin containing many observations. As \(C\) increases, it pays more attention to reducing their hinge losses, accepting a narrower margin.

| \(C\) | Full margin width | Support vectors | Training errors |
|---:|---:|---:|---:|
| 0.01 | 1.685 | 478 | 129 |
| 0.1 | 1.038 | 330 | 127 |
| 1 | 0.853 | 295 | 127 |
| 10 | 0.841 | 290 | 127 |

**Support vectors are not just the misclassified points.** They include correctly classified observations inside the margin, observations on the margin boundaries, and misclassified observations. Notice the rings around some points well across the decision line: a soft-margin support vector need not be close to that line.

Their mathematical role is especially clear from

\[
w=\sum_i \alpha_i t_i z_i,
\qquad 0\leq\alpha_i\leq C,
\]

where \(t_i\) is the signed class label. Support vectors have \(\alpha_i>0\), so they contribute directly to the fitted direction \(w\). Other training observations have zero weight in this representation.

At an exact optimum:

- Points strictly beyond their correct margin boundary have \(\alpha_i=0\).
- Points with \(0<\alpha_i<C\) lie on a margin boundary.
- Points strictly violating the margin have \(\alpha_i=C\).

For example, at \(C=1\), **292 of the 295 support vectors have weights at the cap \(C\)**; the other three have weights between zero and \(C\). The many violations explain why the picture has far more than a few circled points.

The support-vector count decreases in this example, but that is not a universal rule as \(C\) changes. These are two-feature training illustrations, not evidence that one \(C\) predicts new specimens best. The final holdout remains unused.

[Code](/examples/svm-week-v1/banknote_C_support_vectors_v1.py) · [Summary results](/examples/svm-week-v1/banknote_C_support_vectors_summary_v1.csv) · [Point identities and weights](/examples/svm-week-v1/banknote_C_support_vectors_points_v1.csv)

**Instructor**: Correctly classified observations can be support vectors when they lie on or inside the margin. With the feature transformation and C fixed, observations strictly beyond the margin have zero hinge loss. Refitting a scaler after moving points changes the transformation and is a different experiment.

**Human**: Would changing the units of a predictor change this model?

**Instructor**: Suppose variance is multiplied by 1000 in every row. Predict what happens with and without standardization. Which data may determine the scaling constants?

<!-- checkpoint: 5. Units and scaling -->
**Assistant**: **Without standardization, changing a predictor’s units can change the fitted SVM. With the standardization used in our earlier models, a positive unit conversion leaves the fit unchanged, apart from numerical rounding.**

To demonstrate, I multiplied variance by \(0.1\), kept skewness unchanged, and refitted at **\(C=1\)**. This is a numerical unit conversion, with no physical unit implied.

Both panels below display the boundaries in the **original coordinates**, so any difference reflects a changed fit.

![Effect of changing variance units on unstandardized and standardized linear SVM boundaries](/examples/svm-week-v1/banknote_units_comparison_v5.png)

On the left, the boundary moves when variance is rescaled. On the right, standardization makes the two boundaries overlap.

| Fitting procedure | Predictions changed among 1,029 development rows | Support vectors, original → converted |
|---|---:|---:|
| Without standardization | 15 | 289 → 327 |
| With standardization | 0 | 295 → 295 |

All four fits in this comparison converged. With standardization, the largest absolute score difference was about \(3.5\times10^{-14}\), which is numerical rounding.

**Why do units matter?** Recall the objective:

\[
\frac12(w_1^2+w_2^2)
+C\sum_i\max(0,1-t_i f(x_i)).
\]

Suppose we replace \(x_1\) by \(x'_1=a x_1\). To preserve exactly the same scores, its coefficient would need to become

\[
w'_1=\frac{w_1}{a}.
\]

That preserves \(w'_1x'_1=w_1x_1\), but changes the coefficient penalty:

\[
(w'_1)^2=\frac{w_1^2}{a^2}.
\]

For \(a=0.1\), preserving that predictor’s contribution would make its squared coefficient **100 times as large**. The optimizer therefore faces a different tradeoff between coefficient size and hinge loss. Units have changed the geometry underlying the margin.

Standardization cancels this effect. If

\[
z_1=\frac{x_1-\mu_1}{s_1},
\]

then after a positive unit conversion,

\[
\frac{a x_1-a\mu_1}{a s_1}=z_1.
\]

The model receives the same standardized inputs, so it solves the same optimization problem at the same \(C\).

This explains why scaling is part of the model specification. For our earlier plots, changing the raw units and then refitting the standardizer would preserve the model. When we later use cross-validation, each standardizer must learn its means and standard deviations **only from that fold’s training rows**, then apply them to its validation rows.

These results concern the same two-feature development illustration. The final holdout remains unused.

[Code](/examples/svm-week-v1/banknote_units_comparison_v5.py) · [Fitted model results](/examples/svm-week-v1/banknote_units_model_results_v5.csv) · [Prediction comparison](/examples/svm-week-v1/banknote_units_prediction_comparison_v5.csv)

**Human**: How should we choose C and compare this with logistic regression using all four predictors?

**Instructor**: Where do scaling and C selection occur? Explain why a selected CV score needs a separate final test. What does balanced accuracy average?

<!-- checkpoint: 6. Choosing C and a linear baseline -->
**Assistant**: **Choose \(C\) using development cross-validation, and compare both models on the same validation folds.** Using all four predictors, the linear SVM and logistic regression perform almost identically in this comparison.

Logistic regression provides a useful comparator: it models the probability of class 1 as

\[
P(Y=1\mid z)=\frac{1}{1+\exp[-(b+w^\top z)]}.
\]

We predict class 1 when that probability is at least \(0.5\). Its decision boundary is linear, like the linear SVM’s. The main difference is the fitting loss: logistic regression uses log loss, while the SVM uses hinge loss. Both models here penalize squared coefficient size. Increasing \(C\) weakens regularization in both implementations, but **equal \(C\) values do not imply equivalent fits**.

I used the following procedure:

1. Keep the final holdout untouched and use the **1,029 development rows**.
2. Create five stratified group folds, keeping identical feature vectors together.
3. Within each training fold, fit a standardizer and then fit the model using **variance, skewness, kurtosis, and entropy**.
4. Apply that fitted pipeline to the validation fold.
5. Select each model’s \(C\) by its highest mean validation **balanced accuracy**, using smaller \(C\) for exact ties.

Balanced accuracy gives equal weight to the two classes:

\[
\text{Balanced accuracy}
=\frac{\text{class 0 recall}+\text{class 1 recall}}{2}.
\]

![Five-fold development validation curves for four-feature linear SVM and logistic regression](/examples/svm-week-v1/banknote_four_feature_cv_v2.png)

Thick lines show fold means; thin lines show individual folds. Very small \(C\) values perform poorly here. Performance improves as regularization weakens, then largely levels off.

| Model | Selected \(C\) | Mean balanced accuracy | Fold SD | Mean class 1 recall |
|---|---:|---:|---:|---:|
| Linear SVM | 100 | 98.95% | 0.79 percentage points | 99.12% |
| Logistic regression | 1,000 | 98.97% | 1.24 percentage points | 99.34% |

Logistic regression initially reached its best result at the largest candidate \(C\), so I extended its grid. Values of \(10{,}000\) and \(100{,}000\) gave the same mean balanced accuracy; the tie rule selects \(1{,}000\).

**These results provide little evidence for preferring either model on classification performance.** Their mean balanced accuracies differ by only about **0.024 percentage points**, much less than the observed variation across folds. The fold SD describes variability across these splits; it is not a confidence interval.

These are **model-selection CV results**. Because we used them to choose \(C\), their maxima can be optimistic. They also cannot be compared directly with the earlier two-feature training errors.

The resulting candidates are therefore a standardized four-feature linear SVM with \(C=100\) and standardized logistic regression with \(C=1{,}000\). Final evaluation remains deferred until model and tuning choices are frozen. Grouping protects against exact duplicates, but cannot address unknown specimen dependence.

[Code](/examples/svm-week-v1/banknote_four_feature_cv_v2.py) · [Full CV grid](/examples/svm-week-v1/banknote_four_feature_cv_grid_v2.csv) · [Selected results](/examples/svm-week-v1/banknote_four_feature_cv_selected_v2.csv) · [Fold assignments](/examples/svm-week-v1/banknote_four_feature_cv_folds_v2.csv)

**Instructor**: Balanced accuracy averages recall in class 0 and class 1. For ordinary binary classification, those are specificity and sensitivity when class 1 is positive. The same CV scores used to choose C can look optimistic as a performance estimate. Fold standard deviations describe dispersion and are not independent-sample confidence intervals.

**Human**: Suppose an agent says the largest C is always best and an SVM score of 0.8 means an 80 percent probability. How would we check that?

**Instructor**: Constructed claim: 'The largest C is best because it fits training data most closely, and an SVM score of 0.8 means an 80 percent chance of class 1.' Identify the two mistakes and the evidence you would request.

<!-- checkpoint: 7. Checking an agent's interpretation -->
**Assistant**: **We should ask for the evidence behind each claim and check exactly what the model’s output means. Both claims fail those checks.**

**“The largest \(C\) is always best.”** We can check this against our saved four-predictor validation results:

| Linear SVM \(C\) | Mean validation balanced accuracy |
|---:|---:|
| 10 | 98.51% |
| 100 | **98.95%** |
| 1,000 | 98.60% |

The largest tested value, \(C=1{,}000\), performed worse than \(C=100\) by about **0.35 percentage points**. Comparing the same folds, it tied in four folds and performed worse in one.

That does not establish that \(100\) is universally optimal. It does contradict the claim that the largest candidate always gives the best validation result.

The reasoning matters too. Increasing \(C\) makes hinge-loss violations more expensive relative to the coefficient penalty. It does **not** guarantee better predictions on new observations. Even the training classification error need not decrease at every step, because the objective penalizes hinge loss, not simply the number of mistakes.

To audit an agent’s recommendation, check that it:

- Used validation results to select \(C\).
- Compared candidates on the same folds.
- Fit scaling inside each training fold.
- Kept duplicate feature vectors together and left the final holdout untouched.

**“An SVM score of 0.8 means an 80 percent probability.”** First inspect which output the agent used. Our SVM’s `decision_function` returns

\[
f(z)=w^\top z+b.
\]

A score of \(0.8\) is positive, so it predicts class 1. It is also below \(1\), the positive-side margin boundary. **It is not a probability.** SVM scores can be negative or exceed one.

A constructed algebra check makes the distinction clear. Replace \(f(z)\) by \(10f(z)\). The zero boundary and every predicted class stay unchanged, but the score \(0.8\) becomes \(8\). This rescaling is not a claim about a refitted SVM; it demonstrates why classification scores alone have no percentage interpretation.

Applying a sigmoid does not automatically solve the problem either:

\[
\sigma(0.8)\approx0.690,\qquad
\sigma(8)\approx0.9997.
\]

Neither number becomes a justified probability simply because it lies between zero and one.

If probabilities are needed, we must **fit a calibration mapping using predictions from observations excluded from the corresponding SVM fit**, preserve the grouping rules, and assess calibration on separate validation predictions. Useful checks include a reliability plot, Brier score, and log loss. Logistic regression produces probability estimates directly, but their calibration also needs checking.

The appropriate conclusion is: **choose \(C\) through validation, and interpret an SVM score as a decision score unless probability calibration has been fitted and assessed.**

[Check code](/examples/svm-week-v1/banknote_agent_claim_checks_v1.py) · [Numerical results](/examples/svm-week-v1/banknote_agent_claim_checks_v1.json) · [Paired fold comparison](/examples/svm-week-v1/banknote_agent_claim_C_fold_check_v1.csv)

**Instructor**: [Data attribution and reproduction](/examples/svm-week-v1/DATA.md) · [Original data](/examples/svm-week-v1/banknote_raw.csv) · [Environment](/examples/svm-week-v1/requirements.txt) · [Script order](/examples/svm-week-v1/reproduction-order.txt). The same data and reserved split connect both lectures.

