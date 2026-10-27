# Title: Lecture 16: Nonlinear models: features, kernels and validation

**Instructor**: This replay shows a completed Codex CLI conversation. The human prompts were simulated for teaching. The assistant chose and executed the analyses; its responses are preserved with portable asset links. Instructor notes and pauses were added separately. Playback displays saved results.

The real dataset is [UCI Banknote Authentication](https://archive.ics.uci.edu/dataset/267/banknote+authentication), by Volker Lohweg, shared under CC BY 4.0. The lessons build on [Jiecheng Lu's evaluation, classification and scaling labs](https://ljc-fvnr.github.io/isye4031-summer2026-labs/) and the SVM chapter of [An Introduction to Statistical Learning](https://www.statlearning.com/). The banknote analysis is newly executed for these lessons. Any constructed algebra examples or claims are labeled explicitly.

**Instructor**: By the end, you should be able to do the following independently: Explain how transformed features create curved boundaries; interpret kernel similarity and gamma; tune the entire procedure without leakage; compare frozen models on reserved data; state what the result can support. Pause at each question before revealing the response.

**Human**: Remind us of the banknote problem, show the data again, and tell us where the linear models left off.

**Instructor**: Explain what the linear models established and what remains unknown. Which observations are still reserved?

<!-- checkpoint: 1. Return to the banknote question -->
**Assistant**: **Our question is whether four numerical summaries of a banknote image can predict its class. The linear models reached about 99% balanced accuracy in development cross-validation, with very little difference between them.**

Each row represents a digitized banknote-like specimen. The predictors are **variance, skewness, kurtosis, and entropy**, computed from wavelet-transformed images. They are image summaries, not currency amounts.

The response is **class 0 or class 1**. The supplied documentation describes authentic and forged specimens but does not identify which numeric label corresponds to which category.

There are **1,372 observations**, with no missing values. We reserved **343 for the final holdout** and used the remaining **1,029 for development**, including 572 class 0 observations and 457 class 1 observations.

Here are several development observations, rounded for display:

| Row | Variance | Skewness | Kurtosis | Entropy | Class |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.622 | 8.666 | −2.807 | −0.447 | 0 |
| 2 | 4.546 | 8.167 | −2.459 | −1.462 | 0 |
| 3 | 3.866 | −2.638 | 1.924 | 0.106 | 0 |
| 4 | 3.457 | 9.523 | −4.011 | −3.594 | 0 |
| 5 | 0.329 | −4.455 | 4.572 | −0.989 | 0 |

![Development banknote observations shown through two pairs of image summaries](/examples/svm-week-v1/banknote_development_scatter_v1.png)

These panels show two projections of the same four-dimensional data. Visible overlap in either panel does not establish overlap when all four predictors are used.

We first used **variance and skewness alone** to explain the SVM boundary, margin, and support vectors. Those were training illustrations. They showed how larger \(C\) makes margin violations more expensive and how changing predictor units can alter an unstandardized fit.

We then compared **two models using all four predictors**:

| Model | Selected \(C\) | Mean validation balanced accuracy | Fold SD |
|---|---:|---:|---:|
| Linear SVM | 100 | **98.95%** | 0.79 percentage points |
| Logistic regression | 1,000 | **98.97%** | 1.24 percentage points |

Both used the same five stratified group folds. Identical feature vectors stayed together, and standardization was fitted separately inside each training fold. Balanced accuracy averages the recalls for the two classes, giving each class equal weight.

The difference between the models was only **0.024 percentage points**. These results give little basis for declaring a predictive winner. The fold SDs describe variation across splits; they are not confidence intervals.

Two interpretation checks remain important: **the largest \(C\) was not always best**, and **an SVM decision score is not a probability**. Logistic regression supplies probability estimates, whose calibration still needs checking.

We left off with two strong linear candidates. Their reported scores are **model-selection CV results**, which can be optimistic after tuning. The final holdout remains unused, and this historical benchmark does not establish production performance.

[Data-display code](/examples/svm-week-v1/banknote_data_overview_v1.py) · [CV code](/examples/svm-week-v1/banknote_four_feature_cv_v2.py) · [Recap code](/examples/svm-week-v1/banknote_linear_recap_v1.py) · [Results](/examples/svm-week-v1/banknote_linear_recap_results_v1.csv)

**Instructor**: Source detail: UCI describes variance, skewness and kurtosis as wavelet-image features and entropy as an image feature. It does not supply physical units or a mapping from the numeric class labels to authentic and forged specimens.

**Human**: How can a model that is linear in its coefficients make a curved decision boundary? Show us with these data.

**Instructor**: Could a model that is linear in its coefficients have a curved decision boundary in the original inputs? Give a simple feature example.

<!-- checkpoint: 2. Curves through new features -->
**Assistant**: **A model can be linear in its coefficients while using nonlinear functions of its predictors.** The fitted score is then a weighted sum of transformed features, and its zero boundary can be curved in the original feature plane.

Let \(z_1\) be standardized variance and \(z_2\) standardized skewness. Our earlier two-feature linear SVM used

\[
f(z)=b+w_1z_1+w_2z_2.
\]

Setting \(f(z)=0\) gives a straight line. Now expand the inputs to

\[
\phi(z)=
\left(z_1,\ z_2,\ z_1^2,\ z_1z_2,\ z_2^2\right).
\]

A linear SVM fitted to those five features has score

\[
f(z)=b+\beta_1z_1+\beta_2z_2
+\beta_3z_1^2+\beta_4z_1z_2+\beta_5z_2^2.
\]

This is **linear in the fitted coefficients**: each coefficient multiplies a known feature, and the contributions are added. But the equation \(f(z)=0\) is quadratic in \(z_1,z_2\), so its boundary can curve.

Here is that construction fitted to the banknote development observations:

![Straight and quadratic decision boundaries fitted to variance and skewness from the banknote development data](/examples/svm-week-v1/banknote_quadratic_boundary_v1.png)

Both panels use the same **1,029 observations**, the same two original predictors, and a fixed **\(C=1\)**. The left model uses two standardized features. The right model adds their squares and interaction, then standardizes the expanded columns before fitting a linear SVM.

The fitted quadratic score, expressed back in the standardized original predictors, is approximately

\[
\begin{aligned}
f(z)={}&0.411-2.726z_1-0.598z_2\\
&+0.489z_1^2+1.839z_1z_2-1.275z_2^2.
\end{aligned}
\]

The black curve is precisely where this expression equals zero. Positive values predict class 1, shown by the orange background; negative values predict class 0, shown by blue.

Notice how the curved boundary follows the broad bend between the orange observations on the left and blue observations on the right. The fitted training error count falls from **127 to 72**. That demonstrates the additional flexibility of the quadratic features.

It does **not** establish better prediction on new specimens. These are training results from a two-feature illustration. They are separate from the earlier cross-validation comparison using all four predictors. Also, scaling the expanded features changes the regularization geometry, so equal \(C\) values do not imply equivalent penalties between these two representations.

In the five-dimensional transformed space, the quadratic model still uses a flat separating hyperplane. When expressed in the original two predictors, that same decision rule becomes a curve. **“Linear” describes how the coefficients enter the score; the feature transformation determines what shapes that score can produce in the original inputs.**

[Code](/examples/svm-week-v1/banknote_quadratic_boundary_v1.py) · [Training results](/examples/svm-week-v1/banknote_quadratic_boundary_results_v1.csv) · [Fitted coefficients](/examples/svm-week-v1/banknote_quadratic_boundary_coefficients_v1.json)

**Human**: How does a kernel let us do this without writing out all those new features?

**Instructor**: For the polynomial kernel K(z, u) = (1 + z dot u)^2, take z = (1, 0) and u = (0, 1). Calculate K. How could we check the same number using expanded features?

<!-- checkpoint: 3. Kernel similarity -->
**Assistant**: **A kernel computes the dot product between transformed observations without constructing their expanded feature vectors.** This works because the SVM can be fitted and used through dot products.

Let \(z=(z_1,z_2)\) contain standardized variance and skewness. Consider the quadratic feature map

\[
\phi(z)=
\left(
1,\sqrt{2}z_1,\sqrt{2}z_2,
z_1^2,\sqrt{2}z_1z_2,z_2^2
\right).
\]

For two observations \(z\) and \(u\), expanding their transformed dot product gives

\[
\begin{aligned}
\phi(z)^\top\phi(u)
={}&1+2z_1u_1+2z_2u_2\\
&+z_1^2u_1^2+2z_1z_2u_1u_2+z_2^2u_2^2\\
={}&(1+z^\top u)^2.
\end{aligned}
\]

Therefore, we can define the **polynomial kernel**

\[
\boxed{K(z,u)=(1+z^\top u)^2}
\]

and compute the same answer directly from the two original standardized features. The square implicitly supplies the quadratic terms and interaction.

The factors of \(\sqrt{2}\) matter: they make the identity exact. A kernel specifies both the feature expansion and its weighting.

Why is that sufficient for an SVM? Recall its feature-space coefficient vector:

\[
w=\sum_i\alpha_i t_i\phi(z_i).
\]

Substituting this into the prediction score gives

\[
\begin{aligned}
f(z)
&=b+w^\top\phi(z)\\
&=\boxed{b+\sum_{i\in\mathrm{SV}}\alpha_i t_iK(z_i,z)}.
\end{aligned}
\]

Training uses pairwise kernel values, and prediction combines kernel values between the new observation and the support vectors. We never need to store the expanded coordinates or explicitly construct \(w\) in that space.

I checked this with our banknote development data. For the first two development observations:

| Calculation | Result |
|---|---:|
| Explicit transformed dot product, \(\phi(z)^\top\phi(u)\) | 14.803599 |
| Direct kernel calculation, \((1+z^\top u)^2\) | 14.803599 |

I also fitted two SVMs at fixed \(C=1\): a linear SVM using the explicit map above, and a polynomial-kernel SVM using only the two standardized inputs. They produced **zero prediction disagreements**, with score differences below \(2\times10^{-13}\). Both made **73 training errors**.

This weighted feature map differs from the independently standardized quadratic expansion used previously, which made 72 training errors. The equivalence demonstrated here is between the **matching explicit map and kernel**, not between arbitrary quadratic preprocessing choices.

The kernel saves us from writing potentially enormous feature expansions. It does not automatically make computation cheap: pairwise kernel calculations can become costly as the number of observations grows. These checks illustrate the identity on two development features; they do not establish predictive improvement, and the final holdout remains unused.

[Code](/examples/svm-week-v1/banknote_kernel_identity_v1.py) · [Identity checks and results](/examples/svm-week-v1/banknote_kernel_identity_results_v1.json) · [Compared scores](/examples/svm-week-v1/banknote_kernel_identity_scores_v1.csv)

**Instructor**: Constructed calculation: the original dot product is zero, so K = 1. The displayed six-component map also has transformed dot product 1. The weighting factors in that map are part of the kernel, so an arbitrary quadratic feature pipeline need not give the same fitted scores.

**Human**: Show us how an RBF kernel works as we change gamma. Does a more detailed boundary predict better?

**Instructor**: Constructed claim: 'A more detailed boundary and better training accuracy prove that the nonlinear model is better.' What comparison would reveal the problem?

<!-- checkpoint: 4. Gamma and overfitting -->
**Assistant**: **Gamma controls how local an RBF kernel’s similarity becomes. A more detailed boundary can fit training observations better while predicting new observations worse.**

The RBF kernel is

\[
K(z,u)=\exp\!\left(-\gamma\|z-u\|^2\right),
\]

where \(z\) and \(u\) are standardized predictor vectors. Identical observations have similarity \(1\); similarity decreases with distance.

- **Small gamma:** similarity decreases slowly, so each support vector contributes over a broad region.
- **Large gamma:** similarity decreases rapidly, so contributions are concentrated near individual support vectors.

A useful distance scale is \(1/\sqrt{\gamma}\): at that distance, similarity equals \(e^{-1}\). Increasing gamma therefore shrinks the neighborhood over which observations are considered similar. This similarity is not a class probability.

Here is what happens to the banknote boundary when we hold **\(C=1\)** fixed:

![RBF decision boundaries for three gamma values using standardized variance and skewness](/examples/svm-week-v1/banknote_rbf_gamma_boundaries_v2.png)

These are **two-feature training illustrations**, using only variance and skewness. At gamma \(0.01\), the boundary is broad and smooth. At gamma \(1\), it bends around the main class structure. At gamma \(100\), it develops small pockets and islands.

The training error counts fall from **121 to 66 to 31**. That shows increased flexibility, but does not answer whether prediction improves.

To check prediction, I used **all four predictors** and the same five development folds as before. Standardization was fitted inside each training fold, and duplicate feature vectors stayed together.

![RBF similarity decay and four-feature training versus validation performance as gamma changes](/examples/svm-week-v1/banknote_rbf_gamma_similarity_cv_v2.png)

| Gamma | Mean training balanced accuracy | Mean validation balanced accuracy |
|---:|---:|---:|
| 0.01 | 98.05% | 98.16% |
| 0.1 | 99.93% | 99.74% |
| 1 | 100.00% | **100.00%** |
| 10 | 100.00% | 99.89% |
| 100 | 100.00% | **81.84%** |

At gamma \(100\), the four-feature models classify every training observation correctly, but validation performance drops sharply. Their similarity neighborhoods have become so narrow that fitting the training observations does not transfer reliably to validation observations. This is clear evidence of overfitting at that setting.

Gamma \(1\) made no validation errors across these five folds. That is a strong development result, **not a guarantee of perfect future performance**. We examined these folds to compare settings, so the best observed result can be optimistic.

The boundary pictures and validation table answer different questions: the pictures expose geometry in two predictors; the table assesses models using four predictors. They should not be interpreted as validation scores for the displayed curves.

**Boundary detail is something to validate, not something to maximize.** This sweep holds \(C\) fixed; a complete RBF selection should consider \(C\) and gamma together. The final holdout remains untouched.

[Code](/examples/svm-week-v1/banknote_rbf_gamma_v2.py) · [Validation results](/examples/svm-week-v1/banknote_rbf_gamma_four_feature_summary_v2.csv) · [Two-feature illustration results](/examples/svm-week-v1/banknote_rbf_gamma_two_feature_geometry_v2.csv)

**Human**: Can we tune C and gamma together and compare the nonlinear SVM fairly with our linear models?

**Instructor**: Draw one validation fold. Mark where scaling, feature expansion and parameter selection are allowed. Where do repeated feature vectors belong?

<!-- checkpoint: 5. Tuning the complete nonlinear procedure -->
**Assistant**: **Yes. Joint tuning selects an RBF SVM with \(C=1\) and gamma \(=1\). It outperforms our linear candidates on development cross-validation, although that comparison still needs independent final evaluation.**

We should tune the parameters together because they control different, interacting aspects of the fit:

- **Gamma** controls how local the kernel similarity is.
- **\(C\)** controls the penalty for margin violations relative to regularization.

A useful gamma can perform poorly when paired with too small a \(C\), so choosing each parameter in isolation can miss good combinations.

I evaluated a grid of **36 combinations**, using six values of \(C\) and six values of gamma. Every candidate used all four predictors and the **same five development folds** used for the linear models. Identical feature vectors stayed together, and each standardizer learned its means and standard deviations only from its training fold.

![Joint tuning of C and gamma for a four-predictor RBF SVM, showing mean validation balanced accuracy](/examples/svm-week-v1/banknote_joint_rbf_cv_heatmap_v1.png)

Each cell reports mean validation balanced accuracy. The outlined cell is the selected combination.

Several patterns matter. At gamma \(1\), values of \(C\) from \(1\) through \(1{,}000\) all achieved **100% balanced accuracy in every validation fold**. The deterministic tie rule chooses smaller \(C\), then smaller gamma, giving **\(C=1,\ \gamma=1\)**. That rule resolves ties; it does not establish that these parameters are uniquely optimal.

At gamma \(100\), increasing \(C\) does not recover comparable performance. Conversely, \(C=0.01\) performs poorly throughout this grid. The two parameters must work together.

Here is the comparison with our previously tuned linear models:

| Model | Selected settings | Mean validation balanced accuracy | Fold SD |
|---|---|---:|---:|
| Linear SVM | \(C=100\) | 98.95% | 0.79 percentage points |
| Logistic regression | \(C=1{,}000\) | 98.97% | 1.24 percentage points |
| RBF SVM | \(C=1,\ \gamma=1\) | **100.00%** | 0.00 percentage points |

This comparison controls the predictors, validation observations, preprocessing procedure, grouping, and scoring metric. It therefore provides useful evidence that nonlinear structure helps on these development splits. These are four-predictor results, separate from the earlier two-feature boundary illustrations.

**The remaining qualification is selection bias.** We used the validation folds to choose settings and compare models. The RBF search also considered more candidates, giving it more opportunities to obtain a favorable validation result. These maxima are not independent estimates of the complete tuning procedures. Nested cross-validation could assess those procedures; our reserved holdout can assess the frozen selected models.

Zero observed validation errors does not imply zero future error. Unknown specimen dependence also remains unresolved because the supplied data lack specimen and batch identifiers.

The RBF SVM is now the leading development candidate. The final holdout remains untouched.

[Code](/examples/svm-week-v1/banknote_joint_rbf_cv_v1.py) · [Full tuning grid](/examples/svm-week-v1/banknote_joint_rbf_cv_grid_v1.csv) · [Model comparison](/examples/svm-week-v1/banknote_joint_rbf_cv_comparison_v1.csv) · [Protocol](/examples/svm-week-v1/banknote_joint_rbf_cv_protocol_v1.json)

**Human**: Which model would you choose now, and what should we freeze before opening the final test set?

**Instructor**: Choose a model from the available development evidence and explain your reason. State which choices must be frozen before seeing test performance.

<!-- checkpoint: 6. Choose before opening the test set -->
**Assistant**: **I would choose the RBF SVM with \(C=1\) and gamma \(=1\).** It is our strongest development candidate: its mean validation balanced accuracy was **100%**, compared with **98.95%** for the linear SVM and **98.97%** for logistic regression, using the same five development folds.

That is a selection decision, not a claim of perfect future performance. We repeatedly examined these validation folds, so the winning score can be optimistic. Still, the nonlinear model’s advantage and its performance across several neighboring \(C\) settings support choosing it now.

Before opening the final test set, we should freeze the complete prediction procedure:

| Component | Frozen choice |
|---|---|
| Predictors | Variance, skewness, kurtosis, entropy |
| Preprocessing | `StandardScaler`, fitted only on development rows |
| Primary model | RBF SVM |
| Hyperparameters | \(C=1,\ \gamma=1\); no class weighting |
| Classification rule | Default SVM boundary at score \(0\) |
| Probability calibration | None; scores remain decision scores |
| Final fitting | Refit the pipeline on all 1,029 development rows |
| Test membership | Existing 343 reserved rows; preserve saved split assignments |

The standardizer must retain the means and standard deviations learned from development data. We apply those values to the test observations. We do not recalculate them using the test set.

We should also freeze **what we will measure and how we will interpret it**. The primary metric remains balanced accuracy,

\[
\frac{\text{class 0 recall}+\text{class 1 recall}}{2}.
\]

Secondary reporting will include accuracy, class 1 recall and precision, the confusion matrix, and the error count. We retain numeric class labels because their authentic/forged mapping is unspecified.

For context, I would evaluate the already selected linear SVM with \(C=100\) and logistic regression with \(C=1{,}000\) as **predeclared secondary references**. The RBF SVM remains the primary choice. Comparing the three on the test set must not become another tuning round or a way to select a new winner.

Finally, freeze the response to an unfavorable result: **report it without changing the model, threshold, features, or split.** Investigating failures afterward is useful, but any resulting revision needs new untouched data for an independent assessment.

I saved these decisions in a [frozen evaluation plan](/examples/svm-week-v1/banknote_frozen_final_evaluation_plan_v1.json), alongside the [development comparison](/examples/svm-week-v1/banknote_frozen_development_comparison_v1.csv) and [code that records the plan](/examples/svm-week-v1/banknote_freeze_analysis_plan_v1.py). The final holdout remains unopened. Its eventual result will assess this frozen procedure on this benchmark, with unknown specimen dependence still a limitation.

**Human**: Open the reserved test set and compare the frozen models. What can we conclude, and what should we check next?

**Instructor**: Predict the model comparison before revealing it. What would zero test errors establish, and what would it leave unresolved?

<!-- checkpoint: 7. The final comparison and next decision -->
**Assistant**: **The frozen RBF SVM performed best on the reserved test set: one error among 343 observations, compared with four errors for each linear model.** This supports our development-based choice, while leaving uncertainty about the size and stability of its advantage.

Each pipeline was refitted on all **1,029 development observations**. Standardization used only development means and standard deviations. The frozen settings and classification thresholds were unchanged.

| Frozen model | Test balanced accuracy | Test accuracy | Errors |
|---|---:|---:|---:|
| Linear SVM, \(C=100\) | 98.69% | 98.83% | 4 |
| Logistic regression, \(C=1{,}000\) | 98.69% | 98.83% | 4 |
| RBF SVM, \(C=1,\ \gamma=1\) | **99.74%** | **99.71%** | **1** |

The test set contains **190 class 0** and **153 class 1** observations.

![Confusion matrices for the three frozen models on the reserved test set](/examples/svm-week-v1/banknote_frozen_test_confusion_matrices_v1.png)

The confusion matrices reveal a tradeoff hidden by the overall scores. Each linear model correctly classified every class 0 observation but misclassified four class 1 observations. The RBF SVM correctly classified every class 1 observation but misclassified one class 0 observation.

Its class 1 recall was therefore **100%**, and its class 1 precision was **99.35%**. We cannot translate these errors into authentic-versus-forged consequences because the supplied documentation does not identify that label mapping.

The paired comparison is also useful. Against each linear model, the RBF model correctly classified **four observations that the linear model missed**, but missed **one observation that the linear model classified correctly**. Only five observations distinguish their correctness, so this single test split gives limited evidence about how consistently that advantage would recur.

We can conclude that **nonlinear flexibility helped on this reserved benchmark split**. We cannot conclude that the RBF model is error-free, universally superior, or ready for operational use. Its perfect development validation result became a strong, imperfect test result.

The next checks should focus on:

- **Error inspection:** examine the misclassified observations and nearby development cases for unusual feature values, possible label problems, or near duplicates. Any explanation remains exploratory.
- **Dependence and transport:** obtain specimen identifiers, acquisition batches, dates, and data from a separate collection. Exact-duplicate grouping does not resolve unknown dependence or distribution changes.
- **Uncertainty:** assess the paired performance difference with methods that respect available grouping. A narrow-looking percentage difference can rest on very few observations.
- **Decision requirements:** establish the label meanings and consequences of each error. If probabilities are needed, fit and validate calibration separately; an SVM score remains a decision score.

The test set is now used. Any model revision prompted by these results needs new untouched data for another independent assessment.

[Evaluation code](/examples/svm-week-v1/banknote_frozen_test_evaluation_v1.py) · [Test metrics](/examples/svm-week-v1/banknote_frozen_test_metrics_v1.csv) · [Predictions](/examples/svm-week-v1/banknote_frozen_test_predictions_v1.csv) · [Paired errors](/examples/svm-week-v1/banknote_frozen_test_paired_errors_v1.csv)

**Instructor**: Independent synthesis: write a short recommendation that cites the displayed evidence, names the validation population and states one next check. Explain why a high AUC is not proof of calibration, why a two-feature plot is not the full four-feature decision rule, and why exact-duplicate grouping cannot resolve unknown camera or specimen dependence. These questions practice interpreting and auditing supplied output; they do not specify future exam questions.

**Instructor**: [Data attribution and reproduction](/examples/svm-week-v1/DATA.md) · [Original data](/examples/svm-week-v1/banknote_raw.csv) · [Environment](/examples/svm-week-v1/requirements.txt) · [Script order](/examples/svm-week-v1/reproduction-order.txt). The same data and reserved split connect both lectures.

