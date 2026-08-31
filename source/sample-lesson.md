---
title: Sample lesson
description: A self-contained AI-agent-first lesson on simple linear regression and audit.
zh_url: zh/sample-lesson.html
---

<p class="eyebrow">Sample lesson</p>

# From association to simple linear regression

## Learning goals

By the end of this lesson, a learner should be able to:

1. distinguish association from a conditional-mean model;
2. fit and interpret a simple linear regression;
3. connect least squares to residuals and sample covariance;
4. use an AI coding agent to implement the analysis reproducibly; and
5. audit the implementation using mathematical identities and diagnostic questions.

## 1. Begin with the statistical model

For observations \((x_i, y_i)\), simple linear regression assumes

\[
Y_i = \beta_0 + \beta_1 x_i + \varepsilon_i,
\qquad E(\varepsilon_i \mid x_i)=0.
\]

The line \(\beta_0 + \beta_1 x\) describes the conditional mean of \(Y\) at a given value of \(x\). It does not automatically establish causality, and it does not guarantee that extrapolation is valid.

Least squares chooses \(\hat\beta_0\) and \(\hat\beta_1\) to minimize

\[
\sum_{i=1}^n \left[y_i-(b_0+b_1x_i)\right]^2.
\]

For a model with an intercept,

\[
\hat\beta_1 = \frac{\sum_i (x_i-\bar x)(y_i-\bar y)}{\sum_i (x_i-\bar x)^2},
\qquad
\hat\beta_0 = \bar y-\hat\beta_1\bar x.
\]

## 2. Work a transparent five-point example

Suppose \(x\) is hours of controlled testing and \(y\) is an observed performance score.

| Observation | \(x\) | \(y\) |
| ---: | ---: | ---: |
| 1 | 1 | 2.2 |
| 2 | 2 | 2.8 |
| 3 | 3 | 4.2 |
| 4 | 4 | 4.7 |
| 5 | 5 | 6.1 |

Before using software:

1. Sketch the data and predict the sign of the slope.
2. Calculate \(\bar x\) and \(\bar y\).
3. Calculate the least-squares slope and intercept.
4. Interpret the slope using the units of \(x\) and \(y\).
5. Explain why a prediction at \(x=20\) would be more questionable than a prediction at \(x=3.5\).

## 3. Ask an agent to implement, explain, and test

Use a clean project directory. Give the agent the data and the following task:

> Fit a simple linear regression with an intercept. Report the coefficients, fitted values, and residuals. Plot the observations and fitted line. Write tests for the core least-squares identities. Explain the slope in context, flag extrapolation risk, and list the assumptions that cannot be verified from five observations alone. Keep the analysis reproducible and do not invent additional data.

Require the project to include:

- an executable analysis script;
- at least three automated checks;
- a saved figure;
- a short interpretation; and
- an AI usage statement describing what was generated and what was verified.

## 4. Audit the result

For ordinary least squares with an intercept, check that numerical error is near zero for:

\[
\sum_i e_i = 0,
\qquad
\sum_i x_i e_i = 0,
\qquad
\bar{\hat y} = \bar y.
\]

Then ask:

- Did the implementation include an intercept?
- Are the coefficient units explained correctly?
- Does the fitted line pass through \((\bar x, \bar y)\)?
- Are residuals computed as observed minus fitted rather than the reverse?
- Does the plot use the same data and model as the reported coefficients?
- Is a confidence interval being confused with a prediction interval?
- Are causal or extrapolative claims stronger than the data support?

<details>
<summary>Reference values for checking the implementation</summary>

For these five observations, the least-squares line is approximately

\[
\hat y = 1.09 + 0.97x.
\]

The exact numerical display may differ slightly because of rounding. The audit should use a tolerance rather than exact string equality.
</details>

## 5. Defend the analysis

A learner should be able to answer these questions without asking the agent:

1. What does the slope estimate?
2. Why is the intercept difficult to interpret if \(x=0\) is outside the observed range?
3. Which least-squares identity gives you confidence that the implementation used an intercept correctly?
4. What evidence would you need before making a causal statement?
5. What changes when predicting a new observation rather than the conditional mean?

## Exit check

Write three sentences:

1. one sentence stating the fitted relationship;
2. one sentence stating the strongest limitation; and
3. one sentence identifying the most informative next diagnostic or data collection step.
