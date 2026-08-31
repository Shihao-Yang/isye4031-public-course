---
title: Learning path
description: The public learning sequence for regression, statistical learning, and forecasting.
zh_url: zh/learning-path.html
---

<p class="eyebrow">Learning path</p>

# Explanation, prediction, and forecast operations

The course begins with statistical foundations, develops regression and statistical learning, and then moves into time-series forecasting. Each part repeats the same workflow: establish a baseline, use an agent, audit the result, and communicate what can be trusted.

## Part I: Statistical and regression foundations

- Sampling variation, uncertainty, confidence intervals, and hypothesis tests
- Simple and multiple linear regression
- Least squares, interpretation, residuals, and diagnostics
- Transformations, model building, and variable selection
- Reproducible implementation and regression audits

**Public sample:** [From association to simple linear regression](sample-lesson.html)

## Part II: Statistical learning

- Logistic regression and classification metrics
- Ridge regression and LASSO
- Cross-validation, tuning, and model comparison
- Principal component analysis
- Support vector machines and kernels

Students move beyond fitting a model to asking whether the validation scheme matches the real deployment problem.

## Part III: Forecasting

- Time-series regression
- Exponential smoothing
- ARIMA models
- Forecast evaluation, uncertainty, and operational constraints

Forecasting is treated as a decision process. A useful forecast needs a target, issue time, horizon, information set, uncertainty statement, and evaluation protocol.

## Early-course sequence

<div class="timeline">
<div><h3>Class 1: Why forecasting needs an auditable workflow</h3><p>Connect regression and forecasting to decisions, establish the course challenge, and introduce baseline, agent, audit, and provenance.</p></div>
<div><h3>Class 2: Statistical foundations for trustworthy forecasts</h3><p>Review sampling variability, uncertainty, confidence intervals, hypothesis tests, assumptions, and software defaults.</p></div>
<div><h3>Class 3: From p-values to simple linear regression</h3><p>Build the regression line from conditional means and least squares, then inspect residuals and audit the implementation.</p></div>
<div><h3>Homework 1: Statistical review and agent audit</h3><p>Create descriptive summaries, implement a transparent forecast baseline, complete one-sample inference, and critique an agent-generated analysis.</p></div>
</div>

## Repeated questions across the semester

Students return to five questions in every model family:

1. What exactly is the target?
2. What information is available when the prediction is made?
3. What assumptions connect the data to the model?
4. What baseline and tests could reveal a failure?
5. What evidence would justify using the result for a decision?
