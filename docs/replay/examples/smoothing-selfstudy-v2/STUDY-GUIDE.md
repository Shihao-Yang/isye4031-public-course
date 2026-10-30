# Exponential smoothing and Holt–Winters

Lectures 19 and 20 form a two-lecture self-study unit. Both lessons cover smoothing. Stationary models and ARIMA will follow later.

Use the replay in short sessions. At each checkpoint, write a prediction or calculation before revealing the answer. The linked CSV files give full precision, and the scripts reproduce the calculations. You can complete the conceptual route without reading every line of Python.

## Lecture 19

Allow about two hours, including pauses. Split the work into three sessions:

1. Cod catch and level updates: checkpoints 1–4, about 35 minutes. Identify the response and units, compute an update, and explain how alpha controls the weights. Distinguish a forecast initialized after year one from the source's retrospective time-zero fit.
2. Fitting and forecast accuracy: checkpoints 5–7, about 35 minutes. Read the fitting curve, revise a forecast after new data, interpret prediction intervals, and compare MAD, MSD and MAPE.
3. Changing trend: checkpoints 8–11, about 50 minutes. Trace Holt's level and trend states, fit the parameters, revise forecasts and intervals, and explain why software reports differ.

The interval derivations and the detailed software sensitivity calculations are useful optional depth. The interpretation of intervals, initialization and forecast evaluation is part of the core route.

By the end, you should be able to calculate a level or trend update without AI, explain what a smoothing parameter controls, and identify which observations were available when a forecast was issued.

## Lecture 20

Allow about two hours, split into three sessions:

1. Seasonal patterns, the two bike datasets, and additive updates: checkpoints 1–4, about 45 minutes.
2. Seasonal intervals and the Tiger multiplicative example: checkpoints 5–8, about 45 minutes.
3. The Cola comparison, chronological evaluation, and forecast audit: checkpoints 9–11, about 35 minutes.

Keep the state variables separate: the current level, the change per period, and a stored factor for each season.

The seasonal models are core material for this lesson. The extended variance derivations and additional sensitivity checks are optional depth. Do not skip the worked initialization, first updates, interpretation of the forecasts, or comparisons with a seasonal baseline.

The mountain-bike source has two different numerical versions. The earlier Table 6.8 series is in bikes.csv; the Chapter 8 worked spreadsheet is in bikes_ch8.csv. The latter reproduces the stated initialization of level 20.85 and trend 0.980882. The replay shows the discrepancy explicitly.

## Independent practice

Write your answers before reading the explanations below. These are ungraded study questions about interpreting evidence and explaining methods. They do not specify future exam coverage.

1. A cod forecast is 360.6667 tons and the next observation is 362. With alpha = 0.10, calculate the error, new level, and next forecast. Why are the fitted errors for months 1–12 not forecasts issued at those dates if the initial level uses all twelve months?
2. In a constructed series, the initial level is 350 and every subsequent observation is 400. Explain why alpha = 0.70 adapts faster than alpha = 0.10. After two observations, what are the two levels?
3. The optimized cod model has a flat point forecast. Explain why its prediction interval can still widen with the horizon. Distinguish a prediction interval for a future catch from uncertainty about the underlying level.
4. A model has smaller training MSD but larger later forecast MSD. Explain how that can happen. Why does selecting a model after inspecting the later results change the status of those results?
5. In a constructed Holt example, the old level is 100, old trend is 4, and the next observation is 110. Use alpha = 0.50 and gamma = 0.25 to update both states and forecast the following period. Explain why the direct error coefficient for the trend is alpha times gamma.
6. The bike example starts with level 20.85, trend 0.980882, and quarter-one factor −14.216176. Sales are 10, with alpha = 0.20, gamma = 0.10, and delta = 0.10. Calculate the forecast error and the three updated states. Which seasonal factor changes?
7. An additive seasonal model estimates gamma = 0 and delta = 0. Does it have no trend and no seasonality? Explain what remains in the forecast.
8. Seasonal peaks increase in size as the series level rises. Explain why multiplicative seasonality may be useful. Must the amplitude increase with calendar time? What happens if the level falls?
9. An AI report says, “The optimized fit has the lowest MAPE, so it will forecast best and its 95% intervals will cover 95% of next year's observations.” Identify the evidence needed for each claim.
10. Two software reports use the same alpha and gamma but produce different forecasts. List the settings or inputs you would check before concluding that one implementation is wrong.

## Explained answers

1. Error = 362−360.6667 = 1.3333; level = 360.6667+0.10(1.3333) = 360.8000; next forecast = 360.8000. The initial average was not available before the first twelve catches were observed. It can initialize a historical fit, but those early errors do not establish operational forecast accuracy.
2. The remaining fraction of the gap after two updates is (1−alpha)². The levels are 400−50(0.30)² = 395.50 and 400−50(0.90)² = 359.50. A larger alpha also reacts more strongly to a temporary spike.
3. The model projects no systematic future change in level, but unobserved future innovations will affect both catches and subsequent levels. Conditional on the fitted parameters, the SES forecast variance is sigma²[1+(h−1)alpha²]. A future observation also includes its own innovation. Parameter and model uncertainty require additional treatment.
4. Fitting chooses parameters to explain the training observations. Random variation and changes over time can make the later ranking different. Reusing the later comparison to select a winner introduces selection optimism; further evaluation is needed for the revised procedure.
5. One-step prediction = 104 and error = 6. New level = 104+0.50(6) = 107. New trend = 4+(0.50)(0.25)(6) = 4.75. The following forecast is 111.75. Substituting the level update into the trend equation produces the alpha-gamma product.
6. Prediction is approximately 7.614706 and error is +2.385294. The new level is 22.307941; trend is 1.028588; quarter-one factor is −14.025353. The other three seasonal factors remain stored until their corresponding observations arrive. Rounded intermediate values can change the final displayed decimals.
7. A zero trend smoothing constant keeps the initialized trend fixed, and a zero seasonal smoothing constant keeps the seasonal factors fixed. Neither state is necessarily zero. The level can still adapt.
8. Multiplicative factors represent proportional differences from the level. With fixed factors, absolute seasonal amplitude increases when the level rises and decreases when the level falls. It need not grow over calendar time. Check positivity and whether the ratio interpretation fits the data.
9. Forecast superiority needs a chronological comparison of the complete fitting procedure against relevant baselines. Interval coverage also depends on an error model, estimation uncertainty and empirical calibration. Small fitted MAPE alone establishes neither claim; 95% pointwise coverage also does not mean a 95% chance that every future observation falls within its interval.
10. Check the exact data and order, initial states, parameter notation, bounds and optimization, which errors enter the objective, the seasonal update convention, forecast origin and horizon, and the interval assumptions. Label transcribed historical reports separately from newly executed results.

## Source and reproduction notes

The examples use Bowerman, O’Connell and Koehler, Forecasting, Time Series, and Regression, fourth edition, numerical tables 6.1, 6.8, 7.1, 8.1 and 8.2 and the distinct bike series in Figure 8.9. Newly executed figures recreate the instructional comparisons. Original page images and proprietary software screenshots are not included. Historical software numbers are labeled as transcribed reference values where the internal settings are not fully documented.

The replay preserves actual Codex CLI responses. Instructor notes identify transcription errors, rounding conventions and source discrepancies. The plots and results are saved outputs from completed calculations. Use the supplied requirements and reproduction-order.txt to rerun the scripts in a fresh directory.

For software conventions, see Minitab's [Winters' method formulas](https://support.minitab.com/en-us/minitab/help-and-how-to/statistical-modeling/time-series/how-to/winters-method/methods-and-formulas/methods-and-formulas/). Its published initialization and interval procedures illustrate why a method name and three constants do not fully specify a computation. The current documentation does not establish the historical report's exact configuration.
