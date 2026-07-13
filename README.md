# GEFCom2014_solution

## Problem Description

The goal of this project is to forecast hourly electricity demand one month ahead using the GEFCom2014 load dataset ([Hong et al., 2016](https://doi.org/10.1016/j.ijforecast.2016.02.001)). For each hour in the target month, the model must predict 99 quantiles, from the 1st to the 99th percentile, rather than a single load value. These quantiles describe the range of possible future demand and the uncertainty around the forecast. The predictions are evaluated using pinball loss.

## Dataset and EDA

The GEFCom2014 load track ([Hong et al., 2016](https://doi.org/10.1016/j.ijforecast.2016.02.001)) contains 15 sequential monthly forecasting rounds, covering one load zone and 25 temperature stations. It includes 96,408 hourly weather observations from 2001–2011 and 61,344 observed load values from 2005–2011. The earlier missing load values form an intentional weather-only period; otherwise, no missing weather data, duplicate timestamps, hourly gaps, nonpositive loads, or clear signs of corruption were found, so the data was retained as provided.

The main patterns are strong hourly, weekly, and seasonal demand cycles, high dependence on recent load values, and a nonlinear U-shaped relationship between load and temperature. A more detailed analysis of data quality, seasonality, weather effects, serial dependence, and extreme observations is available in the [EDA report](artifacts/eda/report.md).

## Evaluation Strategy

The models are evaluated with expanding-window rolling-origin backtesting, which plays the same role as k-fold cross-validation while preserving the chronological order of the data. At each monthly forecast origin, the model is trained on all observations available before that month and predicts the complete following month. The actual values are revealed only after prediction, then added to the training history for the next fold. Preprocessing and feature construction are repeated using only pre-origin data, preventing information from the forecast period or later from leaking into the model.

Round-1 experiments use January 2009 through December 2010 for validation, giving 24 monthly folds. The two complete annual cycles cover every season twice, making it possible to assess year-to-year stability and reducing the chance that conclusions are driven by unusual conditions in a single year. Round-2 experiments use January through December 2010, giving 12 monthly folds. This shorter window speeds up iteration, while its later start leaves an additional full year of pre-origin history for features aggregated across previous years, increasing their sample sizes and reducing sampling noise. Both designs retain a complete seasonal cycle because load patterns and forecasting difficulty may vary substantially by season; evaluating only a few months could therefore produce an unrepresentative result. Model selection is performed only on the relevant validation folds, and performance is reported both by month and across all folds. 

The final test period likewise spans a full annual cycle, from January through December 2011, so the out-of-sample assessment covers every season instead of depending on a potentially unrepresentative subset of months. It is not used for model selection or hyperparameter tuning.

The main evaluation assumes that the actual temperatures in the target month are not known at forecast time. The current baselines and models therefore use no future observed temperature.

## Baselines

An obvious starting point is seasonal persistence: demand at a given hour is likely to resemble demand at the same hour one year earlier. This idea was also provided as the example benchmark in the competition.

**Baseline 1: Seasonal naive.** For each target operating hour, the load from the same calendar hour one year earlier is used as the forecast and repeated across all 99 quantiles. Pinball loss remains mathematically well-defined for this forecast, but the identical quantiles form a degenerate distribution with zero spread. The baseline therefore provides a useful point-forecast reference but cannot represent forecast uncertainty or produce calibrated prediction intervals.

This limitation motivates a simple probabilistic baseline. Instead of relying on one historical value, the seasonal empirical method treats several comparable observations from neighbouring dates and previous years as a sample from the predictive distribution. Matching the operating hour and separating weekdays from weekends provides basic similarity criteria while keeping the method transparent and free of future information.

**Baseline 2: Seasonal empirical.** For every target operating hour:

1. Centre an inclusive ±8-calendar-day window on the same month and day in each earlier seasonal cycle.
2. Retain the same zone and operating hour.
3. Retain weekdays for weekday targets and weekends for weekend targets.
4. Pool every available candidate from those previous cycles.
5. Calculate quantiles 0.01 through 0.99 using linear interpolation.

Implementation details and complete baseline results are available in the [baseline report](artifacts/baseline/report.md).

## References

Hong, T., Pinson, P., Fan, S., Zareipour, H., Troccoli, A., & Hyndman, R. J. (2016). [Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond](https://doi.org/10.1016/j.ijforecast.2016.02.001). *International Journal of Forecasting, 32*(3), 896–913.
