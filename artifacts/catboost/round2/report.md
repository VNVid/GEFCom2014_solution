# CatBoost round-2 validation report

## Executive summary

Round two combines a deterministic feature-selection recipe with a compact
CPU CatBoost `MultiQuantile` search. Feature selection reduced the 70-feature
candidate matrix to 17 features, after which all
27 effective hyperparameter candidates were evaluated on the
same 12 monthly rolling-origin validation folds from January through December
2010. The configured 2011 test period was not accessed.

The selected configuration is **depth 3, learning rate
0.08, L2 leaf regularization
5, and 125 trees**. Its
hour-weighted pinball loss is **8.373**, compared with
**8.985** for the selected round-one model and
**10.121** for seasonal empirical. This is a
**6.82%** reduction
relative to round one and a
**17.28%**
reduction relative to the empirical baseline. Round two wins 9 of 12 months
against each and all 12 months against seasonal naive.

The HAC mean-loss tests favor round two over round one (p=
0.0111) and seasonal empirical (p=
0.0141), but the exact sign tests give p=
0.146. Feature selection and model
selection both reused 2010 validation. The improvement is therefore strong
exploratory evidence, not a confirmatory generalization result.

## Validation design and leakage protection

Every 2010 month is treated as a separate forecast origin. Training for a
month uses only pseudo-forecast labels whose forecast month has ended by that
origin. All load- and temperature-derived features are constructed from
observations strictly before the origin; only deterministic target calendar
and horizon information uses the future timestamp. Realized target-month
temperature is never supplied.

The comparison is exactly matched: round two, the selected round-one model,
seasonal empirical, and seasonal naive are all restricted to the same 8,760
hours and 12 origins. Round-one results are taken from its already-saved
candidate `depth4_lr0p04_l25_trees250` rather than retuned on 2010. The 2011 test
period remains untouched.

## Feature selection

The candidate matrix contained 70 leakage-safe features. A fixed fast model
(depth 4, learning rate 0.08, L2=5, 125 trees) was fitted on the six odd
months. CatBoost validation `LossFunctionChange` was aggregated by feature.
A feature was retained when its median importance was positive and its
importance was positive in at least four of six screening folds; the fixed
cap of 55 was not reached. This mechanical rule selected
17 features.

On the six complementary even-month folds, the selected fast model improved
the matching round-one fast model by
4.08%
and won 4 of 6 folds.
Those folds later entered the full grid search, so this check is development
evidence rather than an independent test.

The selected features are:

- `load_seasonal_how_30d_mean`
- `load_seasonal_how_15d_mean`
- `month`
- `load_seasonal_daytype_8d_q90`
- `load_seasonal_how_15d_count`
- `load_seasonal_daytype_8d_q75`
- `load_yoy_ratio_365d`
- `seasonal_day_sin`
- `temperature_clim_15d_q10`
- `temperature_recent_std_7d`
- `load_seasonal_daytype_8d_q25`
- `load_seasonal_daytype_8d_q50_scaled_28d`
- `load_seasonal_how_15d_std`
- `temperature_clim_15d_hdd65`
- `load_seasonal_how_30d_std`
- `temperature_clim_15d_q90`
- `hour`

Seven features were new relative to round one: `load_seasonal_how_30d_mean`, `load_seasonal_daytype_8d_q75`, `temperature_clim_15d_q10`, `load_seasonal_daytype_8d_q25`, `temperature_clim_15d_hdd65`, `load_seasonal_how_30d_std`, `temperature_clim_15d_q90`. The set is
dominated by historical seasonal profile location, spread, and support,
augmented by annual load growth, calendar seasonality, and temperature
climatology or recent variability.

## Primary matched validation comparison

Pinball loss averages all 99 quantiles and all 2010 forecast hours; lower is
better. Monthly mean and standard deviation weight each origin equally.
Median bias is `actual - q0.50`, so a positive value denotes under-forecasting.

| Model | Pinball | Improvement vs empirical (%) | Monthly mean | Monthly SD | Median MAE | Median bias |
|---|---|---|---|---|---|---|
| CatBoost round 2 | 8.373 | +17.28 | 8.348 | 4.420 | 23.00 | +4.36 |
| CatBoost round 1 | 8.985 | +11.22 | 8.960 | 4.935 | 24.46 | +8.75 |
| Seasonal empirical | 10.121 | +0.00 | 10.125 | 6.131 | 27.98 | +20.83 |
| Seasonal naive | 15.310 | -51.27 | 15.343 | 7.909 | 30.62 | +12.02 |

Round two reduces median MAE by
1.46 MW relative to round one and
cuts median under-forecast bias from +8.75
to +4.36 MW. Relative to seasonal
empirical, pinball improves by
17.28% and
median MAE improves by 4.99 MW.

![Monthly pinball loss](figures/01_monthly_pinball.png)

## Stability across months and quarters

| Quarter | Round 2 | Round 1 | Empirical | Improvement vs round 1 (%) | Wins vs round 1 |
|---|---|---|---|---|---|
| Q1 | 11.037 | 12.330 | 13.878 | +10.49 | 3/3 |
| Q2 | 6.184 | 6.695 | 8.138 | +7.63 | 2/3 |
| Q3 | 7.081 | 7.154 | 8.359 | +1.02 | 2/3 |
| Q4 | 9.223 | 9.811 | 10.170 | +5.99 | 2/3 |

Round two improves on round one in every quarter, with the largest aggregate
gain in Q1. Its largest monthly improvement against round one is
January (-2.357
loss points); its largest deterioration is
November (+0.247).
The three losses against round one occur in May, August, and November; two are
small. Against seasonal empirical, the losses occur in March, October, and
November. This is meaningful variability despite the strong annual aggregate.

![Fold loss differences](figures/02_fold_differences.png)

## Paired statistical comparison

| Reference | Mean difference | Wins | Losses | HAC p | Sign p | HAC 95% CI |
|---|---|---|---|---|---|---|
| CatBoost round 1 | -0.613 | 9 | 3 | 0.01112 | 0.146 | [-1.055, -0.170] |
| Seasonal empirical | -1.777 | 9 | 3 | 0.01412 | 0.146 | [-3.120, -0.434] |
| Seasonal naive | -6.995 | 12 | 0 | 0.0008123 | 0.0004883 | [-10.369, -3.621] |

The paired unit is a monthly forecast origin rather than an individual hour.
The Diebold–Mariano-style mean-loss test uses a Bartlett HAC variance with lag
2 and a t(11) small-sample reference. It uses the magnitude of monthly
gains and allows short-range serial dependence. The exact two-sided sign test
uses only the 12 win/loss outcomes; with 9 wins and 3 losses its p-value is
0.146, illustrating the low power of a 12-fold comparison.

The HAC intervals exclude zero for round two versus round one and empirical.
However, the same validation year informed the earlier feature search,
inspection of round-one results, feature verification, and final model
selection. Consequently, these validation p-values are descriptive and should
not be presented as confirmatory.

Against seasonal naive the result is much less ambiguous: round two improves
pinball by 45.31%,
wins every month, and has HAC p=0.0008123.

## Calibration, sharpness, and quantile coherence

Calibration MAE is calculated within each month over all 99 marginal
quantiles, then averaged with forecast-hour weights.

| Model | Calibration MAE | 90% coverage | 90% width | Invalid intervals | Crossings | Crossing rate (%) |
|---|---|---|---|---|---|---|
| CatBoost round 2 | 0.134 | 0.818 | 80.96 | 0 | 2849 | 0.332 |
| CatBoost round 1 | 0.161 | 0.729 | 71.27 | 0 | 1236 | 0.144 |
| Seasonal empirical | 0.185 | 0.778 | 75.59 | 0 | 0 | 0.000 |
| Seasonal naive | 0.285 | 0.002 | 0.00 | 0 | 0 | 0.000 |

Round two materially improves calibration relative to round one: monthly
calibration MAE falls from
0.161 to
0.134, while nominal 90% coverage rises
from 72.9% to
81.8%. Coverage remains below the nominal 90%
target. Round two's intervals are wider than both round one and empirical, so
part of the calibration gain comes from reduced sharpness rather than location
accuracy alone.

No model has a reversed 5th/95th percentile interval. Round two has
2,849 adjacent crossings, a
0.332% rate. This is higher than round
one but still affects fewer than one half of one percent of adjacent quantile
pairs. Scores use raw output; monotonic rearrangement has not been applied.

![Calibration and sharpness](figures/04_calibration_sharpness.png)

### Full marginal and interval calibration curves

The aggregate marginal curve evaluates `P(Y ≤ qτ)` at every requested
quantile τ. Round two's predicted median has empirical marginal coverage
50.0%; the complete curve makes remaining
lower- and upper-tail asymmetry visible rather than reducing calibration to one
average. The central-interval panel separately compares empirical coverage at
the configured 50%, 80%, 90%, and 98% levels. Both panels use exactly the same
2010 observations for round two, round one, and the baselines.

![Marginal quantile and central interval calibration](figures/05_calibration_curves.png)

Exact curve values and interval widths are in `quantile_calibration.csv` and
`interval_calibration.csv`.

## What the search learned

The search jointly varied depth {3, 4, 5}, L2 regularization {1, 5, 20},
and tree checkpoints {75, 100, 125} at learning rate 0.08. Every effective
candidate used all 12 folds. Tree prefixes shared parent fits, so 27 candidates
required 108 actual fold fits and 19.7 minutes of summed
model-fit time.

Best checkpoint for each depth/L2 pair:

| Depth | L2 | Best trees | Pinball | 90% coverage | Crossings |
|---|---|---|---|---|---|
| 3 | 1 | 125 | 8.375 | 0.819 | 2453 |
| 3 | 5 | 125 | 8.373 | 0.818 | 2849 |
| 3 | 20 | 125 | 8.461 | 0.814 | 2061 |
| 4 | 1 | 100 | 8.560 | 0.787 | 2304 |
| 4 | 5 | 100 | 8.393 | 0.799 | 1945 |
| 4 | 20 | 75 | 8.407 | 0.831 | 1225 |
| 5 | 1 | 75 | 8.582 | 0.784 | 2367 |
| 5 | 5 | 75 | 8.612 | 0.788 | 878 |
| 5 | 20 | 75 | 8.618 | 0.786 | 758 |

The top candidates are:

| Candidate | Pinball | 90% coverage | Calibration MAE | Crossings |
|---|---|---|---|---|
| depth3_lr0p08_l25_trees125 | 8.3727 | 0.818 | 0.134 | 2849 |
| depth3_lr0p08_l21_trees125 | 8.3746 | 0.819 | 0.135 | 2453 |
| depth4_lr0p08_l25_trees100 | 8.3934 | 0.799 | 0.125 | 1945 |
| depth4_lr0p08_l220_trees75 | 8.4065 | 0.831 | 0.124 | 1225 |
| depth4_lr0p08_l220_trees100 | 8.4089 | 0.811 | 0.130 | 2325 |
| depth3_lr0p08_l21_trees100 | 8.4162 | 0.829 | 0.130 | 1158 |
| depth4_lr0p08_l25_trees75 | 8.4353 | 0.823 | 0.125 | 706 |
| depth3_lr0p08_l220_trees125 | 8.4607 | 0.814 | 0.139 | 2061 |

Depth 3 is preferred: its L2=1 and L2=5 results at 125 trees are nearly tied,
differing by only
0.022%.
This indicates that the leading result is not highly sensitive to modest L2
changes. Depth 4 remains competitive around 75–100 trees, while depth 5 is
worse and deteriorates as trees are added. The deeper models also narrow
intervals and create more crossings, consistent with quantile overfitting.

![Search landscape](figures/03_search_landscape.png)

## Limitations and next decision

- Feature selection, feature-set verification, and hyperparameter selection
  all used 2010. The final 8.373 validation score is therefore optimistic.
- Only 12 monthly folds are available, so paired inference has low power and
  is sensitive to a few large winter-month differences.
- Forecasts use temperature climatology and observed pre-origin weather, not
  realized future temperature. Unexpected target-month weather remains an
  irreducible source of error under this assumption.


## Reproduction and artifacts

From the repository root:

```bash
.venv/bin/python analysis/catboost/search.py --config configs/catboost_round2_phase1.yaml
.venv/bin/python -m analysis.catboost.predict_selected
.venv/bin/python -m analysis.catboost.round2_report
```

The search is resumable. Raw outputs are in
[`../search/round2_phase1`](../search/round2_phase1), the feature-selection
artifacts are in [`../feature_selection/round2`](../feature_selection/round2),
and the selected feature manifest is
[`selected_features.yaml`](../feature_selection/round2/selected_features.yaml).
Supporting outputs in this directory are `model_comparison.csv`,
`fold_comparison.csv`, `quarter_comparison.csv`, `paired_tests.csv`,
`candidate_shortlist.csv`, and `structure_summary.csv`.
