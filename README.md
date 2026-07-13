# GEFCom2014_solution

## Problem Description

The goal of this project is to forecast hourly electricity demand one month ahead using the GEFCom2014 load dataset ([Hong et al., 2016](https://doi.org/10.1016/j.ijforecast.2016.02.001)). For each hour in the target month, the model must predict 99 quantiles, from the 1st to the 99th percentile, rather than a single load value. These quantiles describe the range of possible future demand and the uncertainty around the forecast. The predictions are evaluated using pinball loss.

## Dataset and EDA

The GEFCom2014 load track ([Hong et al., 2016](https://doi.org/10.1016/j.ijforecast.2016.02.001)) contains hourly electricity load for a single zone from 2005 to 2011, together with hourly observations from 25 temperature stations covering 2001 to 2011. In total, it includes 96,408 weather observations and 61,344 observed load values. The earlier missing load values form an intentional weather-only period; otherwise, no missing weather data, duplicate timestamps, hourly gaps, nonpositive loads, or clear signs of corruption were found, so the data was retained as provided.

The main patterns are strong hourly, weekly, and seasonal demand cycles, high dependence on recent load values, and a nonlinear U-shaped relationship between load and temperature. A more detailed analysis of data quality, seasonality, weather effects, serial dependence, and extreme observations is available in the [EDA report](artifacts/eda/report.md).

## Reproducing the results

Use Python 3.9 and run all commands from the repository root. Download the
[GEFCom2014 dataset from Kaggle](https://www.kaggle.com/datasets/cthngon/gefcom2014-dataset),
then copy its `GEFCom2014-L_V2` directory to `data/`. The expected load path is
`data/GEFCom2014-L_V2/Load/`, containing `Task 1` through `Task 15` and
`Solution to Task 15`.

```bash
python3.9 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pytest

# EDA and validation baselines
.venv/bin/python -m analysis.eda.run
.venv/bin/python -m analysis.baseline.run --split validation

# Round 1
.venv/bin/python -m analysis.model_data.run --feature-config configs/features/first_round.yaml --split validation
.venv/bin/python -m analysis.catboost.search --config configs/catboost.yaml
.venv/bin/python -m analysis.catboost.predict_selected --experiment round1
.venv/bin/python -m analysis.catboost.report

# Round 2
.venv/bin/python -m analysis.model_data.run --feature-config configs/features/round2_candidates.yaml --split round2_validation
.venv/bin/python -m analysis.catboost.feature_selection
.venv/bin/python -m analysis.catboost.search --config configs/catboost_round2_phase1.yaml
.venv/bin/python -m analysis.catboost.predict_selected --experiment round2
.venv/bin/python -m analysis.catboost.round2_report

# Locked test
.venv/bin/python -m analysis.model_data.run --feature-config configs/features/round2_candidates.yaml --split test
.venv/bin/python -m analysis.baseline.run --split test
.venv/bin/python -m analysis.catboost.predict_selected --experiment round1_test --experiment round2_test
.venv/bin/python -m analysis.catboost.test_report
```

Generated outputs are written under `artifacts/`; the main reports are linked
throughout this README. The CatBoost searches, feature selection, and prediction
runners are resumable and validate existing checkpoints before reusing them.
Move or remove a matching configured output directory to force a clean refit.
The full CPU workflow takes several hours, dominated by the round-one grid
search.

## Evaluation Strategy

The models are evaluated with expanding-window rolling-origin backtesting, which plays the same role as k-fold cross-validation while preserving the chronological order of the data. At each monthly forecast origin, the model is trained on all observations available before that month and predicts the complete following month. The actual values are revealed only after prediction, then added to the training history for the next fold. Preprocessing and feature construction are repeated using only pre-origin data, preventing information from the forecast period or later from leaking into the model.

Round-1 experiments use January 2009 through December 2010 for validation, giving 24 monthly folds. The two complete annual cycles cover every season twice, making it possible to assess year-to-year stability and reducing the chance that conclusions are driven by unusual conditions in a single year. Round-2 experiments use January through December 2010, giving 12 monthly folds. This shorter window speeds up iteration, while its later start leaves an additional full year of pre-origin history for features aggregated across previous years, increasing their sample sizes and reducing sampling noise. Both designs retain a complete seasonal cycle because load patterns and forecasting difficulty may vary substantially by season; evaluating only a few months could therefore produce an unrepresentative result. Model selection is performed only on the relevant validation folds, and performance is reported both by month and across all folds. 

The final test period likewise spans a full annual cycle, from January through December 2011, so the out-of-sample assessment covers every season instead of depending on a potentially unrepresentative subset of months. It is not used for model selection or hyperparameter tuning.

The main evaluation assumes that the actual temperatures in the target month are not known at forecast time. The current baselines and models therefore use no future observed temperature.

## Baselines

An obvious starting point is seasonal persistence: demand at a given hour is likely to resemble demand at the same hour one year earlier. This idea was also provided as the example benchmark in the competition.

**Baseline 1: Seasonal naive.** For each target operating hour, the load from the same calendar hour one year earlier is used as the forecast and repeated across all 99 quantiles. Pinball loss remains mathematically well-defined for this forecast, but the identical quantiles form a degenerate distribution with zero spread. The baseline therefore provides a useful point-forecast reference but cannot represent forecast uncertainty or produce calibrated prediction intervals.

This limitation motivates a simple probabilistic baseline which is defined below. Instead of relying on one historical value, the seasonal empirical method treats several comparable observations from neighbouring dates and previous years as a sample from the predictive distribution. Matching the operating hour and separating weekdays from weekends provides basic similarity criteria while keeping the method transparent and free of future information.

**Baseline 2: Seasonal empirical.** For every target operating hour:

1. Centre an inclusive ±8-calendar-day window on the same month and day in each earlier seasonal cycle.
2. Retain the same zone and operating hour.
3. Retain weekdays for weekday targets and weekends for weekend targets.
4. Pool every available candidate from those previous cycles.
5. Calculate quantiles 0.01 through 0.99 using linear interpolation.

Implementation details and complete baseline results are available in the [baseline report](artifacts/baseline/report.md).

## Solution

### Forecasting formulation

Each target month is treated as a separate forecasting problem. Let $o_m$ be
the forecast origin for month $m$: 00:00 on its first calendar day,
immediately before any outcomes from that month are observed. The corresponding
set of target hours is

$$\mathcal{T}_m = \{t : o_m \leq t < o_{m+1}\}.$$

For every $t \in \mathcal{T}_m$, the feature vector is constructed as

$$X(o_m,t) = \left[c(t),\ h(o_m,t),\ \phi_Y\left(\{Y_s:s < o_m\},t\right),\ \phi_T\left(\{T_s:s < o_m\},t\right)\right].$$

Here, $c(t)$ contains deterministic target-time information such as hour,
weekday, and season; $h(o_m,t)=t-o_m$ is the forecast horizon; and
$\phi_Y$ and $\phi_T$ summarize historical load and temperature. The
historical transformations may depend on the target's known calendar position,
but they may only read observations timestamped strictly before $o_m$. In
particular, realized load and temperature from the target month are unavailable.
Temperature inputs are therefore limited to pre-origin observations and
historical climatology.

Training data is built from earlier months using the same information boundary.
For a historical month $j$, its features are reconstructed at its own origin
$o_j$, rather than using information available at the later fitting date. A
historical month enters the training set only after all of its outcomes have
been observed. Thus, at origin $o_m$, the expanding training set is

$$
\mathcal{D}(o_m) = \left\{\bigl(X(o_j,t),Y_t\bigr): t\in\mathcal{T}_j,\ o_{j+1}\leq o_m\right\}.
$$

The learned models jointly estimate the 99 conditional quantiles
$\mathcal{Q}=\{0.01,\ldots,0.99\}$:

$$
\widehat q_{t,\tau} = f_\tau\left(X(o_m,t);\widehat\theta_{o_m}\right), \qquad \tau\in\mathcal{Q}.
$$

For an evaluated origin $o_m$, all eligible historical examples are assembled
into the expanding dataset $\mathcal{D}(o_m)$. A single model is fitted on this
dataset and is then used to forecast every hour in $\mathcal{T}_m$.
Rolling-origin validation and testing repeat this fit-and-forecast procedure for
each evaluation month. The feature recipe, selected feature set, and
hyperparameters remain fixed across folds; only the fitted model parameters and
the available history are updated. This reproduces a production setting in
which a new one-month-ahead forecast is issued after the preceding month's
observations have become available.

### Model choice and objective

The learned forecaster is a CPU `CatBoostRegressor` trained with CatBoost's
`MultiQuantile` objective. The modeling table is medium-sized and heterogeneous:
each row represents one future hour, numerical features summarize recent and
seasonal conditions, and calendar fields are categorical. Load also depends on
these inputs through nonlinear effects and interactions—for example,
temperature response varies by season and hour. Tree-based Gradient Boosting is a natural fit:
it learns such relationships without manually enumerating interaction terms,
handles categorical variables directly, requires little preprocessing, and
remains practical on a laptop CPU.

For $N$ training examples $(x_i,y_i)$ and quantile levels
$\mathcal{Q}=\{0.01,\ldots,0.99\}$, the optimized objective is the mean pinball
loss across observations and quantiles:

$$\widehat\theta = \underset{\theta}{\operatorname{arg\,min}}\ \frac{1}{N|\mathcal{Q}|}\sum_{i=1}^{N}\sum_{\tau\in\mathcal{Q}}\rho_\tau\left(y_i-f_\tau(x_i;\theta)\right), \qquad \rho_\tau(u)=u\left(\tau-\mathbf{1}\{u < 0\}\right).$$

This directly aligns model fitting with the task metric and produces all
99 quantiles from one model. The forecast is also **direct**, rather than
recursive: every target-hour feature vector is evaluated using information
available at the monthly origin, and predicted loads are never fed back as
inputs for later hours. Consequently, errors cannot accumulate through a
672–744-hour autoregressive rollout.

The main alternatives were not prioritized for the following reasons:

- A standard random forest optimizes a point-prediction criterion rather than
  pinball loss. Quantile regression forests can recover conditional
  distributions from leaf samples, but do not directly optimize the evaluation
  metric, and extreme quantiles can be unstable when leaf support is limited.
- Linear quantile regression does optimize pinball loss, but a useful model
  would require explicit nonlinear transformations and many calendar,
  temperature, and horizon interactions. Fitting quantiles independently can
  also produce incoherent crossings.
- A feed-forward neural network could use the same multi-quantile objective, but
  would introduce scaling, architecture, regularization, and optimization
  choices. Neural networks also often underperform gradient-boosted trees on
  medium-sized heterogeneous tabular datasets, so there was no clear expected
  benefit to justify the additional tuning complexity here.
- A recursive RNN would expose a month-long forecast to accumulated prediction
  error. A direct sequence-to-sequence model can avoid that problem, but adds
  substantial architectural and training complexity for a task that can be
  expressed naturally as leakage-safe target-hour rows.

CatBoost's loss does not enforce monotonicity across quantiles. Forecasts are
therefore evaluated as emitted, without sorting or post-processing, and
quantile crossings are reported as a separate coherence diagnostic.

### Experimental pipeline

#### Round 1: domain-guided feature engineering

The first round introduced domain knowledge through a deliberately constructed
set of 45 leakage-safe features. The set combined target calendar and horizon
information with recent load level and trend, recent hourly and weekly
profiles, historical seasonal profiles, annual load anchors, temperature
climatology, and the observed pre-origin temperature regime. The intention was
to give a shallow tree model strong candidate summaries while still allowing it
to learn nonlinear interactions between them. Complete definitions are in the
[round-1 validation report](artifacts/catboost/round1/report.md).

A hyperparameter grid search was evaluated on 24 rolling-origin
folds from January 2009 through December 2010. For each fold, all eligible
earlier months formed the expanding training set and the complete next month
was forecast. Model selection used aggregate validation pinball loss; the 2011
test year remained untouched. The complete search space and selected
configuration are documented in the linked report.

The selected model reduced pinball loss by **43.18%** relative to the seasonal
naive baseline, an improvement confirmed by the monthly
Diebold–Mariano-style HAC test. It also reduced loss by **6.39%** relative to
the seasonal empirical baseline, but the same test did not establish this
difference statistically. Round one therefore provided promising, rather than
conclusive, evidence of improvement over the stronger probabilistic baseline.
Its marginal calibration improved, while its sharper intervals had worse
coverage.

The model was almost tied with seasonal empirical in 2009 but performed
substantially better in 2010. One plausible explanation is that later origins
have more historical cycles and training rows, making aggregated seasonal and
climatological features less noisy. This remains a hypothesis because calendar
year is also confounded with load drift and realized weather.

#### Round 2: data-driven feature selection

Round two reduced reliance on manually choosing the final feature set. The
candidate matrix extended the original 45 features to 70, then an
importance-based screening and verification pipeline reduced it to only **17
features**. Seven of the retained features were new, while many round-one
features were removed. The full selection procedure, selected features, and
search configuration are documented in the
[round-2 validation report](artifacts/catboost/round2/report.md).

To reduce computation, round two used the 12 monthly rolling-origin folds from
January through December 2010. This retained a complete seasonal cycle and,
following the round-one results, focused development on origins with more
historical support. Round one and both baselines were evaluated on exactly the
same hours, and a hyperparameter grid search selected the round-two model by
aggregate validation pinball loss. The 2011 test year remained untouched.

On the matched 2010 folds, round two reduced pinball loss by **6.82%** relative
to round one, **17.28%** relative to seasonal empirical, and **45.31%** relative
to seasonal naive. The monthly Diebold–Mariano-style HAC tests indicated
statistically lower average loss against all three references.

Round two also improved calibration and brought 90% interval coverage closer to
nominal, although its wider intervals sacrificed some sharpness and produced
more adjacent quantile crossings. The combined loss and calibration gains
support the value of the data-driven feature-selection step.

### Test evaluation

The validation pipeline clearly selected round two as the primary model: on
the matched 2010 folds it outperformed round one and both baselines. Its feature
set and hyperparameters were therefore frozen before accessing the 2011 test
period. For diagnostic completeness, the locked test evaluated all four
pre-specified methods—the two baselines and the selected models from rounds one
and two—but the additional comparisons were not used to revise the selection
decision.

Seasonal empirical baseline unexpectedly has the lowest realized test loss, despite
being 17.28% worse than the chosen model on the validation year. The chosen model is 11.87%
worse than empirical baseline on test, while the two CatBoost models are essentially
tied. This observed ranking reversal should not be overstated: the chosen model wins
6 of 12 test months against empirical, and the predeclared monthly
Diebold–Mariano-style HAC comparison does not establish a significant mean
loss difference (p=0.279). Nevertheless, seasonal empirical baseline is also much better
calibrated on this sample, so it is the strongest method by observed aggregate
loss and calibration for this particular 2011 sample. Complete monthly,
statistical, and calibration results are in the
[test report](artifacts/catboost/test/report.md).

The temporal context makes this a particularly unlucky test regime for the
learned models. Annual mean load increased in every year from 2006 through
2010, rising from 134.52 to 161.15. Neither validation period exposed a
comparable downward annual shift, so the evidence available for model selection
strongly supported a continuing higher-demand regime. Features that adapt to
recent load level and year-over-year growth were rationally useful on those
data, but became systematically misleading when the pattern reversed only in
the test year: annual mean load fell by 8.3% to 147.71 in 2011. The
resulting negative median biases confirm that both CatBoost models were
systematically biased high.

Seasonal empirical baseline could not extrapolate the preceding growth and consequently
under-forecast load during validation. That was a clear weakness before test,
but it became fortuitous when the 2011 level suddenly dropped: its conservative
forecasts landed much closer to the new regime. In other words, the baseline
did not anticipate the reversal; a validation-period error happened to cancel
an unexpected test-period shift. The selected model encountered the opposite
error alignment.

With only one test
year, load drift is confounded with realized weather, extreme events, and
ordinary finite-sample variation. The validation choice was still the correct
choice given the information available at selection time; choosing empirical
after observing 2011 would leak test information. The fundamental protection
against this kind of regime uncertainty is more representative data. More
generally, the only reliable way to reduce the risk of such failures is to
observe more independent years, repeated structural regimes, or additional
comparable load zones. Further tuning or a more complex model may not recover
evidence of a regime that has not yet occurred in the available data.

### Limitations and future work

This solution deliberately prioritizes a complete, leakage-safe pipeline over
an exhaustive modeling search. The main limitations and corresponding next
steps are:

- **Limited model diversity.** CatBoost gradient boosting was the only learned
  model family developed fully. It is a strong choice for nonlinear tabular
  data, but this leaves uncertainty about whether its errors are specific to
  boosted trees. Future work would compare it with 
  alternatives and test ensembles whose members make
  complementary errors.
- **One probabilistic formulation.** The learned models jointly predict 99
  quantiles by minimizing multi-quantile pinball loss. The outputs are not
  constrained to be monotone, which produced some quantile crossings, and no
  alternative distribution-construction method was evaluated. A useful next
  experiment would fit a strong point forecaster and estimate a 
  empirical distribution of its historical residuals, optionally conditioned
  on season, hour, horizon, or predicted load level. Adding the resulting
  residual quantiles to the point forecast would provide a distinct route to
  calibrated probabilistic predictions. Monotonic
  quantile rearrangement is a further candidate.
- **One seasonal test cycle.** The test contains only the twelve months
  of 2011. Although this covers every season, it provides only one realization
  of each season and one load regime. The validation-to-test reversal shows how
  strongly conclusions can depend on that single cycle. A more reliable
  assessment would require additional untouched years, comparable zones, or
  repeated deployment periods; no statistical adjustment can replace those
  independent observations.
- **Compute-bounded optimization.** Grid searches used compact, manually chosen
  parameter spaces, and feature selection used six screening plus six
  verification months with one fixed fast CatBoost configuration. These were
  appropriate compromises for a one-to-two-day CPU assignment, but are much simpler
  than a production research process. With more resources, I would expand the
  parameter search space; use a more
  efficient search strategy; evaluate feature stability across more origins
  and model configurations.

## References

Hong, T., Pinson, P., Fan, S., Zareipour, H., Troccoli, A., & Hyndman, R. J. (2016). [Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond](https://doi.org/10.1016/j.ijforecast.2016.02.001). *International Journal of Forecasting, 32*(3), 896–913.
