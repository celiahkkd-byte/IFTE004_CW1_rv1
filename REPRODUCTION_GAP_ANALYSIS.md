# Reproduction Gap Analysis
# Paper vs. Codebase: Christensen, Siggaard & Veliyev (JFEC 2023)

> This document is written for an AI reader to understand the current state of the reproduction
> codebase and what still needs to be done. Main operational claims have been verified against the
> paper PDF (`rv1.pdf`) and the live codebase on 2026-05-18.
> Re-audit note: the file was rechecked against the current code and CSV outputs on 2026-05-18.
> 2026-05-24 update: the active configuration and current strict final outputs now
> standardize NN training at **50 seeds** with top-10 ensembling. Older references in
> this historical planning document to 20 or 30 seeds describe earlier staged runs and
> should not be used as the current setting.
> The main reproduction gaps are still open; the operational hazards listed below should be treated
> as blockers before launching long reruns.
>
> **Scope of this document:** Defines the **core reproduction** (Tasks A0 through I)
> under a **constrained compute budget**. Completing all of Section 11 produces:
> - h=1 AND h=5 results matching the paper's Tables 2/3 (h=1) and Tables 4/5 (h=5) structure
> - Paper Figure 4-style Model Confidence Set inclusion rates
> - Paper-style filtered-historical-simulation VaR diagnostics
> - A per-stock Figure 6 ALE analysis (Apple, 3 features × 5 models)
> - Figure 5 RV-decile MSE analysis
> - A partial Figure 7-style ALE-based cross-sectional variable-importance analysis
>
> **constrained compute budget constraints accepted as rational trade-offs:**
> - NN seeds reduced from paper's 100 to **50** per (ticker, dataset, architecture).
>   Justification: paper's own Figure A.3 shows ensemble performance saturates by N≈10-20
>   seeds; the expected gap in NN^10 relative MSE from 100→50 seeds is expected to be
>   smaller than the earlier 100→30 approximation and dominated by other deviations
>   (data source, stock count).
> - h=22 (monthly horizon) is **NOT computed**. The paper shows largest ML gains at h=22;
>   our h=5 results therefore demonstrate the paper's "gains increase with horizon" finding
>   as a conservative lower bound rather than at full magnitude.
>
> **Omitted (acknowledged limitations, not blockers):**
> - Stock-level OptionMetrics IV → PARTIAL_MALL feature set used
> - h=22 multi-horizon results
> - 100-seed NN (using 50)
> - Section 12 enumerates all "must acknowledge" deviations in detail

---

## 0. Agent Handoff: Start Here

**Current status as of the 2026-05-18 recheck: the project is NOT ready for final
paper-reproduction training yet.** Do not launch Task A or Task B until the Phase 0
gates in Section 11 pass. The current repository state still has the following
blocking gaps:

| Blocker | Current state | Required before final training |
|---|---|---|
| Paper-core config | `config/paper_core_rolling.yaml` does not exist | Create it from the locked config in Section 11/13 |
| h=5 targets | `data/processed/forecasting_panel.csv` currently has h=1 targets only | Rebuild the panel so `target_rv_h5` and `target_log_rv_h5` exist |
| Rolling implementation | `_fit_one_asset_rolling()` still creates an extra internal holdout inside each rolling window | Complete Task A0 before any non-NN rolling run |
| Checkpoint reuse | Not yet smoke-tested under the locked config | Run the 2-seed NN reuse smoke test before long NN training |
| Final outputs | Existence of `outputs_rolling/`, `outputs_nn30_checkpointed/`, and `outputs_final/` final prediction files MUST be verified by the next agent (check `predictions/model_predictions.csv` existence + `by_model_manifest.csv` completeness); do not trust this row as a static assertion | Produce them through Tasks A/B/C after Phase 0 if not yet present |
| Post-processing scripts | Agent must verify which of Tasks D/E/F/G/H/I scripts exist under `scripts/`; implement only the missing ones | Task D before Task C; Tasks E/F/G/H/I can be implemented in parallel during Task A/B background runs |
| MCS dependency | `arch` Python package not installed | `pip install arch` before running Task I |

**Correct order for the next agent:**

1. Complete **Task A0**: fix paper-aligned rolling-window behavior in
   `src/rv1rep/forecasting.py`.
2. Create and lock `config/paper_core_rolling.yaml`; do not edit
   `config/default.yaml` for the final run.
3. Rebuild `data/processed/forecasting_panel.csv` with horizons `[1, 5]`.
4. Verify required h=1/h=5 target columns and LevHAR/HARQ feature construction.
5. Run the 2-seed NN checkpoint reuse smoke test. The second identical run must reuse
   checkpoints and finish quickly.
6. Launch **Task A**: non-NN rolling run into `outputs_rolling/` using
   `ml_refit_every=5`.
7. Launch **Task B smoke**: NN fixed-window run with 5 seeds into
   `outputs_nn30_checkpointed/`.
8. Validate the 5-seed NN aggregate; only then scale Task B to 30 seeds.
9. Implement/run **Task D** to extract NN^1 single-best-seed predictions.
10. After Task A completes, evaluate compute budget remaining. If sufficient
    (rough rule: remaining budget ≥ 5× the wall-clock time the original tree
    models took at `ml_refit_every=5`), optionally attempt the daily-refit
    upgrade (`ml_refit_every=1`) in a separate output directory.
    **Default decision is to SKIP** and document `ml_refit_every=5` as an
    approximation in Section 12.
11. Run **Task C** to merge/evaluate 14 rolling non-NN models + 4 NN^10 models +
    4 NN^1 models for h=1 and h=5.
12. Run **Tasks F/H/I** from `outputs_final/` (post-Task-C; Task I requires `arch` package); run **Tasks E/G** after the panel exists
    and preferably reuse Task B's NN2 checkpoints if available.
13. Update Section 12 with the actual completed configuration before using the results
    in the final report.

If any Phase 0 gate fails, stop and fix that gate first. Starting long training before
these checks pass risks silently producing non-paper-method outputs or invalidating
checkpoint reuse.

---

## 1. Paper Identity

**Full citation:** Kim Christensen, Mathias Siggaard, Bezirgen Veliyev,
"A Machine Learning Approach to Volatility Forecasting,"
*Journal of Financial Econometrics (JFEC)*, Vol. 21, No. 5, pp. 1680–1727, 2023.

**Core claim:** Off-the-shelf ML (regularized regression, regression trees, neural networks)
beats the HAR model family at out-of-sample realized variance forecasting, especially at
longer horizons and with a richer feature set. ALE-based variable importance shows
general agreement on dominant predictors but disagreement on their ranking.

---

## 2. Codebase Location

```
/Users/celiawong/Desktop/Course004 machine learning/004_individual/rv1_professional_reproduction_code/
```

Python: `~/.pyenv/versions/3.11.8/bin/python`

Key source files:
- `src/rv1rep/forecasting.py` — fixed and rolling estimation logic
- `src/rv1rep/models.py` — all model fitting (HAR family, regularized linear, trees, NN)
- `src/rv1rep/nn.py` — neural network ensemble
- `src/rv1rep/features.py` — feature construction and dataset assembly
- `src/rv1rep/evaluation.py` — MSE, relative MSE, DM test, pairwise matrix
- `src/rv1rep/split.py` — train/val/test splitting
- `src/rv1rep/explain.py` — ALE and permutation importance (implemented but NOT wired in)
- `config/default.yaml` — all hyperparameters and run settings

Current results:
- `outputs/` — 14-model results (no NN), fixed scheme
- `outputs_full_nn/` — 18-model results (with NN1–NN4), fixed scheme; **historical
  diagnostic output only** because these CSV files were generated before the LogHAR
  `log_vix` correction for the current no-IV panel and must be rerun before final
  reporting

---

## 3. Differences: Training Scheme

This is the most important structural difference.

### Paper's approach (p. 1692–1693)

The paper uses **different estimation schemes for different model families**:

| Model family | Paper's scheme |
|---|---|
| HAR, HAR-X, LogHAR, LevHAR, SHAR, HARQ | Rolling window (retrain each test day) |
| Bagging, Random Forest | Rolling window (no hyperparameter tuning needed) |
| Ridge, Lasso, EN, A-LA, P-LA, GB | Rolling window with validation for hyperparameter selection |
| NN1–NN4 | **Fixed window only** — "weights are only found once in the initial validation sample and not rolled forward" (paper's own words, p. 1692–1693) |

> Paper quote (p. 1692–1693): "At last, we employ a fixed window estimation for the NNs, where the
> weights are only found once in the initial validation sample and not rolled forward in the
> out-of-sample window. This implementation is strictly advantageous to the other competing
> models, which are allowed to adapt to new information."

### Codebase's approach

All models use **fixed scheme** (`scheme: fixed` in `config/default.yaml`).

The rolling implementation exists in `src/rv1rep/forecasting.py:_fit_one_asset_rolling()`
but is not the default. To activate: change `scheme: fixed` to `scheme: rolling` in config,
or pass `--scheme rolling` on the CLI.

**Impact:** For HAR family and tree models (BG, RF), the paper's rolling scheme allows
the model to adapt to new market regimes over the 847-day test window. The current
fixed-scheme results for these models are systematically less comparable to the paper.
For NNs, the fixed scheme is actually what the paper uses, so NN results are not affected
by this discrepancy.

### Data split proportions

| | Paper | Codebase |
|---|---|---|
| Training | 70% (~2,964 days) | 70% (fraction-based) |
| Validation | 10% (~424 days) | 10% |
| Test | 20% (~847 days) | 20% |

Proportions match. The paper also runs robustness checks with fixed_train_days=1000 and
fixed_train_days=2000 (both with val=200 days). These are supported in the config via
`fixed_train_days` / `fixed_val_days` but have not been run.

**Sample-balance caveat:** The AV panel is not a strict paper-style balanced panel. The
processed RV file has 25 tickers with different valid-day counts (roughly 4,025-4,228
days per ticker after the strict intraday grid filter), and the current h=1 test samples
are about 801-842 observations per ticker rather than a single common 847-day test block.
This is a data-source/filtering consequence and should be acknowledged with the 25-stock
AV limitation.

---

## 4. Differences: Data

| Dimension | Paper | Codebase |
|---|---|---|
| Price source | NYSE TAQ (cleaned transaction prices, Barndorff-Nielsen et al. 2009 filter) | Alpha Vantage 5-minute OHLCV bars |
| Number of stocks | 29 DJIA constituents | 25 (CVX, TRV, RTX, DOW excluded — AV lacks predecessor-stitching history) |
| Sample period | 2001-01-29 to 2017-12-31 (T=4,257 days) | 2001-01-02 to 2017-12-29 (effective start ~Feb 2001 after monthly lag burn-in) |
| Intraday frequency | 5-minute, n=78 returns/day | 5-minute, n=78 returns/day — **identical** |
| RV construction | sum of squared 5-min log returns | identical — **matches** |
| Semivariance RV+/RV− | yes | yes — **matches** |
| Realized quarticity RQ | yes | yes — **matches** |

**Note on TAQ vs AV:** TAQ data is cleaned for microstructure noise and outliers. Alpha
Vantage OHLCV bars are not. This introduces systematic differences in RV levels and
distribution, especially in high-volatility days. The codebase applies a strict grid filter
(only days with exactly 78 bars on the 09:30–15:55 grid are kept) but does not apply the
Barndorff-Nielsen et al. (2009) price filter.

---

## 5. Differences: Feature Set (MALL vs PARTIAL_MALL)

The paper's MALL dataset includes 12 variables total:

| # | Variable | Paper | Codebase |
|---|---|---|---|
| 1 | RVD | yes | yes |
| 2 | RVW | yes | yes |
| 3 | RVM | yes | yes |
| 4 | IV (model-free implied vol, OptionMetrics) | yes | **NO** — requires paid OptionMetrics licence |
| 5 | EA (earnings announcement indicator) | yes | yes |
| 6 | VIX | yes | yes |
| 7 | EPU | yes | yes |
| 8 | US3M (first-differenced) | yes | yes |
| 9 | HSI (squared daily log-return) | yes | yes |
| 10 | ADS | yes | yes |
| 11 | M1W (1-week momentum) | yes | yes |
| 12 | $VOL (log-diff dollar volume) | yes | yes (as `dvol`) |

The codebase uses the name `PARTIAL_MALL` (not `MALL`) throughout to explicitly
signal that IV is omitted. Internal dataset column values and filenames use `PARTIAL_MALL`;
do not rename them as it would break the pipeline.

**Impact:** IV is described in the paper as one of the top-ranked predictors (Figure 7).
Its omission is the most material feature-set gap. All public variables are present.

**Implementation note for LogHAR + PARTIAL_MALL:** The paper states that VIX and IV
are log-transformed in LogHAR. In the current public-data panel, **IV is absent**:
`forecasting_panel.csv` contains `vix` and `log_vix`, but not `iv` or `log_iv`.
Therefore the current LogHAR PARTIAL_MALL feature set actually uses `log_vix` and
omits IV entirely. If a real user-supplied `data/external/firm_iv.csv` is later provided,
`feature_columns_for_model()` will use `log_iv` for LogHAR, matching the paper's IV
transform rule. Other model families continue to use level `vix` and would use level
`iv` only if real IV is supplied. Existing CSV outputs produced before the `log_vix`
fix must be rerun before claiming they reflect the corrected LogHAR transform.

---

## 6. Differences: Model Hyperparameters

### Neural Networks

| Hyperparameter | Paper | Codebase |
|---|---|---|
| Architecture NN1–NN4 | [2], [4,2], [8,4,2], [16,8,4,2] | identical |
| Activation | L-ReLU, c=0.01 | LeakyReLU(alpha=0.01) — **matches** |
| Optimizer | Adam, lr=0.001 | Adam, lr=0.001 — **matches** |
| Dropout rate | 0.8 | 0.8 — **matches** |
| Early stopping | patience=100 | patience=100 — **matches** |
| Random seeds | **100** | **20** (config comment: "Paper: 100; reduce for coursework speed") |
| Ensemble | top-10 out of 100 | top-10 out of 20 |
| Reports | NN1_1 (single) and NN10_1 (ensemble) separately | ensemble only |

The paper reports both the single best seed (NN^1) and the top-10 ensemble (NN^10) for
each architecture, giving 8 NN columns in Tables 2/3. The codebase reports only the
ensemble, giving 4 NN columns.

### Regularized Linear Models

| | Paper | Codebase |
|---|---|---|
| Alpha grid size | 1,000 points (log-spaced) | **80 points** (config comment: "Paper: 1000; reduce for coursework speed") |
| Alpha range | [1e-5, 1e2] | [1e-5, 1e2] — **matches** |
| EN l1_ratio / `a` grid | 10 validation-grid points on [0, 1] | 7 grid points: [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0] |

80 log-spaced alpha points is generally sufficient for practical purposes but is a known
reduction. For a strict paper-method run, use 1,000 alpha points and 10 EN mixing-grid
points; otherwise report the coarser grid as a computational approximation.

### Tree-Based Models

| | Paper | Codebase |
|---|---|---|
| Number of trees (BG, RF) | 500 | 500 — **matches** |
| Min leaf size | 5 | 5 — **matches** |
| RF max_features | 1/3 of features | 1/3 — **matches** |
| GB depth search | unspecified | [1, 2] |
| GB n_estimators search | unspecified | [50, 100, ..., 500] |
| GB learning rate search | unspecified | [0.01, 0.1] |

---

## 7. What Has Been Completed

### Completed and verified

| Item | Output location | Status |
|---|---|---|
| RV construction (78 5-min returns, grid filter) | `data/processed/daily_realized_measures.csv` | Done, 104,563 rows, 25 tickers |
| External data download (VIX, EPU, ADS, HSI, US3M) | `data/external/` | Done |
| Earnings announcements (EA) | `data/external/earnings_announcements.csv` | Done, ~1,699 events |
| Feature panel construction | `data/processed/forecasting_panel.csv` | Done |
| h=1 forecasts, 14 non-NN models, fixed scheme | `outputs/predictions/model_predictions.csv` | Done, 583,008 rows |
| h=1 forecasts, 18 models (incl. NN1–NN4), fixed scheme | `outputs_full_nn/predictions/model_predictions.csv` | Done, 749,584 rows; historical/pre-LogHAR-`log_vix`-fix output only |
| Cross-sectional relative MSE summary (Table 2/3 structure) | `outputs/tables/forecast_summary_cross_section.csv` | Done |
| Pairwise relative MSE matrix | `outputs/tables/pairwise_relative_mse_matrix.csv` | Done, 28 rows × 31 cols (long format, 2 datasets) |
| DM test results | `outputs/tables/diebold_mariano_tests.csv` | Done |
| Per-asset MSE metrics | `outputs/tables/forecast_metrics_by_asset.csv` | Done |
| VaR diagnostic (normal-parametric, Kupiec + Christoffersen) | `outputs/tables/var_backtest_summary.csv` | Done, 700 rows; **not paper-style filtered historical simulation** |
| Relative MSE bar figures | `outputs/figures/relative_mse_MHAR.png`, `relative_mse_PARTIAL_MALL.png` | Done |

---

## 8. What Is Missing or Not Completed

### 8.1 Rolling scheme (affects HAR family and ML except NN)

The paper's primary results for all non-NN models use rolling window estimation.
The codebase has `scheme: fixed`. See Section 11 Task A for the exact command to run.
Do not run `04_run_forecasts.py` directly as it would write to the default `outputs/` directory.

**Important caveat on rolling fidelity:** The rolling implementation is not identical to
the paper yet. Two implementation details must be fixed or explicitly treated as
approximations:

1. In `src/rv1rep/forecasting.py:_fit_one_asset_rolling()`, the current rolling code forms
   a train+validation window and then calls `chronological_split()` with 70%/10% fractions.
   This unintentionally leaves the final 20% of each rolling window unused. The paper does
   not introduce this third holdout block inside the rolling window. A paper-aligned rolling
   implementation should use the whole pre-forecast window: the older part as training and
   the most recent validation block only when hyperparameter tuning is needed.
2. The config parameter `ml_refit_every: 20` causes Bagging, RandomForest, and
   GradientBoosting to retrain only once every 20 test days. The paper's non-NN rolling
   design updates the non-NN forecasts day by day. For a strict paper-method run, tree
   models should be refit at every test date (`ml_refit_every: 1` or equivalent). If the
   20-day approximation is retained for budget reasons, it must not be described as a
   full paper-aligned rolling reproduction.

The fitting rule should also follow the paper's use of validation. Non-regularized HAR
models, Bagging, and RandomForest do not tune hyperparameters and should fit on the
merged train+validation window. Ridge, Lasso, ElasticNet, AdaptiveLasso, PostLasso,
and GradientBoosting require validation for tuning; for these models the rolling window
should keep a training block and a validation block, without discarding an additional
test-like slice before the forecast date.

### 8.2 Multi-horizon forecasts — Paper Section 4, Tables 4/5

The current config has `horizons: [1]`. The paper's Section 4 shows the
**largest ML gains** appear at weekly (h=5) and monthly (h=22) horizons, with
40%+ MSE reductions over HAR for some models in MALL.

**Compute-budget decision (h=5 included, h=22 omitted):**
- **h=5 IS in the core plan** (Section 11 Tasks A/B include h=5). Task A non-NN
  rolling and Task B NN fixed both run for h=1 AND h=5 in this revised plan.
- **h=22 is NOT in the core plan.** It would require an additional ~2 days of NN
  training, which is infeasible within the constrained compute budget. Section 12 explicitly
  documents this as an acknowledged limitation. The paper's "gains increase with
  horizon" finding is demonstrated at h=5 as a conservative lower bound; h=22
  would show the largest effect but the directional finding is established at h=5.

Required config change (`config/paper_core_rolling.yaml`):
```yaml
feature_engineering:
  horizons: [1, 5]

experiments:
  horizons: [1, 5]
```
Then re-run `scripts/03_build_features.py` to materialize `target_rv_h5` and
`target_log_rv_h5` in the panel before launching Tasks A and B. `horizon_target_mode:
future_average` is already set correctly (matches paper's definition of multi-step
target as mean of h steps ahead).

**Operational hazard:** `feature_engineering.horizons` controls which target columns are
materialized in `data/processed/forecasting_panel.csv`, while `experiments.horizons`
controls which horizons are forecast. If only `experiments.horizons` is changed, the
forecast step will look for missing columns such as `target_rv_h5` and fail. The current
panel contains only `target_rv_h1` and `target_log_rv_h1`.

**NN horizon support:** `scripts/04_run_nn_checkpoints.py` accepts `--horizons` (defaults
to `experiments.horizons` from config) and segregates per-seed checkpoints by horizon
under `nn_seed_predictions/<dataset>/h{horizon}/<model>/<ticker>/seed_*.csv`. The script
will refuse to start if any requested horizon's `target_rv_h{h}` column is missing from
`forecasting_panel.csv`, so the panel must be rebuilt via `scripts/03_build_features.py`
first when adding new horizons. Pre-existing flat-layout h=1 checkpoints under the legacy
path `nn_seed_predictions/<dataset>/<model>/<ticker>/seed_*.csv` are auto-migrated to the
new horizon-aware layout on first encounter, so existing in-flight runs are not invalidated.

### 8.3 Model Confidence Set (MCS) — Paper Figure 4

The paper constructs an MCS at 90% confidence (Hansen, Lunde, Nason 2011) showing
the percentage of stocks for which each model was retained in the final set. This is
not yet implemented in the codebase; DM tests are present but MCS requires
elimination-tournament + stationary block bootstrap logic.

**Planned remediation:** Task I in Section 11 uses `arch.bootstrap.MCS`
(Hansen-Lunde-Nason 2011 reference implementation maintained by Kevin Sheppard,
co-author of Patton & Sheppard 2015) to compute per-ticker 90% MCS over all 22
final models for both datasets and both horizons. Task I is post-processing only
(reads `outputs_final/predictions/model_predictions.csv`); no retraining required.
Estimated cost: 3-6 hours including dependency install, script writing, ~5,000 bootstrap reps
× 25 tickers × 4 (dataset, horizon) combinations, and the Figure 4 plot.

### 8.4 ALE plots and Variable Importance — Paper Figures 6/7

`src/rv1rep/explain.py` contains `normalized_permutation_importance()` and
`accumulated_local_effect()` implementations. However:
- No script calls these functions
- `scripts/04_run_forecasts.py` does not serialize fitted model objects (no pickle/joblib)
- Post-hoc ALE therefore cannot be run without refitting

The paper's Figure 6 (ALE for RVD, RVW, IV, M1W across HAR-X, LogHAR, EN, RF, NN^10_2)
and Figure 7 (cross-sectional variable importance ranking) are both absent.

**Planned remediation:**
- **Figure 6 (ALE):** Task E in Section 11 will retrain the five reference models once
  for **Apple only**, matching the paper's choice not to average the ALE curve across
  assets. It will compute ALE for **three** reference features (RVD, RVW, M1W) over the
  standardized interval [-1, 1]. The paper's fourth ALE row uses IV (OptionMetrics
  stock-level implied volatility), which is unavailable in this reproduction. VIX is a
  market-level index and is NOT a valid substitute for stock-level IV, so the IV row is
  omitted rather than replaced.
- **Figure 7 (cross-sectional variable importance):** Task G in Section 11 must use the
  paper's ALE-based variable-importance definition, not permutation importance. For each
  ticker and model, compute ALE for each available PARTIAL_MALL predictor, compute
  `I(feature)` as the sample standard deviation of the centered ALE values, normalize by
  the sum across features, and then average those normalized VI values across the 25
  tickers. Because stock-level IV is unavailable, IV is omitted; VIX remains only its own
  market-volatility feature and is not used as an IV substitute.

### 8.5 Figure 5 — MSE by realized variance decile

The paper splits the test set into 10 deciles of observed RV and computes MSE within
each decile for HAR-X, LogHAR, EN, RF, NN^10_2. This analysis does not exist in any
script.

**Planned remediation:** Task F in Section 11 will read `outputs_final/predictions/model_predictions.csv`
after Task C completes, split the PARTIAL_MALL test observations into realized-variance
deciles, and reproduce the paper's selected comparison: HAR-X, LogHAR, EN,
RandomForest, and NN^10_2 in the low, medium, and high deciles shown by the paper.
A complete all-model/all-decile CSV may also be written for auditability, but the primary
figure should follow the paper's selected model and decile set.

### 8.6 Single-seed NN results (NN^1) vs ensemble (NN^10)

The paper reports both NN^1_k (best single seed) and NN^10_k (top-10 ensemble) for
each architecture k=1,2,3,4. The old `outputs_full_nn/` results report only the ensemble
and individual seed predictions cannot be recovered from it. However, after Task B
completes, `outputs_nn30_checkpointed/nn_seed_predictions/` will contain per-seed checkpoint files,
from which the single-best-seed result can be extracted if needed.

**Planned remediation:** Task D in Section 11 will read those per-seed checkpoints
after Task B finishes, select for each `(dataset, model, ticker)` the seed with the
lowest validation MSE, apply the same positivity/insanity filters, and write four
additional model columns (`NN1_1`, `NN2_1`, `NN3_1`, `NN4_1`) merged into the final
prediction file before evaluation.

### 8.7 Robustness checks — Appendix A.1 (train_days=1000, 2000)

Config supports `fixed_train_days` and `fixed_val_days` but these have not been run.

### 8.8 Paper-style VaR application

The current `src/rv1rep/var_backtest.py` builds a simple normal-parametric VaR forecast
from `z_alpha * sqrt(forecast_rv)`. This is useful as a diagnostic, but it is **not** the
paper's VaR construction. The paper uses filtered historical simulation: standardized
returns from the in-sample period are used to estimate the residual quantile, then the
one-day-ahead volatility forecast scales that empirical quantile.

**Planned remediation:** Task H in Section 11 will compute paper-style filtered
historical simulation VaR from `outputs_final/predictions/model_predictions.csv`, using
the same train/validation information set that generated each forecast where available.
If only test prediction CSVs are available, Task H must rebuild the required in-sample
standardized residuals from the panel and fitted model definitions rather than silently
falling back to the normal-parametric version.

### 8.9 Operational hazards in the planned rerun

These are code-level risks discovered in the current implementation:

- `scripts/04_run_forecasts_checkpoints.py` catches per-model exceptions, writes them to
  `by_model_manifest.csv`, and still combines any existing successful by-model files.
  A non-empty `model_predictions.csv` does not prove the full 14-model rolling run
  completed. Always inspect the manifest for `failed`, `empty`, or `missing` rows.
- Task B relies on the NN config staying `scheme: fixed`. Use
  `config/paper_core_rolling.yaml` for Task B as well, but keep its
  `estimation.scheme: fixed`; Task A receives rolling behavior via the CLI
  `--scheme rolling` override. The NN runner uses a fixed train/validation/test split,
  but it writes the `scheme` column from config, so a changed config would mislabel
  fixed NN predictions as rolling.
- Task C's merge code currently checks duplicate keys only. It should also verify the
  expected model set, datasets, horizons, and schemes before evaluation.

---

## 9. NN Performance: Why Results Differ From Paper

The paper (Table 2, MHAR) shows NN^10_k relative MSE vs HAR of approximately:
- NN^10_1: 0.969
- NN^10_2: 0.958
- NN^10_3: 0.954
- NN^10_4: 0.990

The codebase (outputs_full_nn, MHAR) shows:
- NN1: 1.039
- NN2: 1.249
- NN3: 1.117
- NN4: 1.103

All codebase NN results are **above 1.0** (worse than HAR). The paper's NN results are
**below 1.0** (better than HAR). This is a directional reversal, not just a magnitude
difference. The most likely causes in order of impact:

1. **20 seeds vs. 100 seeds:** Fewer seeds means lower-quality ensemble. With only 20 seeds
   the top-10 selected models may include more mediocre fits, increasing variance of the
   ensemble prediction.
2. **TAQ vs. AV data:** Different noise characteristics in the input features affect NN
   training stability more than linear models. AV OHLCV bars are noisier than TAQ
   cleaned transaction prices.
3. **25 stocks vs. 29 stocks and shorter effective sample:** Less data makes NN harder to
   regularize effectively relative to simpler linear models.
4. **Alpha grid 80 vs. 1000 points:** Coarser grid may leave regularized linear competitors
   slightly suboptimally tuned, but this effect is minor.

---

## 10. Current Result Summary (Historical CSVs, Pre-LogHAR-`log_vix` Fix)

The rankings below were verified from `outputs_full_nn/` CSV files, but these files were
generated before the LogHAR `log_vix` correction for the current no-IV panel. Treat this
section as a diagnostic snapshot of the old completed run, not as final paper-comparable
evidence. No `iv` or `log_iv` variable exists in the current panel.

### PARTIAL_MALL, h=1, fixed scheme (outputs_full_nn, 18 models)

Sorted by avg_rel_mse_vs_har (lower = better than HAR):

| Rank | Model | Rel MSE vs HAR |
|---|---|---|
| 1 | LogHAR | 0.904 |
| 2 | RandomForest | 0.929 |
| 3 | ElasticNet | 0.971 |
| 4 | Lasso | 0.973 |
| 5 | Ridge | 0.991 |
| 6 | AdaptiveLasso | 0.992 |
| 7 | Bagging | 0.994 |
| 8 | SHAR | 0.999 |
| 9 | HAR | 1.000 |
| 10 | HARX | 1.000 |
| 11 | PostLasso | 1.006 |
| 12 | HARQ | 1.012 |
| 13 | NN1 | 1.047 |
| 14 | LevHAR | 1.064 |
| 15 | NN2 | 1.091 |
| 16 | NN3 | 1.091 |
| 17 | NN4 | 1.111 |
| 18 | GradientBoosting | 1.280 |

### MHAR, h=1, fixed scheme (outputs_full_nn, 18 models)

| Rank | Model | Rel MSE vs HAR |
|---|---|---|
| 1 | LogHAR | 0.932 |
| 2 | RandomForest | 0.986 |
| 3 | ElasticNet | 0.999 |
| 4 | Ridge | 0.999 |
| 5 | SHAR | 1.000 |
| 6 | PostLasso | 1.000 |
| 7 | HAR | 1.000 |
| 8 | HARX | 1.000 |
| 9 | Lasso | 1.003 |
| 10 | AdaptiveLasso | 1.019 |
| 11 | HARQ | 1.031 |
| 12 | NN1 | 1.039 |
| 13 | Bagging | 1.049 |
| 14 | NN4 | 1.103 |
| 15 | LevHAR | 1.110 |
| 16 | NN3 | 1.117 |
| 17 | NN2 | 1.249 |
| 18 | GradientBoosting | 1.285 |

**Old-run findings consistent with paper:** LogHAR dominates the HAR family; RandomForest
is the strongest tree-based model. These directions match the paper, but the exact
LogHAR numbers must be regenerated after the transform fix before final citation.

**Old-run findings inconsistent with paper:** GradientBoosting is worst in the current
results, whereas in the paper GB is competitive in MALL. NN results are in the opposite
direction (above 1.0 here vs below 1.0 in paper). Both require acknowledgment as
deviations and should be reassessed after the final rerun.

---

## 11. Execution Plan (Compute-Budget-Constrained, Phase-Gated)

The following decisions have been made and confirmed under a constrained compute budget.
Execute exactly as specified.

### Naming convention

| Paper notation | Code/CSV name | Used in |
|---|---|---|
| NN^10_k (top-10 ensemble) | `NN1`, `NN2`, `NN3`, `NN4` | Task B output, predictions CSV |
| NN^1_k (single best seed) | `NN1_1`, `NN2_1`, `NN3_1`, `NN4_1` | Task D output |
| NN^10_2 in ALE/VI analysis | `NN10_2` | Task E/G output tables |

### Decision summary

| Model group | Scheme | Horizons | Seeds | Rationale |
|---|---|---|---|---|
| HAR family (HAR, HARX, LogHAR, LevHAR, SHAR, HARQ) | **rolling** | **h=1, h=5** | n/a | Paper p.1692 |
| Regularized linear (Ridge, Lasso, EN, A-LA, P-LA) | **rolling** | **h=1, h=5** | n/a | Paper p.1692 |
| Tree-based (Bagging, RF, GB) | **rolling, refit every 5 test days at launch** | **h=1, h=5** | n/a | Paper uses daily refit. `ml_refit_every=5` is the locked compute-budget launch setting; optional upgrade to daily refit must use a separate output directory. |
| NN1–NN4 (ensemble NN^10_k) | **fixed** | **h=1, h=5** | **30** | Paper p.1692–1693 uses fixed for NN. Seeds 100→30: paper's Figure A.3 shows ensemble saturation by N≈10-20; expected NN^10 quality loss is 0.5%-2%. |
| NN1_1–NN4_1 (single-best seed NN^1_k) | **fixed** | h=1, h=5 | extracted from 30 seeds | Adds the 4 single-seed columns the paper reports in Tables 2/3. Task D is post-processing (~1 hour, zero retraining). |

**h=22 is explicitly NOT in this plan.** It is the paper's strongest result (largest ML
gains over HAR) but would require an additional ~2 days of NN training. Within the
compute budget, h=5 is sufficient to demonstrate "gains increase with horizon."

**NN rolling is not feasible** (paper estimates 2,525 hours for h=1 alone). The paper
itself states fixed window for NN is outside rolling budget. This is not a deviation —
it matches the paper exactly.

### Execution sequence (gated by completion, not by calendar)

The plan is **not on a fixed timeline**. Every step is gated by the actual completion
of its predecessor. Do not advance to the next phase until the prior phase's exit
criteria are met — wall-clock time may vary significantly with hardware.

| Phase | Triggered by | Exit criterion |
|---|---|---|
| **Phase 0** — Fix + lock + validate | Plan approved | All pre-launch gates 0-5 pass (see below). No long training before this. |
| **Phase 1** — Launch training | Phase 0 complete | Task A and Task B 5-seed smoke test running in screen with non-zero output |
| **Phase 2** — NN smoke validation | Task B smoke 5 seeds finish | 5-seed NN aggregate has expected models + tickers + horizons |
| **Phase 3** — Scale NN | Phase 2 passed | Task B re-launched at `--seed-count 30`, prior 5 seeds reused |
| **Phase 4** — Task A complete | Task A manifest shows all 14 models completed/reused, no failed/empty | Optional tree-model upgrade decision point (see Task A subsection) |
| **Phase 5** — Task B complete | NN aggregate has all 4 architectures × 25 tickers × 2 datasets × 2 horizons × 30 seeds present | Ready for Task D |
| **Phase 6** — Merge + post-processing | Phases 4 + 5 complete | Run Task D before Task C; then run C, F, H, I. Tasks E/G may run after panel exists. |
| **Phase 7** — Reporting | Phase 6 complete | Section 12 limitations updated to reflect what actually completed |

**Rough wall-clock estimates** (informational only; do NOT use as a deadline):

- Phase 0 lock-down: 2-3 hours human work
- Task A non-NN rolling (h=1+h=5, `ml_refit_every=5`): 8-20 hours background
- Task B NN smoke (5 seeds): 6-10 hours background
- Task B full (30 seeds, incremental): additional 30-50 hours background after smoke
- Tasks D / C / E / F / G / H / I: 8-14 hours human + script work combined (Tasks E/F/G/H/I are parallelizable post-Task-C)
- Optional tree-model upgrade (`ml_refit_every: 1`): 15-35 hours additional background

These add up to roughly 4-6 calendar days on a single Mac with no parallel hardware,
but the actual elapsed time depends on hardware, interruption frequency, and whether
the optional tree-model upgrade is attempted. **An agent reading this document must check the
exit criterion of the current phase before proceeding, not the elapsed time.**

### Phase 0 pre-launch gates (must complete before any long training starts)

These parameters affect checkpoint reuse. Once any of `ml_refit_every`,
`alpha_grid_size`, `seeds`, `horizons`, or `forecasting_panel.csv` content changes,
existing checkpoints become unsafe to reuse silently. **Run gates 0-5 before launching
Task A or Task B smoke**:

#### Gate 0 — Complete Task A0 rolling-window implementation

Patch `src/rv1rep/forecasting.py:_fit_one_asset_rolling()` so that it uses the
paper-aligned rolling-window contract in Task A0 below. This gate is mandatory before
Task A because the current rolling code creates an extra internal holdout and discards
part of each rolling train/validation window.

Exit criterion:
- Tuned models use `train_dates = window_dates[:train_n]` and
  `val_dates = window_dates[train_n:]`.
- Non-tuned rolling models fit on the full pre-forecast `window_dates`.
- No extra 70/10/20 split is created inside the rolling window.

#### Gate 1 — Create paper-core config

Write `config/paper_core_rolling.yaml` with the values shown earlier in this section
(see "`config/paper_core_rolling.yaml` — locked pre-launch configuration"). Verify it differs
from `config/default.yaml` only in the documented fields.

#### Gate 2 — Rebuild forecasting panel with h=5 targets and verify columns

Step 2a — rebuild:

```bash
~/.pyenv/versions/3.11.8/bin/python scripts/03_build_features.py \
  --config config/paper_core_rolling.yaml
```

Step 2b — verify columns (this is part of Gate 2's exit criterion; do not split into
a separate gate, both must pass before advancing):

```bash
~/.pyenv/versions/3.11.8/bin/python -c "
import pandas as pd
panel = pd.read_csv('data/processed/forecasting_panel.csv', nrows=1)
required = {'target_rv_h1', 'target_log_rv_h1', 'target_rv_h5', 'target_log_rv_h5', 'log_vix'}
missing = required - set(panel.columns)
if missing:
    raise SystemExit(f'Panel missing required columns: {missing}')
print('Panel column check OK')
"
```

**Exit criterion for Gate 2**: rebuild command (Step 2a) exits with code 0 AND the
column-verification script (Step 2b) prints "Panel column check OK" without
raising. If either fails, do not advance to Gate 3; fix the panel-build
configuration first.

#### Gate 3 — Verify LevHAR / HARQ feature construction matches paper

```bash
~/.pyenv/versions/3.11.8/bin/python -c "
import numpy as np
import pandas as pd
panel = pd.read_csv('data/processed/forecasting_panel.csv',
                    usecols=['cc_logret', 'rd', 'rw', 'rm', 'rq', 'rvd', 'sqrt_rq_x_rvd']).dropna()
# HARQ: sqrt_rq_x_rvd must equal sqrt(rq) * rvd (paper eq. 10)
expected_harq = np.sqrt(np.maximum(panel['rq'], 1e-12)) * panel['rvd']
assert np.allclose(panel['sqrt_rq_x_rvd'], expected_harq, atol=1e-6), 'HARQ interaction broken'
# LevHAR: rd must equal min(0, cc_logret) (paper eqs. 7-8)
expected_rd = np.minimum(0.0, panel['cc_logret'])
assert np.allclose(panel['rd'], expected_rd), 'LevHAR rd is NOT min(0, cc_logret)'
# rw and rm should also be non-positive
assert (panel['rw'] <= 1e-9).all(), 'LevHAR rw has positive values; check aggregation rule'
assert (panel['rm'] <= 1e-9).all(), 'LevHAR rm has positive values; check aggregation rule'
print('LevHAR / HARQ feature construction verified against paper eqs. 7, 8, 10')
"
```

#### Gate 4 — Checkpoint reuse smoke test (validates the entire compute-budget premise)

Run NN with 2 seeds, then re-run the same command. The second run must skip retraining
and complete in under 2 minutes. **If it retrains, the compute-budget plan is at risk** and the
checkpoint logic must be debugged before continuing.

```bash
~/.pyenv/versions/3.11.8/bin/python scripts/04_run_nn_checkpoints.py \
  --config config/paper_core_rolling.yaml \
  --output-dir outputs_smoke_test \
  --datasets PARTIAL_MALL --horizons 1 --models NN1 \
  --seed-count 2 --ensemble-top 2 \
  --base-predictions "" \
  --allow-existing-output-dir

# Re-run the EXACT same command, time it
time ~/.pyenv/versions/3.11.8/bin/python scripts/04_run_nn_checkpoints.py \
  --config config/paper_core_rolling.yaml \
  --output-dir outputs_smoke_test \
  --datasets PARTIAL_MALL --horizons 1 --models NN1 \
  --seed-count 2 --ensemble-top 2 \
  --base-predictions "" \
  --allow-existing-output-dir
# Should complete < 2 minutes with "Reusing seed checkpoint" in logs
```

### Phase 1 launch order

Only after Gates 0-5 pass:

1. Launch Task A (non-NN rolling, h=1+h=5) in `screen` as shown in the Task A subsection.
   Expected wall time at `ml_refit_every: 5`: 8-20 hours background.
2. Launch Task B NN smoke test (5 seeds) in a separate `screen`.
3. Do not scale Task B to 30 seeds until the 5-seed aggregate validates in Phase 2.

Start Task B with `--seed-count 5` to validate the pipeline end-to-end. Once 5 seeds
complete successfully, re-launch with `--seed-count 30` (the 5 existing seeds are
reused automatically). Expected wall time for 30 seeds at h=1+h=5: ~2 days background.

```bash
# Phase 1: smoke test with 5 seeds (rough estimate ~6-10 hours background)
cd "/Users/celiawong/Desktop/Course004 machine learning/004_individual/rv1_professional_reproduction_code"
mkdir -p outputs_nn30_checkpointed/logs
screen -dmS rv1_nn_smoke bash -lc '
~/.pyenv/versions/3.11.8/bin/python scripts/04_run_nn_checkpoints.py \
  --config config/paper_core_rolling.yaml \
  --output-dir outputs_nn30_checkpointed \
  --datasets MHAR PARTIAL_MALL --horizons 1 5 --models NN1 NN2 NN3 NN4 \
  --seed-count 5 --ensemble-top 5 \
  --base-predictions "" \
  --allow-existing-output-dir \
  > outputs_nn30_checkpointed/logs/04_nn_smoke.log 2>&1
'

# Phase 3, only after Phase 2 smoke validation passes:
# scale to 30 seeds and restore paper's top-10 ensemble; the 5 existing seeds are reused automatically.
# Same command, change --seed-count 5 to --seed-count 30 and --ensemble-top 5 to --ensemble-top 10.
```

### Forbidden actions after launch (will break checkpoint reuse)

After Phase 1 launch, do NOT:
- Change `ml_refit_every`, `alpha_grid_size`, `alpha_min`, `alpha_max`, or
  `elastic_l1_ratios` in `config/paper_core_rolling.yaml` in place.
- Rebuild `data/processed/forecasting_panel.csv` (would invalidate all checkpoints).
- Move or rename `outputs_rolling/` or `outputs_nn30_checkpointed/`.

What CAN be changed mid-run safely (increases, not changes, of work):
- `seeds` increment (30 → 50): re-run Task B with `--seed-count 50`, prior 30 are reused.
- Add more NN architectures: existing NN1-4 outputs untouched.
- Apply optional tree-model upgrade via new output directory (see Task A subsection below).

### Alpha boundary stability check (run after Task A completes)

To justify `alpha_grid_size: 80` instead of paper's 1000, verify that the chosen
optimal α* is not boundary-bound. Run this after Task A produces predictions and the
underlying CV log captures the selected α per (model, ticker, day, dataset, horizon).

```python
# scripts/verify_alpha_boundary.py (write while training runs; consume after Task A completes)
# Inspect alpha_chosen logs from regularized linear models
# Report per model: percentile distribution of log10(alpha_chosen)
# Pass criterion: <1% of selections at log α ≤ log(alpha_min)+0.5 OR log α ≥ log(alpha_max)-0.5
# Models to check: Ridge, Lasso, ElasticNet, AdaptiveLasso, PostLasso

for model in ['Ridge', 'Lasso', 'ElasticNet', 'AdaptiveLasso', 'PostLasso']:
    log_alphas = collect_chosen_alphas_log10(model)
    boundary_low_pct = (log_alphas <= np.log10(1e-5) + 0.5).mean()
    boundary_high_pct = (log_alphas >= np.log10(1e2) - 0.5).mean()
    print(f'{model}: lower boundary={boundary_low_pct:.2%}, upper={boundary_high_pct:.2%}')
    assert boundary_low_pct < 0.01, f'{model} alpha_min may be too high (1e-5)'
    assert boundary_high_pct < 0.01, f'{model} alpha_max may be too low (1e-2)'
```

If AdaptiveLasso (most likely candidate) trips the upper boundary, extend
`alpha_max: 1.0e2` to `1.0e3` and rerun **only that model** with `--force --models AdaptiveLasso`.

This check produces stronger evidence than expanding the grid: a converged 80-point
log-spaced grid is statistically equivalent to a converged 1000-point grid for ranking
purposes.

### Task dependency graph

```
Task A0 (paper-aligned rolling implementation/config)
        │
        └──► Task A  (rolling non-NN, h=1+h=5)  ─┐
                                                  ├──► Task C (merge + evaluate) ──► outputs_final/ ──► Task F (RV-decile MSE)
Task B  (NN 30 seeds, h=1+h=5)                  ──┤                                      │
                                                  └──► Task D (NN^1 extraction)           └──► Task H (FHS VaR)
                                                          │
                                                          └──► must complete BEFORE Task C merge

Task E  (Figure 6 ALE) — independent, can run any time after panel exists
Task G  (partial Figure 7 ALE-based VI) — independent, can run any time after panel exists
                                          (optionally reuses Task B's NN2 seeds if available)
```

**Dependency rules:**
- Task A0 must be completed before Task A; otherwise rolling results are not paper-method aligned.
- Task A and Task B write to separate directories and can run in parallel.
- Task D must run after Task B (it reads `outputs_nn30_checkpointed/nn_seed_predictions/`).
- Task D must run before Task C (so Task C merges 14 rolling + 4 NN^10 + 4 NN^1 = 22 models per horizon = **44 model-horizon entries total for h=1+h=5**).
- Task E is independent and may run any time after `forecasting_panel.csv` exists.
- Task F depends on Task C output (`outputs_final/predictions/model_predictions.csv`).
- Task H depends on Task C output and, if needed, refitted in-sample residuals from the panel.
- Task G is independent of A/B/C/D/E/F/H; it may run any time after the panel exists.
  If Task B has already produced NN2 per-seed checkpoints under
  `outputs_nn30_checkpointed/nn_seed_predictions/PARTIAL_MALL/h1/NN2/`, Task G reuses
  them for the NN^10_2 component; otherwise it trains 30 fresh seeds.

### Task A0 — Required rolling-window alignment before Task A

Do not launch the final rolling run until `_fit_one_asset_rolling()` is aligned with the
paper's rolling-window usage.

Required behavior:

| Model group | Rolling window use |
|---|---|
| HAR, HARX, LogHAR, LevHAR, SHAR, HARQ | Fit on the full pre-forecast train+validation window; no validation holdout is needed. |
| Bagging, RandomForest | Fit on the full pre-forecast train+validation window; no validation holdout is needed. |
| Ridge, Lasso, ElasticNet, AdaptiveLasso, PostLasso, GradientBoosting | Use the older training block for candidate fitting and the most recent validation block for tuning; do not create or discard an extra third holdout block inside the rolling window. |

Implementation contract:
- Preserve the initial split lengths from `chronological_split()`: `train_n = len(split.train_dates)` and `val_n = len(split.val_dates)`.
- For a forecast at position `pos`, use exactly the `train_n + val_n` observations before `pos`.
- For tuned models, set `train_dates = window_dates[:train_n]` and `val_dates = window_dates[train_n:]`.
- For non-tuned models, fit on all `window_dates`.

Create a paper-core config for final runs rather than changing `config/default.yaml`.
**`config/paper_core_rolling.yaml` — locked pre-launch configuration:**

```yaml
estimation:
  scheme: fixed                    # default fixed; CLI uses --scheme rolling for Task A
  ml_refit_every: 5                # tree models refit every 5 test days; paper uses 1
                                   # optional tree-model upgrade path: use a new --output-dir, not in-place change

models:
  regularization:
    alpha_grid_size: 80            # log-spaced; paper uses 1000.
                                   # Justified by alpha boundary stability check (see pre-launch checklist).
    alpha_min: 1.0e-5
    alpha_max: 1.0e2
    elastic_l1_ratios: [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]   # 7-point grid

  neural_network:
    seeds: 30                      # paper uses 100; reducible to 30 per Figure A.3 saturation
    ensemble_top: 10               # matches paper

feature_engineering:
  horizons: [1, 5]                 # h=22 omitted under constrained compute budget
  horizon_target_mode: future_average    # matches paper definition

experiments:
  horizons: [1, 5]
```

**Why these values (budget rationale):**

| Parameter | Value | vs Paper | Justification |
|---|---|---|---|
| `ml_refit_every` | **5** | paper=1 (daily) | ~5× speedup on tree models; rolling still captures multi-week regime shifts; may upgrade post-Task-A to 1 if speed allows |
| `alpha_grid_size` | **80** | paper=1000 | Marginal MSE impact <0.5% on log-spaced grid; saved time better spent on tree daily refit or NN seeds; verify via alpha boundary check |
| `seeds` (NN) | **30** | paper=100 | Paper's Figure A.3 (p.1722) shows ensemble saturation by N≈10-20; expected NN^10 gap is 0.5%-2% relative MSE |
| `horizons` | **[1, 5]** | paper=[1,5,22] | h=5 demonstrates "gains increase with horizon" finding as conservative lower bound; h=22 omitted to stay within constrained compute budget |
| `elastic_l1_ratios` | 7 points | paper unspecified | Spans LA (α=0) to RR (α=1); paper text only says α ∈ [0,1] |

**What was rejected and why:**
- `alpha_grid_size: 200` would cost ~1 extra day for <0.5% MSE gain; the same day is better spent on `ml_refit_every: 1` (tree daily refit, +3-7% MSE) or `seeds: 50` (NN +0.5-1% MSE).
- `ml_refit_every: 1` as the pre-launch default would risk budget overrun if tree-model rolling is slower than expected; staged as an optional post-Task-A upgrade with isolated output directory instead.

### Task A — Non-NN models, rolling scheme, h=1 + h=5

Run in a dedicated isolated output directory to protect existing results.
Use `config/paper_core_rolling.yaml` with `ml_refit_every: 5` as the launch-time setting.
Do NOT use the default `ml_refit_every: 20` from `config/default.yaml` for final
paper-method claims. The `=5` setting is a compute-budget approximation of the paper's
daily-refit design; the optional upgrade to `=1` after Task A is described in the
"optional tree-model upgrade (post-Task-A, pre-Task-C)" subsection below.

**Horizon requirement**: `config/paper_core_rolling.yaml` must have both
`feature_engineering.horizons: [1, 5]` and `experiments.horizons: [1, 5]`. Re-run
`scripts/03_build_features.py` first to materialize `target_rv_h5` and
`target_log_rv_h5` columns in `data/processed/forecasting_panel.csv` before launching
Task A; otherwise the forecast step will fail with missing target columns.

**Pre-flight check** (run once before Task A):
```bash
~/.pyenv/versions/3.11.8/bin/python -c "
import pandas as pd
panel = pd.read_csv('data/processed/forecasting_panel.csv', nrows=1)
required = {'target_rv_h1', 'target_log_rv_h1', 'target_rv_h5', 'target_log_rv_h5'}
missing = required - set(panel.columns)
if missing:
    raise SystemExit(f'Panel missing horizon columns: {missing}. Run scripts/03_build_features.py first.')
print('Panel has all required horizon columns.')
"
```

```bash
cd "/Users/celiawong/Desktop/Course004 machine learning/004_individual/rv1_professional_reproduction_code"
mkdir -p outputs_rolling/logs
screen -dmS rv1_rolling bash -lc '
~/.pyenv/versions/3.11.8/bin/python scripts/04_run_forecasts_checkpoints.py \
  --config config/paper_core_rolling.yaml \
  --output-dir outputs_rolling \
  --scheme rolling \
  --skip-nn \
  --allow-existing-output-dir \
  > outputs_rolling/logs/04_forecasts_rolling_checkpoints.log 2>&1
echo $? > outputs_rolling/logs/04_forecasts_rolling_checkpoints.exitcode
'
```

Monitor progress:
```bash
screen -ls
tail -f outputs_rolling/logs/04_forecasts_rolling_checkpoints.log
```

Output locations:
- Per-model checkpoints: `outputs_rolling/predictions/by_model/`
- Run manifest: `outputs_rolling/predictions/by_model_manifest.csv`
- Combined rolling predictions: `outputs_rolling/predictions/model_predictions.csv`

If interrupted, re-run the exact same command. Already completed by-model checkpoints
under `outputs_rolling/predictions/by_model/` will be reused automatically.

After the process exits, validate completion before using the combined file:
```bash
~/.pyenv/versions/3.11.8/bin/python -c "
import pandas as pd
manifest = pd.read_csv('outputs_rolling/predictions/by_model_manifest.csv')
bad = manifest[~manifest['status'].isin(['completed', 'reused'])]
expected_models = {
    'HAR', 'HARX', 'LogHAR', 'LevHAR', 'SHAR', 'HARQ',
    'Ridge', 'Lasso', 'ElasticNet', 'AdaptiveLasso', 'PostLasso',
    'Bagging', 'RandomForest', 'GradientBoosting',
}
seen = set(manifest['model'])
if not bad.empty:
    print(bad.to_string(index=False))
    raise SystemExit('Rolling run has failed/empty/missing checkpoints.')
missing = expected_models - seen
if missing:
    raise SystemExit(f'Missing rolling models in manifest: {sorted(missing)}')
print('Rolling manifest OK:', len(manifest), 'checkpoint rows')
"
```

#### Task A — optional tree-model upgrade (post-Task-A, pre-Task-C) (ml_refit_every: 5 → 1)

**Trigger**: Task A at `ml_refit_every: 5` completes (manifest shows all 14 models
`completed`/`reused`, no `failed`/`empty`/`missing` rows) AND there is spare compute
budget remaining before Task C must run. If so, upgrade the three tree models
(Bagging, RandomForest, GradientBoosting) to paper's daily refit. **Do NOT change
`ml_refit_every` in place** — the existing `outputs_rolling/` checkpoints would be
silently reused with the old setting.

**Safe upgrade procedure (recommended)** — use a separate output directory:

```bash
# 1) Create a new config that differs only in ml_refit_every
cp config/paper_core_rolling.yaml config/paper_core_rolling_mr1.yaml
# Edit the copy to set estimation.ml_refit_every: 1

# 2) Run only the three tree models into a new output directory
mkdir -p outputs_rolling_mr1/logs
screen -dmS rv1_rolling_mr1 bash -lc '
~/.pyenv/versions/3.11.8/bin/python scripts/04_run_forecasts_checkpoints.py \
  --config config/paper_core_rolling_mr1.yaml \
  --output-dir outputs_rolling_mr1 \
  --scheme rolling \
  --skip-nn \
  --models Bagging RandomForest GradientBoosting \
  --allow-existing-output-dir \
  > outputs_rolling_mr1/logs/04_forecasts_rolling_mr1.log 2>&1
'

# 3) Task C merge MUST be updated to source the three tree models from outputs_rolling_mr1/
#    and the other 11 non-NN models (HAR family + regularized linear) from outputs_rolling/.
#    See "Task C — multi-source merge (if optional tree-model upgrade was applied)" subsection below.
```

**Alternative (NOT recommended) — in-place force rerun**:

```bash
# Delete the three tree-model checkpoints, then --force rerun
rm outputs_rolling/predictions/by_model/Bagging_*.csv \
   outputs_rolling/predictions/by_model/RandomForest_*.csv \
   outputs_rolling/predictions/by_model/GradientBoosting_*.csv
# Edit config/paper_core_rolling.yaml to ml_refit_every: 1
# Rerun with --force on those three models
```

This works but is irreversible (the old checkpoints are gone). Prefer the "new output
directory" path for safety and ability to compare results before/after upgrade.

**When to skip the upgrade**: do NOT start the upgrade if the remaining compute budget
is less than ~5× the wall-clock time the original Task A took for the three tree models.
The upgraded run blocks Task C, which blocks Tasks D/F/H downstream. Skipping is the
safer choice; document `ml_refit_every=5` as an approximation in Section 12 and proceed.

**Final report wording (decided after upgrade attempt):**
- If upgrade succeeded → "Tree models refit daily, matching the paper's rolling scheme exactly."
- If upgrade was not attempted or did not finish → "Tree models refit every 5 test days as a compute approximation; paper uses daily refit. Expected MSE impact: 3-7% relative to paper-method tree-model rolling."

### Task B — NN models, fixed scheme, 30 seeds with checkpoints, h=1 + h=5

Paper p.1692–1693 explicitly states NN uses fixed window only. **Seeds are reduced
from the paper's 100 to 30** under the constrained compute budget. Justification: paper's
Figure A.3 shows ensemble performance saturates by N≈10-20 seeds, so the top-10
ensemble (NN^10_k) quality loss vs paper's 100 seeds is estimated at 0.5%-2% relative
MSE — dominated by other deviations (AV data, 25 stocks, no IV). Reporting in Section 12
must explicitly cite this trade-off, with reference to paper Figure A.3.

Do NOT modify `config/default.yaml` for the final run. Use
`config/paper_core_rolling.yaml` for Task B as well, and keep its
`estimation.scheme: fixed`. Task A gets rolling behavior through the CLI
`--scheme rolling` override; Task B must remain fixed to match the paper's NN method.
Use `--seed-count 30` and `--horizons 1 5` on the CLI.
Do NOT pass `--scheme rolling` to Task B.
This script (`04_run_nn_checkpoints.py`) saves one checkpoint file per seed per ticker
per dataset per horizon, so if it is interrupted it can be resumed without retraining
completed seeds. Task B is independent of Task A because it writes NN-only outputs first;
the rolling non-NN predictions are merged only in Task C.

**Smoke test first**: launch with `--seed-count 5` to validate the pipeline end-to-end
before committing to the full 30 seeds. Already-completed seed checkpoints are reused
automatically on the full run, so the 5 smoke-test seeds are not wasted.
For the 5-seed smoke run, use `--ensemble-top 5`; when scaling to 30 seeds, use
`--ensemble-top 10` to match the paper's top-10 ensemble.

```bash
cd "/Users/celiawong/Desktop/Course004 machine learning/004_individual/rv1_professional_reproduction_code"
mkdir -p outputs_nn30_checkpointed/logs
# IMPORTANT: do NOT pass --scheme rolling to Task B. The config's
# estimation.scheme: fixed is correct for NN training; the NN runner
# writes this field into predictions, and Task C's schema check will
# fail if NN predictions are mislabeled as rolling.
screen -dmS rv1_nn30 bash -lc '
~/.pyenv/versions/3.11.8/bin/python scripts/04_run_nn_checkpoints.py \
  --config config/paper_core_rolling.yaml \
  --output-dir outputs_nn30_checkpointed \
  --datasets MHAR PARTIAL_MALL \
  --horizons 1 5 \
  --models NN1 NN2 NN3 NN4 \
  --seed-count 30 \
  --ensemble-top 10 \
  --base-predictions "" \
  --allow-existing-output-dir \
  > outputs_nn30_checkpointed/logs/04_nn30_checkpointed.log 2>&1
echo $? > outputs_nn30_checkpointed/logs/04_nn30_checkpointed.exitcode
'
```

**Pre-flight check**: confirm `04_run_nn_checkpoints.py` accepts `--horizons` and that
`forecasting_panel.csv` has `target_rv_h5` and `target_log_rv_h5` columns (re-run
`scripts/03_build_features.py` first if needed; requires `feature_engineering.horizons`
and `experiments.horizons` both set to `[1, 5]` in config).

If interrupted, re-run the exact same command. Already completed seed checkpoints under
`outputs_nn30_checkpointed/nn_seed_predictions/` will be reused automatically.

Monitor progress:
```bash
tail -f outputs_nn30_checkpointed/logs/04_nn30_checkpointed.log
```

Output locations:
- Per-seed checkpoints: `outputs_nn30_checkpointed/nn_seed_predictions/<dataset>/h{1,5}/<model>/<ticker>/seed_*.csv`
- Dataset-model aggregates: `outputs_nn30_checkpointed/predictions/by_model/`
- NN-only aggregate: `outputs_nn30_checkpointed/predictions/nn_model_predictions.csv`
- NN-only final predictions: `outputs_nn30_checkpointed/predictions/model_predictions.csv`

After the process exits, validate the NN-only aggregate:
```bash
~/.pyenv/versions/3.11.8/bin/python -c "
import pandas as pd
nn = pd.read_csv('outputs_nn30_checkpointed/predictions/nn_model_predictions.csv')
expected = {'NN1', 'NN2', 'NN3', 'NN4'}
models = set(nn['model'])
if models != expected:
    raise SystemExit(f'Unexpected NN model set: {sorted(models)}')
schemes = set(nn['scheme'])
if schemes != {'fixed'}:
    raise SystemExit(f'NN predictions should be fixed-scheme only, got: {sorted(schemes)}')
dup = nn.duplicated(['date', 'ticker', 'dataset', 'horizon', 'model']).sum()
if dup:
    raise SystemExit(f'Duplicate NN prediction keys: {dup}')
print('NN aggregate OK:', len(nn), 'rows')
"
```

### Task D — NN^1 single-best-seed extraction (run after Task B, before Task C)

This task does NOT retrain anything. It reads the per-seed checkpoints written by Task B
in `outputs_nn30_checkpointed/nn_seed_predictions/`, selects for each
`(dataset, model, ticker)` the seed with the lowest `val_mse`, applies the configured
positivity and insanity filters, and writes four single-seed prediction tables named
`NN1_1`, `NN2_1`, `NN3_1`, `NN4_1` (paper notation: NN^1_k for k=1..4).

The script to write is `scripts/06_extract_nn_best_single_seed.py`. Its contract:

| Aspect | Requirement |
|---|---|
| Input | `outputs_nn30_checkpointed/nn_seed_predictions/<dataset>/h{horizon}/<model>/<ticker>/seed_*.csv` (the runner segregates per-seed files by horizon since the 2026-05-18 patch) |
| Selection | argmin over available `val_mse` per `(dataset, model, ticker)` |
| Postprocessing | Same `enforce_positive_forecasts` and `insanity_filter` as Task B, using each seed file's stored `in_sample_min_rv` and `in_sample_mean_rv` |
| Output model names | `NN1_1`, `NN2_1`, `NN3_1`, `NN4_1` (string, exact case) |
| Output schema | Same columns as `outputs_nn30_checkpointed/predictions/nn_model_predictions.csv` (drop-in compatible) |
| Output file | `outputs_nn30_checkpointed/predictions/nn1_model_predictions.csv` |
| Idempotency | Safe to rerun; overwrite the output via atomic write (`.tmp` then `replace`) |
| Failure modes | If a `(dataset, model, ticker)` has zero seed files, raise — do not silently skip |

Run command (once the script is implemented):

```bash
cd "/Users/celiawong/Desktop/Course004 machine learning/004_individual/rv1_professional_reproduction_code"
~/.pyenv/versions/3.11.8/bin/python scripts/06_extract_nn_best_single_seed.py \
  --config config/paper_core_rolling.yaml \
  --output-dir outputs_nn30_checkpointed \
  > outputs_nn30_checkpointed/logs/06_extract_nn1.log 2>&1
```

Validation after the script exits:
```bash
~/.pyenv/versions/3.11.8/bin/python -c "
import pandas as pd
nn1 = pd.read_csv('outputs_nn30_checkpointed/predictions/nn1_model_predictions.csv')
expected = {'NN1_1', 'NN2_1', 'NN3_1', 'NN4_1'}
models = set(nn1['model'])
if models != expected:
    raise SystemExit(f'Unexpected NN1 model set: {sorted(models)}')
schemes = set(nn1['scheme'])
if schemes != {'fixed'}:
    raise SystemExit(f'NN1 predictions should be fixed-scheme only, got: {sorted(schemes)}')
dup = nn1.duplicated(['date', 'ticker', 'dataset', 'horizon', 'model']).sum()
if dup:
    raise SystemExit(f'Duplicate NN1 prediction keys: {dup}')
print('NN1 aggregate OK:', len(nn1), 'rows')
"
```

### Execution order

- Task A writes rolling non-NN outputs to `outputs_rolling/`. Can run in parallel with B.
- Task B writes fixed NN ensemble outputs to `outputs_nn30_checkpointed/`. Can run in parallel with A.
- Task D runs after Task B completes (depends on Task B's per-seed checkpoints).
- Task C runs after BOTH Task A and Task D complete.
- Task E (Figure 6 ALE) is independent and can run any time after `forecasting_panel.csv` exists.
- Task F (Figure 5 RV-decile MSE) runs after Task C (it reads `outputs_final/predictions/model_predictions.csv`).
- Task G (partial Figure 7 variable importance) is independent and can run any time after the panel exists.
  Optionally reuses Task B's NN2 per-seed checkpoints for the NN^10_2 component.

### Task C — Merge and evaluate (run after Task A, Task B, and Task D complete)

The final evaluation requires combining Task A results (14 rolling non-NN models),
Task B results (4 fixed NN^10 ensemble models), and Task D results (4 fixed NN^1
single-seed models) into a single prediction file with **22 models total**.

```bash
cd "/Users/celiawong/Desktop/Course004 machine learning/004_individual/rv1_professional_reproduction_code"
mkdir -p outputs_final/predictions outputs_final/logs outputs_final/tables outputs_final/figures

~/.pyenv/versions/3.11.8/bin/python -c "
import pandas as pd
rolling = pd.read_csv('outputs_rolling/predictions/model_predictions.csv')
nn = pd.read_csv('outputs_nn30_checkpointed/predictions/nn_model_predictions.csv')
nn1 = pd.read_csv('outputs_nn30_checkpointed/predictions/nn1_model_predictions.csv')
combined = pd.concat([rolling, nn, nn1], ignore_index=True)
duplicates = combined.duplicated(['date', 'ticker', 'dataset', 'horizon', 'model']).sum()
if duplicates:
    raise RuntimeError(f'Duplicate prediction keys after merge: {duplicates}')
expected_non_nn = {
    'HAR', 'HARX', 'LogHAR', 'LevHAR', 'SHAR', 'HARQ',
    'Ridge', 'Lasso', 'ElasticNet', 'AdaptiveLasso', 'PostLasso',
    'Bagging', 'RandomForest', 'GradientBoosting',
}
expected_nn = {'NN1', 'NN2', 'NN3', 'NN4'}
expected_nn1 = {'NN1_1', 'NN2_1', 'NN3_1', 'NN4_1'}
models = set(combined['model'])
expected = expected_non_nn | expected_nn | expected_nn1
if models != expected:
    raise RuntimeError(f'Unexpected final model set: got={sorted(models)}, expected={sorted(expected)}')
rolling_schemes = sorted(rolling['scheme'].unique())
nn_schemes = sorted(nn['scheme'].unique())
nn1_schemes = sorted(nn1['scheme'].unique())
datasets = sorted(combined['dataset'].unique())
horizons = sorted(combined['horizon'].unique())
if set(rolling_schemes) != {'rolling'}:
    raise RuntimeError(f'Rolling file has unexpected schemes: {rolling_schemes}')
if set(nn_schemes) != {'fixed'}:
    raise RuntimeError(f'NN ensemble file has unexpected schemes: {nn_schemes}')
if set(nn1_schemes) != {'fixed'}:
    raise RuntimeError(f'NN1 single-seed file has unexpected schemes: {nn1_schemes}')
if set(datasets) != {'MHAR', 'PARTIAL_MALL'}:
    raise RuntimeError(f'Unexpected datasets: {datasets}')
if set(horizons) != {1, 5}:
    raise RuntimeError(f'Unexpected horizons for this h=1+h=5 final run: {horizons}')
combined.to_csv('outputs_final/predictions/model_predictions.csv', index=False)
print('Combined rows:', len(combined))
print('Models:', sorted(combined['model'].unique()))
"

~/.pyenv/versions/3.11.8/bin/python scripts/05_evaluate_outputs_isolated.py \
  --config config/paper_core_rolling.yaml \
  --output-dir outputs_final
```

Final results (22 models: 14 rolling non-NN + 4 fixed NN^10 ensemble + 4 fixed NN^1
single-seed) will be in `outputs_final/tables/` and `outputs_final/figures/`.

#### Task C — multi-source merge (if optional tree-model upgrade was applied)

If the optional tree-model upgrade ran tree models with `ml_refit_every: 1` into
`outputs_rolling_mr1/`, the merge must source the three tree models from the upgraded
directory and the other 11 non-NN models from the original `outputs_rolling/` directory.

```python
# Replace the single rolling read with a model-aware multi-source merge
rolling_default = pd.read_csv('outputs_rolling/predictions/model_predictions.csv')
rolling_mr1 = pd.read_csv('outputs_rolling_mr1/predictions/model_predictions.csv')

tree_models = {'Bagging', 'RandomForest', 'GradientBoosting'}
non_tree_models = {
    'HAR', 'HARX', 'LogHAR', 'LevHAR', 'SHAR', 'HARQ',
    'Ridge', 'Lasso', 'ElasticNet', 'AdaptiveLasso', 'PostLasso',
}

# Take tree models from upgraded run, others from default run
rolling = pd.concat([
    rolling_default[rolling_default['model'].isin(non_tree_models)],
    rolling_mr1[rolling_mr1['model'].isin(tree_models)],
], ignore_index=True)

# Verify model set is complete and unique
seen = set(rolling['model'])
expected = tree_models | non_tree_models
assert seen == expected, f'Missing or extra models: {expected.symmetric_difference(seen)}'
assert not rolling.duplicated(['date', 'ticker', 'dataset', 'horizon', 'model']).any()
```

Document the upgrade in `outputs_final/run_provenance.txt`:
- `Bagging, RandomForest, GradientBoosting`: `ml_refit_every=1` (paper-method daily refit)
- All other non-NN models: `ml_refit_every=5` (irrelevant for HAR family and regularized linear, which retrain every test day regardless)

### Task E — ALE figures (independent, can run any time after panel exists)

Generates Figure 6-style ALE plots for the five reference models and the three available
reference features from the paper. This task does not depend on Task A/B/C/D outputs because it
refits the five models from scratch on Apple (`AAPL`) rows in the existing
`forecasting_panel.csv`. This matches the paper's treatment: the ALE plot is an
asset-level illustration, not a cross-sectional average.

Reference set (matches paper Figure 6):

| Model | Notes |
|---|---|
| HARX | OLS on PARTIAL_MALL features |
| LogHAR | Log-target HAR on PARTIAL_MALL extended features. Current no-IV panel uses `log_vix` and omits IV; `log_iv` is used only if real `firm_iv.csv` is supplied. |
| ElasticNet | Validation-selected alpha/l1 grid from `config/paper_core_rolling.yaml` |
| RandomForest | n_estimators=500, max_features=1/3, min_samples_leaf=5 |
| NN^10_2 | Top-10 ensemble of NN2 architecture; reuse the 30 seeds from Task B for AAPL (PARTIAL_MALL, h=1) if available, else train 30 seeds for AAPL and pick the top 10 by validation MSE |

Reference features (paper Figure 6 minus IV):

| Paper feature | Implementation |
|---|---|
| RVD | `rvd` |
| RVW | `rvw` |
| IV | **OMITTED** — OptionMetrics stock-level IV is unavailable. VIX is a market-level index and is NOT a valid substitute; substituting it would change the semantic meaning of the ALE plot from "individual stock IV effect" to "market volatility effect" and produce misleading conclusions. |
| M1W | `m1w` |

The ALE figure therefore has **3 rows (features) × 5 columns (models)** instead of the
paper's 4×5 grid. Plot the ALE over standardized feature values in the interval [-1, 1],
as in the paper.

The script to write is `scripts/07_compute_ale.py`. Its contract:

| Aspect | Requirement |
|---|---|
| Input panel | `data/processed/forecasting_panel.csv` |
| Dataset | PARTIAL_MALL only (the extended feature set on which the reference models are trained) |
| Asset | AAPL only |
| Pooling | **No cross-sectional pooling** for Figure 6-style ALE. Fit and explain AAPL only. |
| In-sample rows | Use the AAPL training/validation rows consistent with each model's paper fitting rule; compute ALE on the standardized in-sample feature matrix. |
| NN^10_2 seed source | If `outputs_nn30_checkpointed/nn_seed_predictions/PARTIAL_MALL/h1/NN2/AAPL/seed_*.csv` exists, reuse the 10 AAPL seeds with lowest validation MSE; else train 30 AAPL seeds and pick top-10 by validation MSE. |
| ALE function | `rv1rep.explain.accumulated_local_effect`, with paper-style quantile bins (`grid_size=100`) |
| Output table | `outputs_ale/ale_table.csv` with columns `ticker, model, feature, x_standardized, ale` |
| Output figure | `outputs_ale/figure6_ale.png` — AAPL only, 3 rows (features) × 5 cols (models), x restricted to [-1, 1] standard deviations |
| Idempotency | Refits every run; cheap enough to not need checkpointing |
| Failure modes | If AAPL has fewer than 1000 usable in-sample rows after feature filtering, raise |

Run command (once the script is implemented):

```bash
cd "/Users/celiawong/Desktop/Course004 machine learning/004_individual/rv1_professional_reproduction_code"
mkdir -p outputs_ale/logs
~/.pyenv/versions/3.11.8/bin/python scripts/07_compute_ale.py \
  --config config/paper_core_rolling.yaml \
  --output-dir outputs_ale \
  > outputs_ale/logs/07_ale.log 2>&1
```

Validation after the script exits:
```bash
~/.pyenv/versions/3.11.8/bin/python -c "
import pandas as pd
ale = pd.read_csv('outputs_ale/ale_table.csv')
expected_models = {'HARX', 'LogHAR', 'ElasticNet', 'RandomForest', 'NN10_2'}
expected_features = {'rvd', 'rvw', 'm1w'}
models = set(ale['model'])
features = set(ale['feature'])
tickers = set(ale['ticker'])
if models != expected_models:
    raise SystemExit(f'Unexpected ALE model set: {sorted(models)}')
if features != expected_features:
    raise SystemExit(f'Unexpected ALE feature set: {sorted(features)}')
if tickers != {'AAPL'}:
    raise SystemExit(f'ALE should be AAPL-only, got: {sorted(tickers)}')
print('ALE table OK:', len(ale), 'rows;', len(models), 'models x', len(features), 'features')
import os
fig = 'outputs_ale/figure6_ale.png'
if not os.path.exists(fig):
    raise SystemExit(f'Missing figure: {fig}')
print('Figure file present:', fig, os.path.getsize(fig), 'bytes')
"
```

### Task F — Figure 5: MSE by Realized Variance Decile (post-processing)

#### Purpose
Reproduce the paper's realized-variance-decile comparison: split the PARTIAL_MALL test
set into deciles of observed RV and compute MSE for the selected reference models within
selected low, medium, and high volatility states. This shows where each model is strong or
weak across volatility regimes.

#### Prerequisites
- Task C complete (`outputs_final/predictions/model_predictions.csv` exists with 22
  models: 14 rolling non-NN + 4 fixed NN^10 + 4 fixed NN^1).

#### Script to create
`scripts/06b_compute_rv_decile_mse.py`

#### Contract

| Aspect | Requirement |
|---|---|
| Input | `outputs_final/predictions/model_predictions.csv` |
| Dataset | PARTIAL_MALL only for the primary paper-style figure |
| Models in primary figure | HARX, LogHAR, ElasticNet, RandomForest, NN2 (`NN^10_2`) |
| Decile basis | Compute deciles of `actual_rv` for PARTIAL_MALL h=1 test observations. Use `pandas.qcut` with `q=10`, `duplicates='drop'`. |
| Deciles in primary figure | (0.0,0.1), (0.1,0.2), (0.5,0.6), (0.8,0.9), (0.9,1.0), represented as decile labels 1, 2, 6, 9, 10 |
| Pooling | Pool all 25 tickers within a decile (paper-style pooled MSE per decile) |
| MSE | Standard mean squared error per (dataset, horizon, model, decile) |
| Output table | `outputs_final/tables/rv_decile_mse.csv` with all models/all deciles for auditability, columns `dataset, horizon, model, decile, percentile_bin, n_obs, mse, rel_mse_vs_har` |
| Output figure | `outputs_final/figures/figure5_rv_decile_mse.png` — primary paper-style figure: selected PARTIAL_MALL models and selected deciles only |
| Idempotency | Overwrite outputs each run; no checkpointing needed |
| Failure modes | If any (dataset, horizon, model, decile) has fewer than 10 observations, log warning but do not abort. If `actual_rv` has zero variance, raise. |

#### Run command
```bash
cd "/Users/celiawong/Desktop/Course004 machine learning/004_individual/rv1_professional_reproduction_code"
~/.pyenv/versions/3.11.8/bin/python scripts/06b_compute_rv_decile_mse.py \
  --predictions outputs_final/predictions/model_predictions.csv \
  --output-dir outputs_final \
  > outputs_final/logs/06b_rv_decile_mse.log 2>&1
```

#### Validation
```bash
~/.pyenv/versions/3.11.8/bin/python -c "
import os
import pandas as pd
df = pd.read_csv('outputs_final/tables/rv_decile_mse.csv')
expected_cols = {'dataset', 'horizon', 'model', 'decile', 'percentile_bin', 'n_obs', 'mse', 'rel_mse_vs_har'}
missing = expected_cols - set(df.columns)
if missing:
    raise SystemExit(f'Missing columns: {missing}')
deciles = sorted(df['decile'].unique())
if deciles != list(range(1, 11)) and deciles != list(range(0, 10)):
    raise SystemExit(f'Unexpected decile labels: {deciles}')
primary = df[(df['dataset'] == 'PARTIAL_MALL') & (df['horizon'].astype(int) == 1)]
required_models = {'HARX', 'LogHAR', 'ElasticNet', 'RandomForest', 'NN2'}
missing_primary = required_models - set(primary['model'])
if missing_primary:
    raise SystemExit(f'Missing primary Figure 5 models: {sorted(missing_primary)}')
if df['rel_mse_vs_har'].isna().any():
    raise SystemExit('rel_mse_vs_har contains NaN')
fig = 'outputs_final/figures/figure5_rv_decile_mse.png'
if not os.path.exists(fig):
    raise SystemExit(f'Missing figure: {fig}')
print('Task F OK:', len(df), 'rows;', df['model'].nunique(), 'models;', df['dataset'].nunique(), 'datasets')
"
```

### Task G — Partial Figure 7: Cross-sectional Variable Importance (independent, can run any time after panel exists)

#### Purpose
Produce a Figure 7-style variable-importance analysis using the paper's ALE-based VI
definition. For each ticker and each reference model, compute ALE for every available
PARTIAL_MALL predictor, convert each ALE curve into an importance score via the sample
standard deviation of centered ALE values, normalize importances to sum to one, and then
average across the 25 tickers. This is partial because the paper's stock-level IV feature is
unavailable. Do not replace IV with VIX; VIX remains only the distinct market-volatility
predictor.

#### Prerequisites
- `data/processed/forecasting_panel.csv` exists (built by `scripts/03_build_features.py`).
- This task does NOT depend on Task A/B/C/D outputs because it refits models from scratch.
- Task B output (`outputs_nn30_checkpointed/nn_seed_predictions/`) is optional: if
  available, the NN^10_2 seed selection reuses those checkpoints; otherwise the script
  trains 30 fresh seeds under the same compute-budget seed policy.

#### Script to create
`scripts/06c_compute_variable_importance.py`

#### Contract

| Aspect | Requirement |
|---|---|
| Reference models | HARX, ElasticNet, RandomForest, NN^10_2 (top-10 ensemble of NN2) |
| Dataset | PARTIAL_MALL only (extended features) |
| Per-ticker fitting | For each of 25 tickers, fit each model on its in-sample rows using the model's paper fitting rule, then compute ALE-based VI on that ticker's standardized in-sample feature matrix. |
| NN^10_2 source | If Task B is complete, reuse the top-10 selected seeds from `outputs_nn30_checkpointed/nn_seed_predictions/PARTIAL_MALL/h1/NN2/<ticker>/` (read params, pick lowest 10 val_mse); else train 30 fresh seeds and pick top-10. |
| Importance function | ALE-based: `I(feature) = sample_std(centered_ALE(feature))`; `VI(feature) = I(feature) / sum_j I(feature_j)` |
| ALE grid | 100 quantile intervals, matching the paper's ALE implementation |
| Aggregation | Mean and median of `vi_normalized` across contributing tickers per (model, feature) |
| Output table | `outputs_final/tables/variable_importance.csv` with columns `model, feature, n_tickers, vi_mean, vi_median, vi_std` |
| Output figure | `outputs_final/figures/figure7_variable_importance.png` — horizontal bar chart, one panel per model, sorted by `vi_mean` descending |
| IV handling | `iv` and `log_iv` must be absent unless the user supplies real `data/external/firm_iv.csv`; never substitute VIX for IV. |
| Idempotency | Refits every run; no checkpointing |
| Failure modes | If any ticker has fewer than 100 usable in-sample rows, log and skip that ticker for that model. If fewer than 10 tickers contribute, raise. If VI does not sum to approximately one within a model/ticker, raise. |

#### Run command
```bash
cd "/Users/celiawong/Desktop/Course004 machine learning/004_individual/rv1_professional_reproduction_code"
mkdir -p outputs_final/logs outputs_final/tables outputs_final/figures
~/.pyenv/versions/3.11.8/bin/python scripts/06c_compute_variable_importance.py \
  --config config/paper_core_rolling.yaml \
  --nn-seed-source outputs_nn30_checkpointed/nn_seed_predictions \
  --output-dir outputs_final \
  > outputs_final/logs/06c_variable_importance.log 2>&1
```

#### Validation
```bash
~/.pyenv/versions/3.11.8/bin/python -c "
import os
import pandas as pd
df = pd.read_csv('outputs_final/tables/variable_importance.csv')
expected_models = {'HARX', 'ElasticNet', 'RandomForest', 'NN10_2'}
models = set(df['model'])
if models != expected_models:
    raise SystemExit(f'Unexpected model set: {sorted(models)}')
features = set(df['feature'].astype(str).str.lower())
if 'iv' in features or 'log_iv' in features:
    raise SystemExit('IV appears in variable importance output; this is only allowed with real firm_iv.csv input.')
if (df['n_tickers'] < 10).any():
    raise SystemExit('Some (model, feature) pairs have fewer than 10 contributing tickers')
for model, g in df.groupby('model'):
    total = float(g['vi_mean'].sum())
    if not 0.95 <= total <= 1.05:
        raise SystemExit(f'VI means should sum to about 1 for {model}, got {total}')
fig = 'outputs_final/figures/figure7_variable_importance.png'
if not os.path.exists(fig):
    raise SystemExit(f'Missing figure: {fig}')
print('Task G OK:', len(df), 'rows;', df['model'].nunique(), 'models;', df['feature'].nunique(), 'features')
"
```

### Task H — Paper-style filtered historical simulation VaR

#### Purpose
Replace the current normal-parametric VaR diagnostic with the paper's filtered historical
simulation (FHS) VaR application. The paper evaluates one-day-ahead 5% VaR forecasts
using the quantile/check loss and Kupiec/Christoffersen coverage tests.

#### Prerequisites
- Task C complete (`outputs_final/predictions/model_predictions.csv` exists).
- The script must be able to recover or refit in-sample volatility forecasts/residuals for
  the same ticker, dataset, horizon, model, and scheme used in the final prediction file.

#### Script to create
`scripts/06d_compute_fhs_var.py`

#### Contract

| Aspect | Requirement |
|---|---|
| Input predictions | `outputs_final/predictions/model_predictions.csv` |
| Alpha | 0.05 primary; optionally allow 0.01 as a robustness flag |
| Calibration | For each `(ticker, dataset, model)`, estimate the empirical alpha quantile of standardized in-sample returns. Standardize returns using the model's in-sample volatility forecasts, not a normal quantile. |
| Test VaR | `VaR_t = empirical_quantile_alpha * sqrt(forecast_rv_t)` |
| Loss | Quantile/check loss: `(alpha - 1{return < VaR}) * (return - VaR)` |
| Coverage tests | Kupiec unconditional coverage and Christoffersen independence |
| Output forecasts | `outputs_final/predictions/var_forecasts_fhs.csv` |
| Output summary | `outputs_final/tables/var_backtest_fhs_summary.csv` |
| Method guard | Do not call `rv1rep.var_backtest.make_var_forecasts()` for the primary FHS output; that function is normal-parametric. |
| Failure modes | If in-sample standardized residuals cannot be recovered for a model/ticker, fail loudly rather than falling back to normal VaR. |

#### Run command
```bash
cd "/Users/celiawong/Desktop/Course004 machine learning/004_individual/rv1_professional_reproduction_code"
~/.pyenv/versions/3.11.8/bin/python scripts/06d_compute_fhs_var.py \
  --config config/paper_core_rolling.yaml \
  --predictions outputs_final/predictions/model_predictions.csv \
  --output-dir outputs_final \
  --alpha 0.05 \
  > outputs_final/logs/06d_fhs_var.log 2>&1
```

#### Validation
```bash
~/.pyenv/versions/3.11.8/bin/python -c "
import os
import pandas as pd
var = pd.read_csv('outputs_final/predictions/var_forecasts_fhs.csv')
summary = pd.read_csv('outputs_final/tables/var_backtest_fhs_summary.csv')
required_var = {'date', 'ticker', 'dataset', 'horizon', 'model', 'return', 'var_forecast', 'hit', 'var_loss', 'method'}
required_summary = {'dataset', 'horizon', 'ticker', 'model', 'n', 'exceedance_rate', 'mean_var_loss', 'kupiec_p', 'christoffersen_ind_p', 'method'}
if missing := (required_var - set(var.columns)):
    raise SystemExit(f'Missing VaR forecast columns: {missing}')
if missing := (required_summary - set(summary.columns)):
    raise SystemExit(f'Missing VaR summary columns: {missing}')
var_methods = set(var['method'])
summary_methods = set(summary['method'])
if var_methods != {'filtered_historical_simulation'}:
    raise SystemExit(f'Unexpected VaR method labels: {sorted(var_methods)}')
if summary_methods != {'filtered_historical_simulation'}:
    raise SystemExit(f'Unexpected summary method labels: {sorted(summary_methods)}')
print('Task H OK:', len(var), 'VaR rows;', len(summary), 'summary rows')
"
```

### Task I — Model Confidence Set (Figure 4)

#### Purpose
Reproduce paper Figure 4: per-ticker MCS at 90% confidence (Hansen, Lunde, Nason 2011)
showing the percentage of tickers for which each model is retained in the final
confidence set. MCS is set-level statistical evidence complementing the pairwise DM
test results; it directly supports paper's finding (1) "ML beats HAR." Implementation
uses the `arch` package's `arch.bootstrap.MCS` class to avoid reimplementing the
elimination tournament + stationary block bootstrap.

#### Prerequisites
- Task C complete (`outputs_final/predictions/model_predictions.csv` exists with all 22 models).
- `arch` Python package installed.

#### Dependency installation

```bash
~/.pyenv/versions/3.11.8/bin/pip install arch
# Verify
~/.pyenv/versions/3.11.8/bin/python -c "from arch.bootstrap import MCS; print('MCS OK')"
```

#### Script to create
`scripts/06e_compute_mcs.py`

#### Contract

| Aspect | Requirement |
|---|---|
| Input predictions | `outputs_final/predictions/model_predictions.csv` |
| Loss function | Squared error: `(actual_rv_t − forecast_rv_t)^2`, computed per (ticker, dataset, horizon, model, test_date) |
| Per-ticker pivot | For each (ticker, dataset, horizon), build a `(n_test_days × 22 models)` loss matrix. Drop test dates where any model has NaN forecast. |
| MCS engine | `arch.bootstrap.MCS(losses, size=0.10, reps=5000, block_size=10, method='max')`. `size=0.10` gives 90% confidence set matching paper. `method='max'` uses T_max statistic. `block_size=10` for stationary block bootstrap. |
| Reps | 5,000 bootstrap reps (paper does not specify; 5,000 is the standard `arch` default with statistically stable MCS results; 10,000 if budget allows) |
| Aggregation | For each (dataset, horizon, model) pair, compute `inclusion_rate = (# tickers where model is in MCS) / 25`. |
| Output table | `outputs_final/tables/mcs_inclusion_rates.csv` with columns `dataset, horizon, model, inclusion_rate, n_tickers_included, n_tickers_total` |
| Output detail (per-ticker) | `outputs_final/tables/mcs_per_ticker.csv` with columns `dataset, horizon, ticker, model, in_mcs, mcs_pvalue` |
| Output figure | `outputs_final/figures/figure4_mcs.png` — 4 panels (2 datasets × 2 horizons), each panel: bar chart of inclusion rate by model, ordered by paper-style grouping (HAR family / regularized linear / trees / NN^10 / NN^1) |
| Idempotency | Overwrite outputs each run; no checkpointing needed |
| Failure modes | If any (dataset, horizon, ticker) has fewer than 200 valid test rows, log warning and skip that ticker. If fewer than 20 of 25 tickers contribute to any (dataset, horizon, model) MCS, raise. |

#### Implementation skeleton

```python
# scripts/06e_compute_mcs.py
import numpy as np
import pandas as pd
from arch.bootstrap import MCS

preds = pd.read_csv('outputs_final/predictions/model_predictions.csv')
# Need actual_rv per (date, ticker, dataset, horizon) — already in preds as 'actual_rv'
preds['sq_err'] = (preds['actual_rv'] - preds['forecast_rv']) ** 2

results_per_ticker = []
for (dataset, horizon, ticker), grp in preds.groupby(['dataset', 'horizon', 'ticker']):
    # Pivot to (test_date × model) loss matrix
    loss_mat = grp.pivot(index='date', columns='model', values='sq_err').dropna(how='any')
    if len(loss_mat) < 200:
        continue
    mcs = MCS(loss_mat.values, size=0.10, reps=5000, block_size=10, method='max')
    mcs.compute()
    # mcs.included is an array of column indices of retained models
    included_models = loss_mat.columns[mcs.included].tolist()
    pvals = dict(zip(loss_mat.columns, mcs.pvalues['Pvalue']))
    for model in loss_mat.columns:
        results_per_ticker.append({
            'dataset': dataset, 'horizon': horizon, 'ticker': ticker,
            'model': model,
            'in_mcs': model in included_models,
            'mcs_pvalue': pvals[model],
        })

per_ticker = pd.DataFrame(results_per_ticker)
per_ticker.to_csv('outputs_final/tables/mcs_per_ticker.csv', index=False)

# Aggregate to inclusion rates
agg = per_ticker.groupby(['dataset', 'horizon', 'model']).agg(
    n_tickers_included=('in_mcs', 'sum'),
    n_tickers_total=('in_mcs', 'count'),
).reset_index()
agg['inclusion_rate'] = agg['n_tickers_included'] / agg['n_tickers_total']
agg.to_csv('outputs_final/tables/mcs_inclusion_rates.csv', index=False)

# Plot figure4_mcs.png (4 panels: 2 datasets × 2 horizons)
# ... matplotlib code with paper-style grouping ...
```

#### Run command
```bash
cd "/Users/celiawong/Desktop/Course004 machine learning/004_individual/rv1_professional_reproduction_code"
~/.pyenv/versions/3.11.8/bin/python scripts/06e_compute_mcs.py \
  --predictions outputs_final/predictions/model_predictions.csv \
  --output-dir outputs_final \
  --confidence 0.90 \
  --reps 5000 \
  --block-size 10 \
  > outputs_final/logs/06e_mcs.log 2>&1
```

#### Validation
```bash
~/.pyenv/versions/3.11.8/bin/python -c "
import os
import pandas as pd
agg = pd.read_csv('outputs_final/tables/mcs_inclusion_rates.csv')
per_ticker = pd.read_csv('outputs_final/tables/mcs_per_ticker.csv')
expected_models = {
    'HAR', 'HARX', 'LogHAR', 'LevHAR', 'SHAR', 'HARQ',
    'Ridge', 'Lasso', 'ElasticNet', 'AdaptiveLasso', 'PostLasso',
    'Bagging', 'RandomForest', 'GradientBoosting',
    'NN1', 'NN2', 'NN3', 'NN4',
    'NN1_1', 'NN2_1', 'NN3_1', 'NN4_1',
}
got_models = set(agg['model'])
if got_models != expected_models:
    raise SystemExit(f'Unexpected MCS model set: missing={expected_models - got_models}, extra={got_models - expected_models}')
if set(agg['dataset']) != {'MHAR', 'PARTIAL_MALL'}:
    raise SystemExit(f'Unexpected datasets in MCS: {sorted(set(agg[\"dataset\"]))}')
if set(agg['horizon']) != {1, 5}:
    raise SystemExit(f'Unexpected horizons in MCS: {sorted(set(agg[\"horizon\"]))}')
# Sanity: inclusion_rate in [0, 1]
if not ((agg['inclusion_rate'] >= 0).all() and (agg['inclusion_rate'] <= 1).all()):
    raise SystemExit('inclusion_rate outside [0, 1]')
# Sanity: every (dataset, horizon, model) must have at least 20 contributing tickers
low_coverage = agg[agg['n_tickers_total'] < 20]
if not low_coverage.empty:
    print('WARNING: low ticker coverage:'); print(low_coverage.to_string(index=False))
fig = 'outputs_final/figures/figure4_mcs.png'
if not os.path.exists(fig):
    raise SystemExit(f'Missing figure: {fig}')
print('Task I OK:', len(agg), 'inclusion rows;', len(per_ticker), 'per-ticker rows;', agg['model'].nunique(), 'models')
"
```

#### Why arch.bootstrap.MCS
The `arch` package (maintained by Kevin Sheppard, co-author of Patton & Sheppard 2015 —
the SHAR model used in this paper) implements the Hansen-Lunde-Nason MCS exactly as
described in the original paper, with stationary block bootstrap. Using it avoids
the risk of bugs from reimplementing the elimination tournament. The package is
the de-facto standard in financial econometrics for MCS computation.

#### Wall-clock estimate
- Script writing + skeleton verification: 1-2 hours human
- Running 100 MCS calls (25 tickers × 2 datasets × 2 horizons) at 5,000 reps × 22 models × ~800 days: ~1-3 hours background
- Plotting Figure 4: 1 hour human
- **Total: 3-6 hours, fully parallelizable with Tasks E/F/G/H**

---

## 12. What Can Legitimately Be Claimed in a Report

### Based on currently completed historical results (outputs_full_nn/, fixed scheme, 20 seeds)

> **NOTE (for future agents): This subsection describes a pre-rerun historical state
> captured before Phase 0 of the current plan. After Phase 6 (post-processing) completes,
> the canonical results live in `outputs_final/`, and this subsection is preserved
> for traceability only. When updating Section 12 with actual completed results,
> update the "After planned rerun completes" subsection below; do NOT modify this
> historical subsection.**

These claims are only valid as a description of the old completed CSV run. They are not
final paper-comparable claims until the LogHAR transform fix has been rerun.

**Can claim:**
- Full pipeline implemented: RV construction → feature engineering → 18 models →
  evaluation → normal-parametric VaR diagnostic
- LogHAR best in HAR family, RF best in ML family — directionally consistent with paper
- NN successfully trained (4 architectures, 20-seed ensemble, top-10 average)
- PARTIAL_MALL results obtained for all 18 models
- DM tests and pairwise relative MSE matrix computed
- VaR diagnostic coverage tests (Kupiec + Christoffersen) computed

**Must acknowledge:**
- Fixed scheme used throughout (paper uses rolling for non-NN models)
- Current rolling implementation must be corrected before final paper-method use because it
  discards part of each rolling train/validation window.
- The default tree rolling setting uses `ml_refit_every=20`; final paper-method rolling
  requires daily refit or must be labelled as an approximation.
- 25 stocks not 29 (CVX, TRV, RTX, DOW missing due to AV data limitation)
- AV panel is not a strict balanced 29-stock TAQ panel; effective test counts differ by ticker
- IV omitted from MALL; dataset labelled PARTIAL_MALL
- Existing completed CSV outputs were generated before the LogHAR `log_vix` fix for the
  current no-IV panel and should be rerun before using them for final claims
- 20 NN seeds not 100; only ensemble reported, not single-seed
- h=1 only; weekly/monthly horizons not computed
- ALE (Figure 6), variable-importance ranking (Figure 7), and RV-decile MSE (Figure 5)
  not yet computed in `outputs_full_nn/` — these are planned as Tasks E, G, and F
- Alpha grid 80 points not 1000
- Current VaR diagnostic is normal-parametric, not the paper's filtered historical simulation
- MCS not implemented

**Must not claim:**
- That NN results replicate the paper (direction reversed: codebase NN > HAR, paper NN < HAR)
- That PARTIAL_MALL is equivalent to MALL
- That fixed scheme results are directly comparable to paper's rolling scheme results
- That the existing `var_backtest_summary.csv` reproduces the paper's VaR application

### After planned rerun completes (outputs_final/ + outputs_ale/) — compute-budget plan deliverables

**Additional claims unlocked after Task A0 + A + B + C + D + E + F + G + H + I:**
- Non-NN models use the paper-aligned rolling window for both **h=1 and h=5**
  (Tasks A0/A). HAR-family and regularized linear models refit at each test date;
  tree models launch with `ml_refit_every=5` as a compute-budget approximation unless
  the optional daily-refit upgrade succeeds. The paper's "gains increase with horizon"
  finding (Abstract item 4, paper Section 4) is verifiable at h=5.
- NN uses 30 seeds with top-10 ensemble (NN^10_k); paper uses 100 seeds. Paper's own
  Figure A.3 (p.1722) demonstrates ensemble performance saturation by N≈10-20 seeds,
  so the gap to the 100-seed paper setup is estimated at 0.5%-2% relative MSE — much
  smaller than other deviations such as data source and stock count (Task B).
- Both NN^10 ensemble AND NN^1 single-best-seed reported for h=1 and h=5, matching
  the paper's Tables 2/3 and Tables 4/5 column structure (Task D).
- Apple-only ALE plots produced for the five reference models × **three** reference
  features (RVD, RVW, M1W), matching the paper's asset-level Figure 6 treatment except
  for the omitted IV row (Task E). Implementation uses paper-specified `grid_size=100`
  quantile-based partition.
- The paper's selected RV-decile MSE comparison is reproduced for PARTIAL_MALL across
  selected low, medium, and high deciles for the paper-selected reference models
  (HAR-X, LogHAR, EN, RF, NN^10_2). Both h=1 and h=5 covered (Task F).
- Partial Figure 7-style variable importance is produced using the paper's ALE-based
  VI formula (paper eqs. 30-31) for HAR-X, EN, RF, and NN^10_2 on PARTIAL_MALL features
  (Task G). IV is omitted, and VIX is not used as a substitute.
- Paper-style filtered historical simulation VaR diagnostics are computed, separate from
  the older normal-parametric diagnostic (Task H).
- Paper Figure 4 Model Confidence Set is reproduced at 90% confidence using
  `arch.bootstrap.MCS` (Hansen-Lunde-Nason 2011 with stationary block bootstrap),
  reporting per-model inclusion rates across the 25 tickers for both datasets and
  both horizons (Task I).
- HARQ uses the paper's interaction term `sqrt(RQ) × RVD` (paper eq. 10); verified
  in `src/rv1rep/features.py:109-110` as `sqrt_rq_x_rvd`.
- LevHAR uses the paper's negative aggregated returns `min(0, mean(r))` (paper eqs. 7-8);
  verified in `src/rv1rep/features.py:32-34` (columns `rd`, `rw`, `rm`).

**Remaining acknowledged deviations after the planned rerun:**
- 25 stocks not 29 (CVX, TRV, RTX, DOW missing due to AV data limitation)
- AV panel not strictly balanced; per-ticker test sample varies 801–842 obs
- IV omitted from MALL (PARTIAL_MALL); ALE's IV row also omitted (VIX is not a valid
  substitute for stock-level IV)
- **h=22 NOT computed** (constrained compute budget). Paper shows largest ML gains at h=22;
  our h=5 results therefore serve as a **conservative lower bound** on ML's long-horizon
  advantage rather than a full replication of that finding.
- **30 NN seeds not 100** (constrained compute budget). Paper Figure A.3 supports the
  trade-off: ensemble saturates by N≈10-20; expected gap is 0.5%-2% relative MSE.
- Alpha grid 80 log-spaced points vs paper's 1000. Reproduction stance: treat the
  80-point grid as acceptable only if the alpha boundary stability check (see Section 11)
  confirms that <1% of optimal alpha selections hit the grid boundaries. If the check
  fails, widen the alpha range and rerun only the affected regularized model(s).
  Expanding directly to 1000 points would cost ~2-5 days for <0.5% expected MSE gain.
- Tree-model `ml_refit_every`: launched at 5 under the compute budget; if the optional tree-model upgrade
  to `=1` (paper-method daily refit) succeeded, this deviation is fully eliminated for
  Bagging / RandomForest / GradientBoosting. Report wording adapts to upgrade outcome
  (see Task A — optional tree-model upgrade subsection).
- NN hyperparameter cross-check (epochs, initializer) is documentary only; the code uses
  reasonable defaults consistent with the paper's spirit but specific values
  (epochs=500, Glorot normal) are not asserted in this rerun.

**Must not claim:**
- That NN results replicate the paper's exact magnitudes (data source and stock count
  gaps dominate; expect directional consistency at best)
- That PARTIAL_MALL is equivalent to MALL
- Replication of the paper's h=22 monthly-horizon results

---

## 13. Compute-Budget Plan: Final Summary Table

| Dimension | Status under compute-budget plan | Paper alignment |
|---|---|---|
| Core paper structure (HAR, ML, ALE, VaR, MCS) | Planned via Tasks A-I | Full if all tasks complete |
| Multi-horizon: **h=5** | Planned in Tasks A/B | Demonstrates "gains increase with horizon" |
| Multi-horizon: **h=22** | ❌ Skipped (budget) | Paper's strongest result omitted; documented in Section 12 |
| MCS (Figure 4) | Planned Task I via `arch.bootstrap.MCS` | Matches paper Section 1.6 / Figure 4 (HLN 2011 with stationary block bootstrap) |
| HARQ feature construction (`sqrt(RQ)×RVD`) | ✅ Verified correct in `features.py:109-110` | Matches paper eq. 10 |
| LevHAR feature construction (`min(0, mean(r))`) | ✅ Verified correct in `features.py:32-34` | Matches paper eqs. 7-8 |
| `ml_refit_every` (tree models) | 🟡 Launch-time: 5 (approximation); optional post-Task-A upgrade to 1 | Paper uses 1 (daily refit); upgrade path via new output dir |
| `alpha_grid_size` | 🟡 80 + boundary stability check required | Paper uses 1000; accept 80 only if boundary check passes |
| `alpha_min` / `alpha_max` | 🟡 [1e-5, 1e2]; widen if boundary check fails | Paper text only says "broad enough" |
| `elastic_l1_ratios` | ✅ 7-point grid `[0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]` | Paper unspecified; spans LA→RR |
| ALE implementation (per-stock Apple, grid=100) | Planned Task E | Matches paper Figure 6 except omitted IV row |
| VI method (ALE-based, paper eqs. 30-31) | Planned Task G | Matches paper Section 3.2 except omitted IV |
| VaR method (Filtered Historical Simulation) | Planned Task H | Matches paper Section 5 |
| NN seeds: **30** (paper 100) | 🟡 Rational trade-off | Paper Figure A.3 supports saturation by N≈10-20 |
| NN^1 single-best-seed (Task D) | Planned Task D, ~1 hour post-processing | Matches paper Tables 2/3 column structure |
| NN hyperparameters (epochs, initializer) | 🟡 Documentary only | Not strictly verified |
| Data/sample (25 vs 29 stocks, no IV) | 🟡 Permanent gap | Acknowledged in Section 12 |
| Scheme (rolling for non-NN, fixed for NN) | ✅ Matches paper after Phase 0/A/B complete | Task A0 must be completed before Task A |
| Checkpoint robustness | Available by design; verify with Gate 5 before launch | Enables compute-budget plan only if reuse smoke test passes |

### Locked pre-launch configuration

```yaml
estimation:
  scheme: fixed                    # CLI override to rolling for Task A
  ml_refit_every: 5                # optional tree-model upgrade to 1 via new --output-dir if possible

models:
  regularization:
    alpha_grid_size: 80            # backed by boundary stability check
    alpha_min: 1.0e-5
    alpha_max: 1.0e2
    elastic_l1_ratios: [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
  neural_network:
    seeds: 30                      # incremental upgrade to 50 allowed
    ensemble_top: 10

feature_engineering:
  horizons: [1, 5]
  horizon_target_mode: future_average

experiments:
  horizons: [1, 5]
```

### Why these and not alternatives

| Considered | Decision | Reason |
|---|---|---|
| `alpha_grid_size: 200` | ❌ rejected | +1 day cost for <0.5% MSE gain; saved time better spent on tree daily refit or NN seeds |
| `alpha_grid_size: 1000` (paper) | ❌ rejected | +2-5 days; same as above, even worse trade-off |
| `ml_refit_every: 1` as launch-time default | ❌ rejected | Risks budget overrun if tree-model rolling slower than expected; staged as optional post-Task-A upgrade |
| `ml_refit_every: 20` (current default) | ❌ rejected | Too coarse; ~5× faster than =5 but misses important rolling-window refit fidelity |
| `horizons: [1, 5, 22]` | ❌ rejected | +~2 days for h=22 NN training; h=5 alone demonstrates the long-horizon finding |
| `seeds: 50` as launch-time default | 🟡 deferred | Optional incremental upgrade during Task B once 30 seeds complete and budget allows; 30 → 50 is zero-waste (existing 30 seeds reused) |
| `seeds: 100` (paper) | ❌ rejected | +~5 days NN training; saturation already by N≈10-20 per paper Figure A.3 |

### Final deliverables after all phases complete

- 22 models × 2 horizons (h=1, h=5) × 2 datasets (MHAR, PARTIAL_MALL) = 88 model-horizon-dataset cells
- Pairwise relative MSE matrices and DM test results
- Figure 4 (per-model MCS inclusion rates, 90% confidence) for both datasets × both horizons
- Figure 5 (RV-decile MSE) for both horizons
- Figure 6 (per-stock ALE for Apple)
- Figure 7 (cross-sectional ALE-based VI)
- FHS VaR backtest (Kupiec + Christoffersen) for both horizons
- Alpha boundary stability check report (justifies grid_size=80)
- Section 12 honest enumeration of gaps to paper

### Critical pre-launch gating

Both must be true before declaring "training launched":

1. Phase 0 Gates 0-5 all passed (including Task A0 and checkpoint reuse smoke test)
2. Task A and Task B (smoke test, 5 seeds) launched in screen and verified producing non-zero output

If either is not true, do NOT advance to Phase 2 (scale Task B to 30 seeds) or to any
downstream task. The single largest risk to this plan is launching training without
having validated checkpoint reuse — if reuse silently fails, all subsequent restarts
retrain from scratch and the budget collapses.
