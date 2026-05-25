# Reproduction Extended Plan
# Optional Tasks Beyond the Core 3-Day Reproduction

> This document defines **optional tasks** that extend the core reproduction defined in
> `REPRODUCTION_GAP_ANALYSIS.md`. These tasks are designed so that **none of them are
> required for the 3-day core deliverable**. Each task can be skipped or interrupted
> without invalidating the results produced by the core plan.
>
> **Read this AFTER `REPRODUCTION_GAP_ANALYSIS.md`.** That document is the source of truth
> for Tasks A through G and the core h=1 results. This document only adds Task H/I/J/K.

---

## 0. How To Use This Document

This document is structured so an AI executor can run tasks one at a time, in dependency
order, and stop at any point.

**Universal rules:**
1. Do not run any task here until the corresponding prerequisites in
   `REPRODUCTION_GAP_ANALYSIS.md` are complete.
2. Each task writes to its own isolated output directory. Do not overwrite
   `outputs/`, `outputs_full_nn/`, `outputs_rolling/`, `outputs_nn100_checkpointed/`,
   or `outputs_final/`. These are protected by the core plan.
3. Each task has a **completion contract**: a validation snippet that proves the task
   succeeded. Do not declare a task done if its validation fails.
4. If a task fails or is interrupted, the next executor should be able to resume from
   the documented checkpoint structure. Do not retry failures by re-running unrelated
   tasks.

**Priority order (run in this sequence if budget permits):**

| Order | Task | Prerequisite | Reason for ordering |
|---|---|---|---|
| 1 | Task H (MCS, Figure 4) | Core Task C | Independent algorithm, ~6h implementation + ~2h run |
| 2 | Task I + J + K (h=5 multi-horizon) | Core Tasks A and B complete | Largest budget item, ~43 hours of compute |

Tasks F (Figure 5 RV-decile MSE) and G (Figure 7 variable importance) used to live in
this document but were promoted into the core plan because their compute cost is small
relative to their reproduction value. They are defined in `REPRODUCTION_GAP_ANALYSIS.md`
Section 11.

---

## 1. Task H — Model Confidence Set (MCS, Figure 4)

### Purpose
Reproduce Figure 4 of the paper: compute Hansen-Lunde-Nason (2011) Model Confidence Set
at 90% confidence per ticker, then report the percentage of tickers for which each model
was retained in the MCS.

### Prerequisites
- Core Task C complete (`outputs_final/predictions/model_predictions.csv` exists).

### Module to create
`src/rv1rep/mcs.py` — implementation of HLN (2011) algorithm.

### Script to create
`scripts/06d_compute_mcs.py`

### Algorithm contract (must implement)

The MCS procedure tests the null hypothesis of equal predictive ability among a candidate
set of models, iteratively eliminating the worst until no rejection can be made. The
reference is Hansen, P. R., Lunde, A., & Nason, J. M. (2011), "The Model Confidence Set,"
Econometrica, 79(2), 453-497.

Implementation requirements:

| Aspect | Requirement |
|---|---|
| Loss function | Squared error: `(forecast_rv - actual_rv)^2` |
| Test statistic | T_max (range statistic of standardized loss differentials) |
| Variance estimation | Block bootstrap with block length = ceil(min(20, sqrt(T))) where T is test sample size per ticker |
| Bootstrap reps | 5000 |
| Confidence level | 0.90 (90% MCS) |
| Per-ticker MCS | Compute one MCS per ticker per (dataset, horizon), pooling across the ticker's test dates |
| Multiple comparisons | Apply Bonferroni-style elimination tournament as in HLN section 3 |
| Random seed | `cfg['project']['random_seed']` for bootstrap reproducibility |

### Unit tests required

The MCS implementation is error-prone. Before declaring Task H complete, the executor
MUST add and pass unit tests in `tests/test_mcs.py`:

```python
# Test 1: Three identical-loss models → all retained at 90%
# Test 2: One dominant model + two equally-bad clones → MCS = {dominant, possibly one clone}
# Test 3: Compare against the `arch` Python package's `MCS` class on a small simulated
#         dataset (if `arch` is installable). If MCS retention rates differ by more than
#         10 percentage points on a 5-model toy example, raise.
```

### Script contract

| Aspect | Requirement |
|---|---|
| Input | `outputs_final/predictions/model_predictions.csv` |
| Per-(dataset, horizon, ticker) loop | Run MCS on the 22-model loss matrix |
| Output table | `outputs_extended/mcs_results.csv` with columns `dataset, horizon, ticker, model, in_mcs (bool), elimination_order, p_value` |
| Aggregate table | `outputs_extended/mcs_retention_rates.csv` with columns `dataset, horizon, model, retention_pct (0-100), n_tickers` |
| Output figure | `outputs_extended/figure4_mcs.png` — horizontal bar chart per dataset, sorted by retention_pct |
| Idempotency | Overwrite each run; no checkpointing |
| Failure modes | If a ticker has fewer than 200 test obs, skip and log. If bootstrap fails to converge for any ticker, log warning and mark that ticker as `mcs_failed=True` in the per-ticker table. |

### Run command
```bash
cd "/Users/celiawong/Desktop/Course004 machine learning/004_individual/rv1_professional_reproduction_code"
mkdir -p outputs_extended/logs
~/.pyenv/versions/3.11.8/bin/python -m pytest tests/test_mcs.py -v \
  > outputs_extended/logs/06d_mcs_tests.log 2>&1
~/.pyenv/versions/3.11.8/bin/python scripts/06d_compute_mcs.py \
  --predictions outputs_final/predictions/model_predictions.csv \
  --config config/default.yaml \
  --output-dir outputs_extended \
  > outputs_extended/logs/06d_mcs.log 2>&1
```

### Validation
```bash
~/.pyenv/versions/3.11.8/bin/python -c "
import os
import pandas as pd
per_ticker = pd.read_csv('outputs_extended/mcs_results.csv')
agg = pd.read_csv('outputs_extended/mcs_retention_rates.csv')
expected_models = {
    'HAR', 'HARX', 'LogHAR', 'LevHAR', 'SHAR', 'HARQ',
    'Ridge', 'Lasso', 'ElasticNet', 'AdaptiveLasso', 'PostLasso',
    'Bagging', 'RandomForest', 'GradientBoosting',
    'NN1', 'NN2', 'NN3', 'NN4',
    'NN1_1', 'NN2_1', 'NN3_1', 'NN4_1',
}
agg_models = set(agg['model'])
if agg_models != expected_models:
    raise SystemExit(f'Unexpected MCS model set: missing={expected_models - agg_models}, extra={agg_models - expected_models}')
if (agg['retention_pct'] < 0).any() or (agg['retention_pct'] > 100).any():
    raise SystemExit('retention_pct out of [0, 100]')
if (agg['n_tickers'] < 10).any():
    raise SystemExit('Some models have fewer than 10 contributing tickers')
fig = 'outputs_extended/figure4_mcs.png'
if not os.path.exists(fig):
    raise SystemExit(f'Missing figure: {fig}')
print('Task H OK:', len(per_ticker), 'per-ticker rows;', len(agg), 'aggregate rows')
"
```

---

## 2. Task I + J + K — Multi-horizon Forecasts (h=5)

### Scope

The paper reports h=1, h=5, h=22 in Tables 2/3/4/5. The core plan covers h=1 only. This
extended plan adds **h=5 only**. h=22 is intentionally excluded because it doubles the
compute budget without proportionally more reproduction value.

### Prerequisites
- Core Task A/B complete (so the h=1 results are preserved in `outputs_rolling/` and
  `outputs_nn100_checkpointed/`).
- This extension does NOT overwrite those directories. It writes to new directories
  `outputs_rolling_h5/` and `outputs_nn100_checkpointed_h5/`.

### One-time preparation

#### Step 1: Modify config

Open `config/default.yaml` and add h=5 to **both** horizon lists. The codebase has two
separate horizon settings that must be kept consistent (see Section 8.2 of
`REPRODUCTION_GAP_ANALYSIS.md`):

```yaml
feature_engineering:
  horizons: [1, 5]

experiments:
  horizons: [1, 5]
```

Keep `horizon_target_mode: future_average` unchanged.

#### Step 2: Rebuild forecasting panel

The current panel has only `target_rv_h1` and `target_log_rv_h1` columns. Rebuild it to
materialize `target_rv_h5` and `target_log_rv_h5`:

```bash
cd "/Users/celiawong/Desktop/Course004 machine learning/004_individual/rv1_professional_reproduction_code"
~/.pyenv/versions/3.11.8/bin/python scripts/03_build_features.py \
  --config config/default.yaml \
  > data/processed/03_build_features_h5.log 2>&1
```

Verify the rebuilt panel:
```bash
~/.pyenv/versions/3.11.8/bin/python -c "
import pandas as pd
p = pd.read_csv('data/processed/forecasting_panel.csv', nrows=5)
required = {'target_rv_h1', 'target_log_rv_h1', 'target_rv_h5', 'target_log_rv_h5'}
missing = required - set(p.columns)
if missing:
    raise SystemExit(f'Panel rebuild incomplete: missing columns {missing}')
print('Panel rebuilt OK, has h=1 and h=5 targets')
"
```

#### Step 3: NN runner horizon support (already in place)

`scripts/04_run_nn_checkpoints.py` already supports `--horizons` (plural; takes one or
more integers) and writes per-seed checkpoints into a horizon-aware directory layout
`nn_seed_predictions/<dataset>/h{horizon}/<model>/<ticker>/seed_*.csv`. No script patch
is needed for h=5. The default value of `--horizons` is `experiments.horizons` from the
config file, so after Step 1 sets `experiments.horizons: [1, 5]`, omitting `--horizons`
on the CLI will run both. Pass `--horizons 5` explicitly to run h=5 only and leave the
existing h=1 checkpoints untouched.

### Task I — h=5 non-NN models, rolling scheme

```bash
cd "/Users/celiawong/Desktop/Course004 machine learning/004_individual/rv1_professional_reproduction_code"
mkdir -p outputs_rolling_h5/logs
screen -dmS rv1_rolling_h5 bash -lc '
~/.pyenv/versions/3.11.8/bin/python scripts/04_run_forecasts_checkpoints.py \
  --config config/default.yaml \
  --output-dir outputs_rolling_h5 \
  --scheme rolling \
  --horizons 5 \
  --skip-nn \
  --allow-existing-output-dir \
  > outputs_rolling_h5/logs/04_forecasts_rolling_h5.log 2>&1
echo $? > outputs_rolling_h5/logs/04_forecasts_rolling_h5.exitcode
'
```

Note: `04_run_forecasts_checkpoints.py` already accepts `--horizons`, so no patch needed.

### Task J — h=5 NN models, fixed scheme, 100 seeds

```bash
cd "/Users/celiawong/Desktop/Course004 machine learning/004_individual/rv1_professional_reproduction_code"
mkdir -p outputs_nn100_checkpointed_h5/logs
screen -dmS rv1_nn100_h5 bash -lc '
~/.pyenv/versions/3.11.8/bin/python scripts/04_run_nn_checkpoints.py \
  --config config/default.yaml \
  --output-dir outputs_nn100_checkpointed_h5 \
  --datasets MHAR PARTIAL_MALL \
  --models NN1 NN2 NN3 NN4 \
  --horizons 5 \
  --seed-count 100 \
  --ensemble-top 10 \
  --base-predictions "" \
  --allow-existing-output-dir \
  > outputs_nn100_checkpointed_h5/logs/04_nn100_h5.log 2>&1
echo $? > outputs_nn100_checkpointed_h5/logs/04_nn100_h5.exitcode
'
```

Note: `--horizons` is plural and takes one or more integers. Passing `--horizons 5`
runs h=5 only; omitting the flag would default to `experiments.horizons` from the
config and may include h=1 redundantly.

### Task K — h=5 NN^1 extraction + merge + evaluate

After Task I and Task J complete:

```bash
cd "/Users/celiawong/Desktop/Course004 machine learning/004_individual/rv1_professional_reproduction_code"

# K.1 NN^1 single-seed extraction for h=5 (mirrors core Task D)
~/.pyenv/versions/3.11.8/bin/python scripts/06_extract_nn_best_single_seed.py \
  --config config/default.yaml \
  --output-dir outputs_nn100_checkpointed_h5 \
  > outputs_nn100_checkpointed_h5/logs/06_extract_nn1_h5.log 2>&1

# K.2 Merge h=5 results
mkdir -p outputs_final_h5/predictions outputs_final_h5/logs outputs_final_h5/tables outputs_final_h5/figures

~/.pyenv/versions/3.11.8/bin/python -c "
import pandas as pd
rolling = pd.read_csv('outputs_rolling_h5/predictions/model_predictions.csv')
nn = pd.read_csv('outputs_nn100_checkpointed_h5/predictions/nn_model_predictions.csv')
nn1 = pd.read_csv('outputs_nn100_checkpointed_h5/predictions/nn1_model_predictions.csv')
combined = pd.concat([rolling, nn, nn1], ignore_index=True)
dup = combined.duplicated(['date', 'ticker', 'dataset', 'horizon', 'model']).sum()
if dup:
    raise RuntimeError(f'Duplicate prediction keys: {dup}')
horizons = sorted(combined['horizon'].unique())
if set(horizons) != {5}:
    raise RuntimeError(f'Unexpected horizons in h=5 merge: {horizons}')
combined.to_csv('outputs_final_h5/predictions/model_predictions.csv', index=False)
print('h=5 combined rows:', len(combined), 'models:', sorted(combined['model'].unique()))
"

# K.3 Evaluate h=5
~/.pyenv/versions/3.11.8/bin/python scripts/05_evaluate_outputs_isolated.py \
  --config config/default.yaml \
  --output-dir outputs_final_h5
```

### Validation for I + J + K
```bash
~/.pyenv/versions/3.11.8/bin/python -c "
import pandas as pd
df = pd.read_csv('outputs_final_h5/predictions/model_predictions.csv')
expected_models = {
    'HAR', 'HARX', 'LogHAR', 'LevHAR', 'SHAR', 'HARQ',
    'Ridge', 'Lasso', 'ElasticNet', 'AdaptiveLasso', 'PostLasso',
    'Bagging', 'RandomForest', 'GradientBoosting',
    'NN1', 'NN2', 'NN3', 'NN4',
    'NN1_1', 'NN2_1', 'NN3_1', 'NN4_1',
}
models = set(df['model'])
if models != expected_models:
    raise SystemExit(f'Missing h=5 models: {expected_models - models}')
horizons = set(df['horizon'])
if horizons != {5}:
    raise SystemExit(f'h=5 file contains unexpected horizons: {horizons}')
non_nn = df[~df['model'].str.startswith('NN')]
nn_all = df[df['model'].str.startswith('NN')]
if set(non_nn['scheme']) != {'rolling'}:
    raise SystemExit(f'h=5 non-NN should be rolling: got {set(non_nn[\"scheme\"])}')
if set(nn_all['scheme']) != {'fixed'}:
    raise SystemExit(f'h=5 NN should be fixed: got {set(nn_all[\"scheme\"])}')
print('h=5 final OK:', len(df), 'rows;', df['model'].nunique(), 'models')
"
```

---

## 3. After All Extended Tasks Complete

### Final report claim additions (beyond core 3-day claims)

**Unlocked claims (beyond what Tasks A through G already deliver):**
- Figure 4 reproduced: MCS retention rates at 90% confidence per ticker, aggregated
- h=5 results (Table 4 of the paper) reproduced with 22 models matching the core
  h=1 structure

**Still acknowledged as not done (even with one-week budget):**
- h=22 (monthly horizon) not computed — explicitly out of scope to preserve budget
- Robustness check `fixed_train_days=1000, 2000` (paper Appendix A.1) not run
- Tree-based models still use `ml_refit_every=20`, not true daily refit
- IV (OptionMetrics) still unavailable; impacts MALL completeness and ALE
- 25 stocks not 29
- Alpha grid 80 points not 1000

### File layout summary after full extended completion

```
outputs_final/                            # Core h=1 results (Tasks C, F, G)
├── predictions/model_predictions.csv     # 22 models
├── tables/
│   ├── (forecast_summary, DM, VaR, pairwise, ...)
│   ├── rv_decile_mse.csv                 # Task F
│   └── variable_importance.csv           # Task G
└── figures/
    ├── (relative_mse, ...)
    ├── figure5_rv_decile_mse.png         # Task F
    └── figure7_variable_importance.png   # Task G
outputs_ale/                              # Core Task E
└── figure6_ale.png                       # 3×5 ALE grid
outputs_extended/                         # Extended Task H (MCS)
├── mcs_results.csv
├── mcs_retention_rates.csv
└── figure4_mcs.png
outputs_rolling_h5/                       # Extended Task I
outputs_nn100_checkpointed_h5/            # Extended Task J + K.1
outputs_final_h5/                         # Extended Task K.2 + K.3
└── predictions/model_predictions.csv     # 22 models at h=5
```

---

## 4. Stopping Rules

If you are an AI executing this plan and time is running out:

1. **Finish whatever task is currently running.** Do not abort mid-task; the checkpoint
   structure means an aborted task wastes work.
2. **Skip remaining tasks in this document.** They are explicitly optional.
3. **Report what completed.** The validation snippets above produce structured output
   suitable for a status report.
4. **Do NOT touch `outputs_final/` or any core directory.** If only some extended tasks
   completed, the core 3-day results are still the canonical report basis.
