# H=22 Reproduction Notes

The auxiliary-machine h=22 folder is not required for GitHub review. The
repository keeps the relevant reproducibility artefacts in normal project
locations:

- code: `src/` and `scripts/`
- h=22 configuration: `config/paper_core_rolling_tuned_no_refit_h22_panel.yaml`
- GB 40-grid configuration: `config/paper_core_rolling_gb_tuned_no_refit_40grid.yaml`
- processed panel with h=1, h=5 and h=22 targets: `data/processed/forecasting_panel.csv`
- submitted h=22 result tables: `results_release/mainline_h22_regenerated/`
- integrated h=1/h=5/h=22 result tables: `results_release/mainline_h1h5h22/`
- h=22 corrected three-ticker checks: `results_release/appendix_d_corrected_h22/`

The full daily h=22 prediction files and seed checkpoints are intentionally not
committed because they are large intermediate outputs. The lightweight result
tables and audit files are committed.

## Mainline h=22 Design

The h=22 mainline matches the submitted design:

- 25 tickers.
- datasets: `MHAR` and `PARTIAL_MALL`.
- horizon: `h=22`.
- HAR family: rolling estimation with train+validation combined as the
  in-sample fitting window.
- regularized models: rolling daily refit, train fit plus validation tuning, no
  final train+validation refit.
- Bagging and RandomForest: rolling estimation with train+validation combined,
  refitted every five test observations.
- GradientBoosting: validation-tuned no-refit with the paper-style 40-grid.
- NNs: fixed split, 50 seeds, validation seed selection and top-10 ensemble.

## Submitted h=22 Tables

Use these files for report verification:

```text
results_release/mainline_h22_regenerated/
results_release/mainline_h1h5h22/
```

The pairwise table convention is:

```text
cell(row i, column j) = MSE_j / MSE_i
```

Therefore, to compare any model with HAR, read the `HAR` row.

## Rebuilding h=22 Inputs

The committed `data/processed/forecasting_panel.csv` already contains:

```text
target_rv_h1, target_log_rv_h1,
target_rv_h5, target_log_rv_h5,
target_rv_h22, target_log_rv_h22
```

If the panel needs to be rebuilt, run:

```bash
python scripts/03_build_features.py --config config/paper_core_rolling_tuned_no_refit_h22_panel.yaml
```

## Re-running h=22 Forecasts

Full h=22 model reruns are computationally heavy. Use isolated output
directories and do not write into `outputs/`.

Non-neural h=22 models, including GradientBoosting:

```bash
python scripts/04_run_forecasts_checkpoints.py \
  --config config/paper_core_rolling_tuned_no_refit_h22_panel.yaml \
  --output-dir outputs_h22_rerun_nonnn \
  --horizons 22 \
  --skip-nn \
  --scheme rolling \
  --allow-existing-output-dir
```

Neural networks h=22:

```bash
python scripts/04_run_nn_checkpoints.py \
  --config config/paper_core_rolling_tuned_no_refit_h22_panel.yaml \
  --output-dir outputs_h22_rerun_nn50 \
  --datasets MHAR PARTIAL_MALL \
  --horizons 22 \
  --models NN1 NN2 NN3 NN4 \
  --seed-count 50 \
  --ensemble-top 10 \
  --base-predictions outputs_h22_rerun_nonnn/predictions/model_predictions.csv \
  --allow-existing-output-dir
```

After rebuilding daily prediction files, regenerate evaluation tables with:

```bash
python scripts/05_evaluate_outputs_isolated.py \
  --config config/paper_core_rolling_tuned_no_refit_h22_panel.yaml \
  --output-dir outputs_h22_rerun_nn50
```

To rebuild the h=22 report-facing tables from the regenerated daily prediction
file, run:

```bash
python scripts/09_regenerate_h22_all_ticker_model_tables.py \
  --source-dir outputs_h22_rerun_nn50 \
  --output-dir outputs_h22_all_ticker_model_results_regenerated
```

## Appendix D h=22 Corrected Check

The corrected three-ticker h=22 check is already summarized in:

```text
results_release/appendix_d_corrected_h22/
```

It uses AAPL, JPM and MSFT with `Dropout(rate=0.2)`, training-sample target
standardization with inverse transformation before evaluation, and EA kept as a
binary `0/1` predictor.
