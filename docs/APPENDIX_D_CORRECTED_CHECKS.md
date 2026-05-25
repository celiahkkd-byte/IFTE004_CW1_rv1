# Appendix D Corrected-Check Reproduction Notes

Appendix D is a controlled robustness check, not the 25-ticker mainline. It is
run on three representative tickers:

```text
AAPL, JPM, MSFT
```

It covers both datasets and all submitted horizons:

```text
datasets: MHAR, PARTIAL_MALL
horizons: h=1, h=5, h=22
```

The corresponding submitted result tables are committed in:

```text
results_release/appendix_d_corrected_h1h5/
results_release/appendix_d_corrected_h22/
```

Large per-seed NN checkpoints and full daily prediction tables are not committed
because they are intermediate outputs. They can be regenerated with the commands
below.

## Corrected Specification

The corrected check applies three changes relative to the mainline NN setting:

- NNs use `Dropout(rate=0.2)`, interpreting the paper's reported dropout value
  of 0.8 as a keep probability.
- The forecast target is standardized within the training sample and predictions
  are inverse-transformed before MSE and DM evaluation.
- EA is kept in its original binary `0/1` scale; continuous predictors are still
  standardized on the training window.

For non-neural models, the target-standardization check is also run so the
comparison is internally consistent. `LogHAR` is the exception: it keeps the
baseline log-target specification and is not additionally standardized.

## Model Design In The Corrected Check

- HAR family: rolling estimation with train+validation combined as the in-sample
  fitting window.
- Regularized linear models: rolling daily refit, train fit plus validation
  tuning, no final train+validation refit.
- Bagging, RandomForest and GradientBoosting: rolling estimation with
  `ml_refit_every=5`.
- NNs: fixed split, 50 seeds, validation seed selection and top-10 ensemble.

## Re-run h=1 And h=5 Corrected Check

Use a fresh output directory:

```bash
OUT=outputs_corrected_paper_style_h1h5_rerun
```

Run NNs:

```bash
python scripts/13_run_corrected_nn_combined.py \
  --config config/paper_core_rolling_tuned_no_refit.yaml \
  --output-dir "$OUT" \
  --datasets MHAR PARTIAL_MALL \
  --horizons 1 5 \
  --tickers AAPL JPM MSFT \
  --seed-count 50 \
  --ensemble-top 10 \
  --workers 3 \
  --allow-existing-output-dir
```

Run non-neural models:

```bash
python scripts/14_run_corrected_nonnn_combined.py \
  --config config/paper_core_rolling_tuned_no_refit.yaml \
  --output-dir "$OUT" \
  --datasets MHAR PARTIAL_MALL \
  --horizons 1 5 \
  --tickers AAPL JPM MSFT \
  --workers 3 \
  --allow-existing-output-dir
```

Build corrected result tables:

```bash
python scripts/15_build_corrected_paper_tables.py \
  --output-dir "$OUT" \
  --datasets MHAR PARTIAL_MALL \
  --horizons 1 5
```

## Re-run h=22 Corrected Check

Use a fresh output directory:

```bash
OUT=outputs_corrected_paper_style_h22_rerun
```

Run NNs:

```bash
python scripts/13_run_corrected_nn_combined.py \
  --config config/paper_core_rolling_tuned_no_refit_h22_panel.yaml \
  --output-dir "$OUT" \
  --datasets MHAR PARTIAL_MALL \
  --horizons 22 \
  --tickers AAPL JPM MSFT \
  --seed-count 50 \
  --ensemble-top 10 \
  --workers 3 \
  --allow-existing-output-dir
```

Run non-neural models:

```bash
python scripts/14_run_corrected_nonnn_combined.py \
  --config config/paper_core_rolling_tuned_no_refit_h22_panel.yaml \
  --output-dir "$OUT" \
  --datasets MHAR PARTIAL_MALL \
  --horizons 22 \
  --tickers AAPL JPM MSFT \
  --workers 3 \
  --allow-existing-output-dir
```

Build corrected result tables:

```bash
python scripts/15_build_corrected_paper_tables.py \
  --output-dir "$OUT" \
  --datasets MHAR PARTIAL_MALL \
  --horizons 22
```

## Provenance Files

Each submitted corrected-check result folder includes provenance JSON files:

```text
run_provenance_nn.json
run_provenance_nonnn.json
run_provenance_evaluation.json
```

These record tickers, horizons, seed range, dropout, target scaling and
standardization policy.
