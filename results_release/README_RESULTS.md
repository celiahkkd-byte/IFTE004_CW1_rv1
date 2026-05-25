# Release Results Manifest

This directory contains lightweight result tables for the submitted reproduction. Large intermediate outputs are intentionally excluded from GitHub, including full daily prediction tables, per-seed NN checkpoints, logs, Word files, and generated figures.

## Mainline Results

`mainline_h1h5h22/` contains the integrated all-ticker mainline evaluation tables for horizons `h=1`, `h=5`, and `h=22`.

Key files:

- `pairwise_relative_mse_matrix.csv`
- `diebold_mariano_tests.csv`
- `forecast_summary_cross_section.csv`
- `forecast_metrics_by_asset.csv`
- `h22_nn_single_best_seed_selection.csv`
- `integration_audit.json`
- `h22_audit_report.json`
- `h22_run_provenance.json`

`mainline_h22_regenerated/` contains the regenerated all-ticker `h=22` evaluation tables used to validate the final monthly-horizon results.

## Appendix D Corrected Checks

`appendix_d_corrected_h1h5/` and `appendix_d_corrected_h22/` contain the three-ticker robustness checks for AAPL, JPM, and MSFT. These checks use:

- `Dropout(rate=0.2)` for NNs, interpreting the reported dropout value as a keep probability.
- training-window target standardisation with inverse transformation before evaluation.
- EA kept in its original binary `0/1` scale.

Only evaluation tables and provenance files are included. Per-seed checkpoints are excluded.

## VaR And Descriptive Tables

`var_h1/` contains one-day-ahead VaR relative check-loss and Diebold-Mariano tables.

`table1/` contains the explanatory-variable summary statistics table and audit file.

## Excluded Files

The following are excluded to keep the repository reproducible and manageable:

- `outputs*/predictions/model_predictions.csv` and other large prediction tables.
- `nn_seed_predictions/` checkpoint folders.
- `logs/`, cache folders, generated Word files, and generated figures.
- raw Alpha Vantage intraday files. The processed forecasting panel is kept under `data/processed/`.
