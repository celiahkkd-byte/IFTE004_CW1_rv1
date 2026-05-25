# GitHub Upload Guide

Repository target:

`https://github.com/celiahkkd-byte/IFTE004_CW1_rv1.git`

## What To Upload

The repository should contain:

- source code: `src/`, `scripts/`, `main.py`
- configuration: `config/`
- reproducibility notes: root-level `.md` files and `docs/`
- Python environment file: `requirements.txt`
- processed modelling data: selected files in `data/processed/`
- external-data documentation and small external files in `data/external/`
- lightweight result tables: `results_release/`, including the 25-ticker
  mainline `h=1`, `h=5`, and `h=22` result tables

## What Not To Upload

The following are intentionally excluded by `.gitignore`:

- cache folders: `__pycache__/`, `.pytest_cache/`, `.matplotlib-cache/`
- logs and generated Word/PDF/image files
- large `outputs*/` directories
- full prediction tables such as `model_predictions.csv`
- NN seed checkpoints
- raw Alpha Vantage intraday data

## Recommended Upload Commands

From the project root:

```bash
git init
git branch -M main
git remote add origin https://github.com/celiahkkd-byte/IFTE004_CW1_rv1.git
git add .gitignore README.md requirements.txt main.py config src scripts docs data results_release *.md
git status --short
git commit -m "Add reproducible RV forecasting replication code and result tables"
git push -u origin main
```

Before committing, check that no file above 100MB is staged:

```bash
git ls-files -s | awk '{print $4}' | xargs -I{} sh -c 'test -f "{}" && du -m "{}"' | sort -nr | head
```

If GitHub rejects the push because the remote already has commits, use:

```bash
git pull --rebase origin main
git push -u origin main
```

Do not use `git add .` unless `.gitignore` has been checked first.
