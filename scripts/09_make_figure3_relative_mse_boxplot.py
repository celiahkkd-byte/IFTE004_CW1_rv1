from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/rv1rep_matplotlib')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PANEL_MODEL_ORDER = {
    'MHAR': [
        'LogHAR',
        'LevHAR',
        'SHAR',
        'HARQ',
        'Ridge',
        'Lasso',
        'ElasticNet',
        'AdaptiveLasso',
        'PostLasso',
        'Bagging',
        'RandomForest',
        'GradientBoosting',
        'NN1_1',
        'NN1',
        'NN2_1',
        'NN2',
        'NN3_1',
        'NN3',
        'NN4_1',
        'NN4',
    ],
    'PARTIAL_MALL': [
        'HARX',
        'LogHAR',
        'LevHAR',
        'SHAR',
        'HARQ',
        'Ridge',
        'Lasso',
        'ElasticNet',
        'AdaptiveLasso',
        'PostLasso',
        'Bagging',
        'RandomForest',
        'GradientBoosting',
        'NN1_1',
        'NN1',
        'NN2_1',
        'NN2',
        'NN3_1',
        'NN3',
        'NN4_1',
        'NN4',
    ],
}


LABELS = {
    'HARX': 'HAR-X',
    'LogHAR': 'LogHAR',
    'LevHAR': 'LevHAR',
    'SHAR': 'SHAR',
    'HARQ': 'HARQ',
    'Ridge': 'RR',
    'Lasso': 'LA',
    'ElasticNet': 'EN',
    'AdaptiveLasso': 'A-LA',
    'PostLasso': 'P-LA',
    'Bagging': 'BG',
    'RandomForest': 'RF',
    'GradientBoosting': 'GB',
    'NN1_1': r'NN$^{1}_{1}$',
    'NN1': r'NN$^{10}_{1}$',
    'NN2_1': r'NN$^{1}_{2}$',
    'NN2': r'NN$^{10}_{2}$',
    'NN3_1': r'NN$^{1}_{3}$',
    'NN3': r'NN$^{10}_{3}$',
    'NN4_1': r'NN$^{1}_{4}$',
    'NN4': r'NN$^{10}_{4}$',
}


PANEL_TITLES = {
    'MHAR': r'(A) $\mathcal{M}_{\mathrm{HAR}}$',
    'PARTIAL_MALL': r'(B) $\mathcal{M}_{\mathrm{PARTIAL\_MALL}}$ (IV omitted)',
}


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_relative_mse(metrics_path: Path, horizon: int, datasets: list[str]) -> pd.DataFrame:
    if not metrics_path.exists():
        raise FileNotFoundError(f'Missing forecast metrics file: {metrics_path}')
    metrics = pd.read_csv(metrics_path)
    required = {'dataset', 'horizon', 'ticker', 'model', 'mse', 'har_mse', 'relative_mse_vs_har'}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f'{metrics_path} missing required columns: {sorted(missing)}')
    metrics['horizon'] = metrics['horizon'].astype(int)
    out = metrics[(metrics['horizon'] == int(horizon)) & (metrics['dataset'].isin(datasets))].copy()
    out['relative_mse_vs_har'] = pd.to_numeric(out['relative_mse_vs_har'], errors='coerce')
    out = out.dropna(subset=['relative_mse_vs_har'])
    return out


def validate_panel_data(df: pd.DataFrame, datasets: list[str]) -> None:
    for dataset in datasets:
        models = PANEL_MODEL_ORDER[dataset]
        sub = df[df['dataset'] == dataset]
        missing = [m for m in models if m not in set(sub['model'])]
        if missing:
            raise ValueError(f'Missing Figure 3 models for dataset={dataset}: {missing}')


def write_plot_data(df: pd.DataFrame, output_dir: Path, datasets: list[str]) -> tuple[Path, Path]:
    rows = []
    for dataset in datasets:
        sub = df[df['dataset'] == dataset]
        for model in PANEL_MODEL_ORDER[dataset]:
            g = sub[sub['model'] == model].copy()
            for _, r in g.iterrows():
                rows.append(
                    {
                        'dataset': dataset,
                        'horizon': int(r['horizon']),
                        'ticker': r['ticker'],
                        'model': model,
                        'model_label': LABELS.get(model, model),
                        'relative_mse_vs_har': float(r['relative_mse_vs_har']),
                    }
                )
    plot_df = pd.DataFrame(rows)
    plot_path = output_dir / 'figure3_relative_mse_by_ticker.csv'
    plot_df.to_csv(plot_path, index=False)

    summary = (
        plot_df.groupby(['dataset', 'horizon', 'model', 'model_label'])['relative_mse_vs_har']
        .agg(n='count', mean='mean', median='median', minimum='min', q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75), maximum='max')
        .reset_index()
    )
    summary_path = output_dir / 'figure3_relative_mse_summary.csv'
    summary.to_csv(summary_path, index=False)
    return plot_path, summary_path


def clipped_counts(df: pd.DataFrame, datasets: list[str], y_min: float, y_max: float) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        sub = df[df['dataset'] == dataset]
        for model in PANEL_MODEL_ORDER[dataset]:
            x = sub.loc[sub['model'] == model, 'relative_mse_vs_har']
            rows.append(
                {
                    'dataset': dataset,
                    'model': model,
                    'n': int(x.shape[0]),
                    'below_y_min': int((x < y_min).sum()),
                    'above_y_max': int((x > y_max).sum()),
                    'max_relative_mse': float(x.max()),
                }
            )
    return pd.DataFrame(rows)


def draw_figure(
    df: pd.DataFrame,
    output_path: Path,
    datasets: list[str],
    *,
    y_min: float | None,
    y_max: float | None,
    title_suffix: str = '',
) -> None:
    plt.rcParams.update(
        {
            'font.family': 'DejaVu Sans',
            'axes.edgecolor': '#666666',
            'axes.linewidth': 0.8,
        }
    )
    fig, axes = plt.subplots(1, len(datasets), figsize=(14.5, 6.2), sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        sub = df[df['dataset'] == dataset]
        models = PANEL_MODEL_ORDER[dataset]
        data = [sub.loc[sub['model'] == model, 'relative_mse_vs_har'].to_numpy(dtype=float) for model in models]
        labels = [LABELS.get(model, model) for model in models]
        positions = np.arange(1, len(models) + 1)
        ax.boxplot(
            data,
            positions=positions,
            widths=0.55,
            patch_artist=False,
            whis=1.5,
            medianprops={'color': 'red', 'linewidth': 1.7},
            boxprops={'color': 'blue', 'linewidth': 1.7},
            whiskerprops={'color': 'black', 'linewidth': 1.2, 'linestyle': '--'},
            capprops={'color': 'black', 'linewidth': 1.2},
            flierprops={
                'marker': 'o',
                'markerfacecolor': 'none',
                'markeredgecolor': 'magenta',
                'markersize': 7,
                'linestyle': 'none',
                'markeredgewidth': 1.3,
            },
        )
        ax.axhline(1.0, color='#9a9a9a', linewidth=1.2, linestyle=(0, (4, 4)))
        ax.set_title(PANEL_TITLES[dataset], loc='left', fontsize=14, fontweight='bold', pad=8)
        ax.set_ylabel('relative mse', fontsize=11)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=9)
        ax.tick_params(axis='both', direction='in', top=True, right=True)
        ax.grid(True, axis='both', color='#e2e2e2', linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        if y_min is not None and y_max is not None:
            ax.set_ylim(y_min, y_max)

    fig.suptitle(f'Figure 3 Boxplot of cross-sectional out-of-sample relative MSE{title_suffix}', y=0.98, fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Create paper-style Figure 3 relative-MSE boxplots.')
    parser.add_argument(
        '--metrics',
        default='outputs_final_core_with_bagging_gb_harfix_nn50_tuned_no_refit_h1h5_20260523/tables/forecast_metrics_by_asset.csv',
    )
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--datasets', nargs='+', default=['MHAR', 'PARTIAL_MALL'])
    parser.add_argument('--paper-y-min', type=float, default=0.75)
    parser.add_argument('--paper-y-max', type=float, default=1.75)
    args = parser.parse_args()

    metrics_path = _resolve(args.metrics)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [str(d) for d in args.datasets]

    rel = load_relative_mse(metrics_path, args.horizon, datasets)
    validate_panel_data(rel, datasets)
    plot_path, summary_path = write_plot_data(rel, output_dir, datasets)

    clipped = clipped_counts(rel, datasets, float(args.paper_y_min), float(args.paper_y_max))
    clipped_path = output_dir / 'figure3_paper_axis_clipping_audit.csv'
    clipped.to_csv(clipped_path, index=False)

    paper_png = output_dir / 'figure3_relative_mse_boxplot_h1_paper_axis.png'
    full_png = output_dir / 'figure3_relative_mse_boxplot_h1_full_range.png'
    draw_figure(
        rel,
        paper_png,
        datasets,
        y_min=float(args.paper_y_min),
        y_max=float(args.paper_y_max),
        title_suffix=' (h=1, paper-style axis)',
    )
    y_min = max(0.0, float(rel['relative_mse_vs_har'].min()) * 0.95)
    y_max = float(rel['relative_mse_vs_har'].max()) * 1.05
    draw_figure(
        rel,
        full_png,
        datasets,
        y_min=y_min,
        y_max=y_max,
        title_suffix=' (h=1, full range)',
    )

    provenance = {
        'created_at_utc': _utc_now(),
        'metrics': str(metrics_path),
        'horizon': int(args.horizon),
        'datasets': datasets,
        'outputs': {
            'paper_axis_png': str(paper_png),
            'full_range_png': str(full_png),
            'plot_data_csv': str(plot_path),
            'summary_csv': str(summary_path),
            'clipping_audit_csv': str(clipped_path),
        },
        'paper_axis': {'y_min': float(args.paper_y_min), 'y_max': float(args.paper_y_max)},
    }
    (output_dir / 'run_provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding='utf-8')
    print(f'Wrote {paper_png}')
    print(f'Wrote {full_png}')
    print(f'Wrote {plot_path}')
    print(f'Wrote {summary_path}')
    print(f'Wrote {clipped_path}')


if __name__ == '__main__':
    main()
