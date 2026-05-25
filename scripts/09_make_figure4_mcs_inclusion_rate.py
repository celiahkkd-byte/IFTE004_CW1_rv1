from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/rv1rep_matplotlib')

import matplotlib.pyplot as plt
import pandas as pd


MODEL_ORDER = [
    'HAR',
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
]

LABELS = {
    'HAR': 'HAR',
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

DATASET_LABELS = {
    'MHAR': r'$\mathcal{M}_{\mathrm{HAR}}$',
    'PARTIAL_MALL': r'$\mathcal{M}_{\mathrm{PARTIAL\_MALL}}$ (IV omitted)',
}


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_inclusion_rates(path: Path, horizon: int, datasets: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Missing MCS inclusion-rate table: {path}')
    df = pd.read_csv(path)
    required = {'dataset', 'horizon', 'model', 'n_tickers_included', 'n_tickers_total', 'inclusion_rate'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'{path} missing required columns: {sorted(missing)}')
    df['horizon'] = df['horizon'].astype(int)
    out = df[(df['horizon'] == int(horizon)) & (df['dataset'].isin(datasets))].copy()
    if out.empty:
        raise ValueError(f'No MCS rows for horizon={horizon} datasets={datasets}')
    return out


def build_plot_data(df: pd.DataFrame, datasets: list[str]) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        for dataset in datasets:
            g = df[(df['dataset'] == dataset) & (df['model'] == model)]
            if g.empty:
                raise ValueError(f'Missing MCS inclusion rate for dataset={dataset} model={model}')
            r = g.iloc[0]
            rows.append(
                {
                    'model': model,
                    'model_label': LABELS.get(model, model),
                    'dataset': dataset,
                    'dataset_label': DATASET_LABELS.get(dataset, dataset),
                    'horizon': int(r['horizon']),
                    'n_tickers_included': int(r['n_tickers_included']),
                    'n_tickers_total': int(r['n_tickers_total']),
                    'inclusion_rate': float(r['inclusion_rate']),
                    'mean_mcs_pvalue': float(r['mean_mcs_pvalue']) if 'mean_mcs_pvalue' in r else float('nan'),
                    'mean_loss': float(r['mean_loss']) if 'mean_loss' in r else float('nan'),
                }
            )
    return pd.DataFrame(rows)


def draw_figure(plot_df: pd.DataFrame, output_path: Path, datasets: list[str]) -> None:
    plt.rcParams.update(
        {
            'font.family': 'DejaVu Sans',
            'axes.edgecolor': '#666666',
            'axes.linewidth': 0.8,
        }
    )
    fig, ax = plt.subplots(figsize=(11.2, 6.6))
    x = list(range(len(MODEL_ORDER)))
    colors = {'MHAR': '#2354a6', 'PARTIAL_MALL': '#ff3b30'}
    markers = {'MHAR': 'o', 'PARTIAL_MALL': 'x'}
    for dataset in datasets:
        sub = plot_df[plot_df['dataset'] == dataset].set_index('model').loc[MODEL_ORDER].reset_index()
        marker = markers.get(dataset, 'o')
        marker_kwargs = (
            {'facecolors': 'none', 'edgecolors': colors.get(dataset, '#333333')}
            if marker == 'o'
            else {'c': colors.get(dataset, '#333333')}
        )
        ax.scatter(
            x,
            sub['inclusion_rate'],
            s=85,
            marker=marker,
            linewidths=2.3,
            label=DATASET_LABELS.get(dataset, dataset),
            **marker_kwargs,
        )
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('inclusion rate', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(m, m) for m in MODEL_ORDER], rotation=55, ha='right', fontsize=10)
    ax.grid(True, axis='both', color='#e2e2e2', linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', direction='in', top=True, right=True)
    ax.legend(loc='upper left', frameon=True, fancybox=False, framealpha=1.0, edgecolor='#666666', fontsize=11)
    ax.set_title('Figure 4 Inclusion rate in the MCS (h=1)', loc='left', fontsize=14, fontweight='bold')
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def draw_appendix_clean_figure(plot_df: pd.DataFrame, output_path: Path, datasets: list[str]) -> None:
    plt.rcParams.update(
        {
            'font.family': 'DejaVu Sans',
            'axes.edgecolor': '#666666',
            'axes.linewidth': 0.8,
        }
    )
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    x = list(range(len(MODEL_ORDER)))
    colors = {'MHAR': '#2354a6', 'PARTIAL_MALL': '#ff3b30'}
    markers = {'MHAR': 'o', 'PARTIAL_MALL': 'x'}
    for dataset in datasets:
        sub = plot_df[plot_df['dataset'] == dataset].set_index('model').loc[MODEL_ORDER].reset_index()
        marker = markers.get(dataset, 'o')
        marker_kwargs = (
            {'facecolors': 'none', 'edgecolors': colors.get(dataset, '#333333')}
            if marker == 'o'
            else {'c': colors.get(dataset, '#333333')}
        )
        ax.scatter(
            x,
            sub['inclusion_rate'],
            s=82,
            marker=marker,
            linewidths=2.2,
            label=DATASET_LABELS.get(dataset, dataset),
            clip_on=False,
            **marker_kwargs,
        )

    ax.set_ylim(-0.025, 1.055)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel('Inclusion rate', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(m, m) for m in MODEL_ORDER], rotation=48, ha='right', fontsize=10)
    ax.grid(True, axis='both', color='#e2e2e2', linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', direction='in', top=True, right=True)
    ax.legend(
        loc='lower left',
        bbox_to_anchor=(0.015, 0.035),
        ncol=1,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor='#666666',
        fontsize=11,
    )
    ax.set_title('Inclusion rate in the MCS, h=1', loc='left', fontsize=13, fontweight='bold', pad=10)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.25, top=0.91)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Create paper-style Figure 4 MCS inclusion-rate plot.')
    parser.add_argument(
        '--mcs-rates',
        default='outputs_mcs_core_with_bagging_gb_harfix_nn50_tuned_no_refit_h1h5_20260523/tables/mcs_inclusion_rates.csv',
    )
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--datasets', nargs='+', default=['MHAR', 'PARTIAL_MALL'])
    parser.add_argument(
        '--appendix-clean-only',
        action='store_true',
        help='Write only the unclipped appendix-clean figure and do not overwrite the original figure.',
    )
    args = parser.parse_args()

    mcs_path = _resolve(args.mcs_rates)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [str(d) for d in args.datasets]

    rates = load_inclusion_rates(mcs_path, args.horizon, datasets)
    plot_df = build_plot_data(rates, datasets)

    data_path = output_dir / 'figure4_mcs_inclusion_rate_h1.csv'
    png_path = output_dir / 'figure4_mcs_inclusion_rate_h1.png'
    appendix_png_path = output_dir / 'figure4_mcs_inclusion_rate_h1_appendix_clean.png'
    if args.appendix_clean_only:
        appendix_data_path = output_dir / 'figure4_mcs_inclusion_rate_h1_appendix_clean.csv'
        plot_df.to_csv(appendix_data_path, index=False)
        draw_appendix_clean_figure(plot_df, appendix_png_path, datasets)
    else:
        plot_df.to_csv(data_path, index=False)
        draw_figure(plot_df, png_path, datasets)
        draw_appendix_clean_figure(plot_df, appendix_png_path, datasets)

    provenance = {
        'created_at_utc': _utc_now(),
        'mcs_rates': str(mcs_path),
        'horizon': int(args.horizon),
        'datasets': datasets,
        'model_order': MODEL_ORDER,
        'outputs': {
            'figure': str(png_path),
            'appendix_clean_figure': str(appendix_png_path),
            'plot_data': str(data_path),
        },
    }
    provenance_path = output_dir / (
        'run_provenance_appendix_clean.json' if args.appendix_clean_only else 'run_provenance.json'
    )
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding='utf-8')
    if args.appendix_clean_only:
        print(f'Wrote {appendix_png_path}')
    else:
        print(f'Wrote {png_path}')
        print(f'Wrote {appendix_png_path}')
        print(f'Wrote {data_path}')


if __name__ == '__main__':
    main()
