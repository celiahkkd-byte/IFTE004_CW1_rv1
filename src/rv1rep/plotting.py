from __future__ import annotations

import os
from pathlib import Path
import pandas as pd

os.environ.setdefault('MPLCONFIGDIR', '/tmp/rv1rep_matplotlib')
import matplotlib.pyplot as plt


def plot_relative_mse(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for dataset in summary['dataset'].unique():
        sub = summary[(summary['dataset'] == dataset) & (summary['horizon'] == summary['horizon'].min())].copy()
        sub = sub.sort_values('avg_rel_mse_vs_har')
        plt.figure(figsize=(10, max(4, len(sub) * 0.28)))
        plt.barh(sub['model'], sub['avg_rel_mse_vs_har'])
        plt.axvline(1.0, linestyle='--')
        plt.xlabel('Average relative MSE vs HAR')
        plt.title(f'Relative MSE, {dataset}')
        plt.tight_layout()
        plt.savefig(output_path.parent / f'relative_mse_{dataset}.png', dpi=200)
        plt.close()


def plot_realized_volatility(daily: pd.DataFrame, output_path: Path) -> None:
    if daily.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    for ticker, g in daily.groupby('ticker'):
        plt.plot(g['date'], g['rv'], label=ticker, linewidth=0.8)
    plt.legend()
    plt.ylabel('Realized variance')
    plt.title('Daily realized variance from 5-minute returns')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
