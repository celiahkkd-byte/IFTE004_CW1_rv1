from __future__ import annotations

from pathlib import Path
import logging
import zipfile
from typing import Dict, Iterable, Optional
import pandas as pd

logger = logging.getLogger(__name__)

RAW_COLUMNS = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']


def extract_intraday_source(raw_zip: Optional[Path], raw_dir: Path) -> Path:
    if raw_dir.exists() and any(raw_dir.glob('*.txt')):
        logger.info('Raw directory already exists: %s', raw_dir)
        return raw_dir
    if raw_zip is None:
        raise FileNotFoundError(f'Cannot find intraday txt files in {raw_dir}. Update config/default.yaml or provide raw_zip.')
    if not raw_zip.exists():
        raise FileNotFoundError(f'Cannot find raw zip {raw_zip}. Put the source zip in data/raw/ or update config/default.yaml.')
    raw_dir.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(raw_zip) as zf:
        zf.extractall(raw_dir.parent)
    # The zip contains data4rv/*.txt. If config raw_dir is data/raw/data4rv, this will match.
    logger.info('Extracted %s to %s', raw_zip, raw_dir.parent)
    return raw_dir


def extract_teacher_zip(raw_zip: Optional[Path], raw_dir: Path) -> Path:
    return extract_intraday_source(raw_zip, raw_dir)


def read_one_ticker_txt(path: Path, ticker: str | None = None) -> pd.DataFrame:
    ticker = ticker or path.stem.upper()
    df = pd.read_csv(path, header=None, names=RAW_COLUMNS)
    df['ticker'] = ticker
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
    # Combine date and time; data are regular U.S. market timestamps.
    df['timestamp'] = pd.to_datetime(df['date'].dt.strftime('%Y-%m-%d') + ' ' + df['time'].astype(str), errors='coerce')
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = df['timestamp'].dt.normalize()
    return df[['ticker', 'date', 'timestamp', 'open', 'high', 'low', 'close', 'volume']].sort_values(['ticker', 'timestamp'])


def load_intraday_txt(raw_zip: Optional[Path], raw_dir: Path, tickers: Iterable[str]) -> Dict[str, pd.DataFrame]:
    raw_dir = extract_intraday_source(raw_zip, raw_dir)
    data: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        p = raw_dir / f'{ticker}.txt'
        if not p.exists():
            # Some users pass raw_dir as the parent directory after extraction.
            candidates = list(raw_dir.rglob(f'{ticker}.txt'))
            if candidates:
                p = candidates[0]
        if not p.exists():
            raise FileNotFoundError(f'Missing {ticker}.txt in {raw_dir}')
        logger.info('Reading %s', p)
        data[ticker] = read_one_ticker_txt(p, ticker)
        logger.info('%s rows: %d, dates: %s to %s', ticker, len(data[ticker]), data[ticker]['date'].min().date(), data[ticker]['date'].max().date())
    return data


def load_teacher_intraday(raw_zip: Optional[Path], raw_dir: Path, tickers: Iterable[str]) -> Dict[str, pd.DataFrame]:
    return load_intraday_txt(raw_zip, raw_dir, tickers)
