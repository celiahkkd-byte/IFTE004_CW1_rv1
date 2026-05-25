from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlencode
import datetime as dt
import time

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


def _download_to_file(url: str, path: Path, timeout: int = 30) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        logger.info('Downloading %s -> %s', url, path)
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        path.write_bytes(r.content)
        return True
    except Exception as exc:
        logger.warning('Download failed for %s: %s', url, exc)
        return False


def load_fred_csv(path: Path, url: str, value_name: str, allow_network: bool = True) -> pd.DataFrame:
    if not path.exists() and allow_network:
        _download_to_file(url, path)
    if not path.exists():
        logger.warning('Missing FRED file %s; variable %s unavailable.', path, value_name)
        return pd.DataFrame(columns=['date', value_name])
    df = pd.read_csv(path)
    date_col = 'DATE' if 'DATE' in df.columns else df.columns[0]
    value_col = [c for c in df.columns if c != date_col][0]
    out = pd.DataFrame({'date': pd.to_datetime(df[date_col], errors='coerce'), value_name: pd.to_numeric(df[value_col], errors='coerce')})
    out = out.dropna(subset=['date']).sort_values('date')
    return out


def load_epu_daily(path: Path, url: str, allow_network: bool = True) -> pd.DataFrame:
    if not path.exists() and allow_network:
        _download_to_file(url, path)
    if not path.exists():
        logger.warning('Missing EPU daily file %s; EPU unavailable.', path)
        return pd.DataFrame(columns=['date', 'epu'])
    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}
    if {'day', 'month', 'year', 'daily_policy_index'}.issubset(lower):
        out = pd.DataFrame({
            'date': pd.to_datetime(dict(year=df[lower['year']], month=df[lower['month']], day=df[lower['day']]), errors='coerce'),
            'epu': pd.to_numeric(df[lower['daily_policy_index']], errors='coerce'),
        })
    else:
        # Fallback for alternative column naming.
        date_col = next((c for c in df.columns if 'date' in c.lower()), df.columns[0])
        val_col = next((c for c in df.columns if 'policy' in c.lower() or 'epu' in c.lower()), df.columns[-1])
        out = pd.DataFrame({'date': pd.to_datetime(df[date_col], errors='coerce'), 'epu': pd.to_numeric(df[val_col], errors='coerce')})
    return out.dropna(subset=['date']).sort_values('date')


def load_ads(path: Path, url: str, allow_network: bool = True) -> pd.DataFrame:
    if not path.exists() and allow_network:
        _download_to_file(url, path)
    if not path.exists():
        logger.warning('Missing ADS file %s; ADS unavailable.', path)
        return pd.DataFrame(columns=['date', 'ads'])
    try:
        df = pd.read_excel(path)
    except Exception as exc:
        logger.warning('Cannot read ADS XLSX %s: %s', path, exc)
        return pd.DataFrame(columns=['date', 'ads'])
    # The current-vintage spreadsheet has changed names historically; infer date and ADS columns.
    date_col = None
    for c in df.columns:
        if 'date' in str(c).lower():
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]
    value_candidates = [c for c in df.columns if 'ads' in str(c).lower() or 'index' in str(c).lower()]
    value_col = value_candidates[0] if value_candidates else df.columns[1]
    date_raw = df[date_col]
    date = pd.to_datetime(date_raw, format='%Y:%m:%d', errors='coerce')
    if date.isna().all():
        date = pd.to_datetime(date_raw, errors='coerce')
    out = pd.DataFrame({'date': date, 'ads': pd.to_numeric(df[value_col], errors='coerce')})
    return out.dropna(subset=['date']).sort_values('date')


def _yahoo_unix(date_str: str) -> int:
    return int(pd.Timestamp(date_str, tz='UTC').timestamp())


def load_hsi(path: Path, symbol: str, start: str, end: str, allow_network: bool = True) -> pd.DataFrame:
    """Load Hang Seng squared log return.

    Tries an existing local file first, then Yahoo chart/download endpoints, then yfinance.
    """
    if not path.exists() and allow_network:
        # Yahoo query1 CSV endpoint. It may fail if Yahoo changes access requirements.
        params = {
            'period1': _yahoo_unix(start),
            'period2': _yahoo_unix(end),
            'interval': '1d',
            'events': 'history',
            'includeAdjustedClose': 'true',
        }
        url = f'https://query1.finance.yahoo.com/v7/finance/download/{symbol}?{urlencode(params)}'
        ok = _download_to_file(url, path)
        if not ok:
            try:
                import yfinance as yf  # type: ignore
                df_yf = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
                if not df_yf.empty:
                    df_yf.reset_index().to_csv(path, index=False)
            except Exception as exc:
                logger.warning('yfinance fallback failed for HSI: %s', exc)
    if not path.exists():
        logger.warning('Missing HSI file %s; HSI unavailable.', path)
        return pd.DataFrame(columns=['date', 'hsi'])
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if str(c).lower() in ['date', 'datetime']), df.columns[0])
    # If the user supplies hsi directly, keep it. Otherwise infer close and square log return.
    if 'hsi' in [c.lower() for c in df.columns]:
        val_col = [c for c in df.columns if c.lower() == 'hsi'][0]
        out = pd.DataFrame({'date': pd.to_datetime(df[date_col], errors='coerce'), 'hsi': pd.to_numeric(df[val_col], errors='coerce')})
    else:
        close_col = next((c for c in df.columns if str(c).lower() in ['adj close', 'adj_close', 'close']), None)
        if close_col is None:
            logger.warning('Cannot infer HSI close column in %s', path)
            return pd.DataFrame(columns=['date', 'hsi'])
        close = pd.to_numeric(df[close_col], errors='coerce')
        out = pd.DataFrame({'date': pd.to_datetime(df[date_col], errors='coerce'), 'hsi_close': close})
        out['hsi'] = np.log(out['hsi_close']).diff() ** 2
        out = out[['date', 'hsi']]
    return out.dropna(subset=['date']).sort_values('date')


def load_optional_iv(path: Path) -> pd.DataFrame:
    """Optional user-supplied firm-level implied volatility.

    Expected columns: date,ticker,iv. If not present, returns empty.
    """
    if not path.exists():
        return pd.DataFrame(columns=['date', 'ticker', 'iv'])
    df = pd.read_csv(path)
    needed = {'date', 'ticker', 'iv'}
    if not needed.issubset({c.lower() for c in df.columns}):
        logger.warning('Optional IV file exists but does not contain date,ticker,iv: %s', path)
        return pd.DataFrame(columns=['date', 'ticker', 'iv'])
    cols = {c.lower(): c for c in df.columns}
    out = pd.DataFrame({
        'date': pd.to_datetime(df[cols['date']], errors='coerce'),
        'ticker': df[cols['ticker']].astype(str).str.upper(),
        'iv': pd.to_numeric(df[cols['iv']], errors='coerce'),
    })
    return out.dropna(subset=['date', 'ticker']).sort_values(['ticker', 'date'])


def load_optional_earnings(path: Path) -> pd.DataFrame:
    """Optional historical earnings-announcement dates.

    Expected columns: date,ticker. Returns date,ticker,ea=1.
    """
    if not path.exists():
        return pd.DataFrame(columns=['date', 'ticker', 'ea'])
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if not {'date', 'ticker'}.issubset(cols):
        logger.warning('Optional earnings file exists but does not contain date,ticker: %s', path)
        return pd.DataFrame(columns=['date', 'ticker', 'ea'])
    out = pd.DataFrame({
        'date': pd.to_datetime(df[cols['date']], errors='coerce'),
        'ticker': df[cols['ticker']].astype(str).str.upper(),
        'ea': 1,
    })
    return out.dropna(subset=['date', 'ticker']).drop_duplicates(['date', 'ticker'])


def download_and_load_external(cfg: Dict) -> Dict[str, pd.DataFrame]:
    ext_dir = Path(cfg['_base_dir']) / cfg['paths']['external_dir'] if not Path(cfg['paths']['external_dir']).is_absolute() else Path(cfg['paths']['external_dir'])
    ext_dir.mkdir(parents=True, exist_ok=True)
    e = cfg['external_data']
    allow = bool(e.get('allow_network', True))
    data = {
        'vix': load_fred_csv(ext_dir / 'vix_fred.csv', e['vix_fred_url'], 'vix', allow),
        'us3m': load_fred_csv(ext_dir / 'us3m_fred.csv', e['us3m_fred_url'], 'us3m', allow),
        'epu': load_epu_daily(ext_dir / 'epu_daily.csv', e['epu_daily_url'], allow),
        'ads': load_ads(ext_dir / 'ads.xlsx', e['ads_xlsx_url'], allow),
        'hsi': load_hsi(ext_dir / 'hsi.csv', e.get('hsi_yahoo_symbol', '^HSI'), e['start'], e['end'], allow),
        'iv': load_optional_iv(ext_dir / 'firm_iv.csv'),
        'earnings': load_optional_earnings(ext_dir / 'earnings_announcements.csv'),
    }
    for k, v in data.items():
        logger.info('External %s rows=%d cols=%s', k, len(v), list(v.columns))
    return data
