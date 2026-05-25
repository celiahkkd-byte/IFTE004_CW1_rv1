from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def test_imports():
    import rv1rep.config
    import rv1rep.intraday
    import rv1rep.features
    import rv1rep.models
    import rv1rep.evaluation


def test_five_min_return_count_small():
    import pandas as pd
    import numpy as np
    from rv1rep.intraday import _daily_5min_returns
    base = pd.Timestamp('2020-01-02 09:30')
    df = pd.DataFrame({
        'timestamp': [base + pd.Timedelta(minutes=i) for i in range(390)],
        'open': np.linspace(100, 101, 390),
        'high': np.linspace(100, 101, 390),
        'low': np.linspace(100, 101, 390),
        'close': np.linspace(100.001, 101.001, 390),
        'volume': 1,
    })
    r = _daily_5min_returns(df, expected_bars=390, bar_interval_minutes=1)
    assert len(r) == 78


def test_five_min_return_count_from_five_min_bars():
    import pandas as pd
    import numpy as np
    from rv1rep.intraday import _daily_5min_returns
    base = pd.Timestamp('2020-01-02 09:30')
    df = pd.DataFrame({
        'timestamp': [base + pd.Timedelta(minutes=5 * i) for i in range(78)],
        'open': np.linspace(100, 101, 78),
        'high': np.linspace(100, 101, 78),
        'low': np.linspace(100, 101, 78),
        'close': np.linspace(100.001, 101.001, 78),
        'volume': 1,
    })
    r = _daily_5min_returns(df, expected_bars=78, bar_interval_minutes=5)
    assert len(r) == 78


def test_intraday_validation_removes_half_day_and_off_grid_bars():
    import pandas as pd
    import numpy as np
    from rv1rep.intraday import validate_intraday_panel

    base = pd.Timestamp('2020-01-02 09:30')
    full = pd.DataFrame({
        'ticker': 'AAPL',
        'date': pd.Timestamp('2020-01-02'),
        'timestamp': [base + pd.Timedelta(minutes=5 * i) for i in range(78)],
        'open': np.linspace(100, 101, 78),
        'high': np.linspace(100, 101, 78),
        'low': np.linspace(100, 101, 78),
        'close': np.linspace(100.001, 101.001, 78),
        'volume': 1,
    })
    half = full.iloc[:42].copy()
    half['date'] = pd.Timestamp('2020-01-03')
    half['timestamp'] = half['timestamp'] + pd.Timedelta(days=1)
    off_grid = full.copy()
    off_grid['date'] = pd.Timestamp('2020-01-06')
    off_grid['timestamp'] = off_grid['timestamp'] + pd.Timedelta(days=4)
    off_grid.loc[off_grid.index[10], 'timestamp'] = off_grid.loc[off_grid.index[10], 'timestamp'] + pd.Timedelta(minutes=1)

    valid, summary = validate_intraday_panel(
        {'AAPL': pd.concat([full, half, off_grid], ignore_index=True)},
        trading_start='09:30',
        trading_end='15:55',
        bars_per_day=78,
        bar_interval_minutes=5,
        timestamp_label='bar_start',
    )
    assert valid['AAPL']['date'].nunique() == 1
    assert len(valid['AAPL']) == 78
    assert int(summary.loc[summary['ticker'].eq('AAPL'), 'bad_bar_count_days'].iloc[0]) == 1
    assert int(summary.loc[summary['ticker'].eq('AAPL'), 'off_grid_days'].iloc[0]) == 2


def test_loghar_frame_keeps_level_target_for_evaluation():
    import pandas as pd
    import numpy as np
    from rv1rep.features import make_model_frame

    panel = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=3),
        'ticker': ['AAPL'] * 3,
        'rv': [0.1, 0.2, 0.3],
        'oc_logret': [0.01, 0.02, 0.03],
        'cc_logret': [0.01, 0.02, 0.03],
        'log_rvd': np.log([0.1, 0.2, 0.3]),
        'log_rvw': np.log([0.1, 0.2, 0.3]),
        'log_rvm': np.log([0.1, 0.2, 0.3]),
        'target_rv_h1': [0.2, 0.3, 0.4],
        'target_log_rv_h1': np.log([0.2, 0.3, 0.4]),
    })
    frame, _, target_col = make_model_frame(panel, 'LogHAR', 'MHAR', 1)
    assert target_col == 'target_log_rv_h1'
    assert 'target_rv_h1' in frame.columns


def test_loghar_partial_mall_uses_paper_log_transforms_for_vix_and_iv():
    from rv1rep.features import feature_columns_for_model

    columns = {
        'rvd', 'rvw', 'rvm', 'log_rvd', 'log_rvw', 'log_rvm',
        'iv', 'log_iv', 'vix', 'log_vix',
        'ea', 'm1w', 'dvol', 'hsi', 'ads', 'us3m_diff', 'epu',
    }
    loghar_cols = feature_columns_for_model('LogHAR', 'PARTIAL_MALL', columns)
    harx_cols = feature_columns_for_model('HARX', 'PARTIAL_MALL', columns)

    assert {'log_iv', 'log_vix'}.issubset(loghar_cols)
    assert 'iv' not in loghar_cols
    assert 'vix' not in loghar_cols
    assert {'iv', 'vix'}.issubset(harx_cols)
    assert 'log_iv' not in harx_cols
    assert 'log_vix' not in harx_cols


def test_standardizer_mainline_standardizes_ea_by_default():
    import numpy as np
    import pandas as pd
    from rv1rep.preprocessing import Standardizer

    train = pd.DataFrame({
        'rvd': [1.0, 2.0, 3.0, 4.0],
        'ea': [0, 1, 0, 1],
    })
    test = pd.DataFrame({
        'rvd': [5.0, 6.0],
        'ea': [1, 0],
    })

    scaler = Standardizer().fit(train)
    transformed = scaler.transform(test)

    expected_rvd = (test['rvd'] - train['rvd'].mean()) / train['rvd'].std()
    expected_ea = (test['ea'] - train['ea'].mean()) / train['ea'].std()
    assert np.allclose(transformed['rvd'], expected_rvd)
    assert np.allclose(transformed['ea'], expected_ea)
    assert scaler.categorical_columns_ == []
    assert scaler.continuous_columns_ == ['rvd', 'ea']


def test_standardizer_can_leave_ea_indicator_unscaled_for_robustness():
    import numpy as np
    import pandas as pd
    from rv1rep.preprocessing import Standardizer

    train = pd.DataFrame({
        'rvd': [1.0, 2.0, 3.0, 4.0],
        'ea': [0, 1, 0, 1],
    })
    test = pd.DataFrame({
        'rvd': [5.0, 6.0],
        'ea': [1, 0],
    })

    scaler = Standardizer(standardize_binary_features=False).fit(train)
    transformed = scaler.transform(test)

    expected_rvd = (test['rvd'] - train['rvd'].mean()) / train['rvd'].std()
    assert np.allclose(transformed['rvd'], expected_rvd)
    assert transformed['ea'].tolist() == [1, 0]
    assert scaler.categorical_columns_ == ['ea']
    assert scaler.continuous_columns_ == ['rvd']


def test_accumulated_local_effect_handles_binary_feature():
    import numpy as np
    import pandas as pd
    from rv1rep.explain import accumulated_local_effect

    X = pd.DataFrame({
        'ea': [0, 0, 1, 1],
        'rvd': [1.0, 2.0, 3.0, 4.0],
    })

    def predict(frame):
        return 2.0 * frame['ea'].to_numpy(dtype=float)

    ale = accumulated_local_effect(predict, X, 'ea', grid_size=100)

    assert ale['x'].tolist() == [0.0, 1.0]
    assert np.allclose(ale['ale'], [-1.0, 1.0])


def test_tuned_ridge_no_refit_uses_training_only_after_validation_selection():
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Ridge
    from rv1rep.models import fit_sklearn_model

    X_train = pd.DataFrame({'x': [0.0, 1.0, 2.0, 3.0]})
    y_train = pd.Series([0.0, 1.0, 2.0, 3.0])
    X_val = pd.DataFrame({'x': [4.0, 5.0]})
    y_val = pd.Series([40.0, 50.0])
    cfg = {
        'project': {'random_seed': 42},
        'estimation': {'refit_tuned_models_on_train_validation': False},
        'models': {
            'regularization': {
                'alpha_min': 1.0,
                'alpha_max': 1.0,
                'alpha_grid_size': 1,
                'elastic_l1_ratios': [0.5],
            }
        },
    }

    est, params = fit_sklearn_model('Ridge', X_train, y_train, X_val, y_val, cfg)
    expected_train = Ridge(alpha=1.0, random_state=42).fit(X_train, y_train)
    expected_train_val = Ridge(alpha=1.0, random_state=42).fit(
        pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
    )

    x_check = pd.DataFrame({'x': [6.0]})
    assert params['fit_sample'] == 'train_only_after_validation_selection'
    assert np.allclose(est.predict(x_check), expected_train.predict(x_check))
    assert not np.allclose(est.predict(x_check), expected_train_val.predict(x_check))


def test_tuned_ridge_default_is_no_refit():
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Ridge
    from rv1rep.models import fit_sklearn_model

    X_train = pd.DataFrame({'x': [0.0, 1.0, 2.0, 3.0]})
    y_train = pd.Series([0.0, 1.0, 2.0, 3.0])
    X_val = pd.DataFrame({'x': [4.0, 5.0]})
    y_val = pd.Series([40.0, 50.0])
    cfg = {
        'project': {'random_seed': 42},
        'models': {
            'regularization': {
                'alpha_min': 1.0,
                'alpha_max': 1.0,
                'alpha_grid_size': 1,
                'elastic_l1_ratios': [0.5],
            }
        },
    }

    est, params = fit_sklearn_model('Ridge', X_train, y_train, X_val, y_val, cfg)
    expected_train = Ridge(alpha=1.0, random_state=42).fit(X_train, y_train)

    x_check = pd.DataFrame({'x': [6.0]})
    assert params['fit_sample'] == 'train_only_after_validation_selection'
    assert np.allclose(est.predict(x_check), expected_train.predict(x_check))


def test_har_still_uses_train_plus_validation_when_tuned_no_refit_enabled():
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from rv1rep.models import fit_sklearn_model

    X_train = pd.DataFrame({'x': [0.0, 1.0, 2.0, 3.0]})
    y_train = pd.Series([0.0, 1.0, 2.0, 3.0])
    X_val = pd.DataFrame({'x': [4.0, 5.0]})
    y_val = pd.Series([40.0, 50.0])
    cfg = {
        'project': {'random_seed': 42},
        'estimation': {'refit_tuned_models_on_train_validation': False},
        'models': {
            'regularization': {
                'alpha_min': 1.0,
                'alpha_max': 1.0,
                'alpha_grid_size': 1,
                'elastic_l1_ratios': [0.5],
            }
        },
    }

    est, params = fit_sklearn_model('HAR', X_train, y_train, X_val, y_val, cfg)
    expected = LinearRegression().fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

    assert params['method'] == 'OLS_train_plus_validation'
    assert np.allclose(est.predict(pd.DataFrame({'x': [6.0]})), expected.predict(pd.DataFrame({'x': [6.0]})))
