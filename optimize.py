import optuna
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from backtesting import run_backtest
from metrics import calculate_calmar_for_optimization


def get_params_from_trial(trial):
    """Añade documentacion"""
    return {
        'rsi_window': trial.suggest_int('rsi_window', 5, 50),
        'rsi_lower': trial.suggest_int('rsi_lower', 5, 35),
        'rsi_upper': trial.suggest_int('rsi_upper', 65, 95),
        'bb_window': trial.suggest_int('bb_window', 10, 50),
        'stoch_window': trial.suggest_int('stoch_window', 10, 50),
        'stoch_smooth_k': trial.suggest_int('stoch_smooth_k', 3, 10),
        'stoch_buy_th': trial.suggest_int('stoch_buy_th', 10, 30),
        'stoch_sell_th': trial.suggest_int('stoch_sell_th', 70, 90),
        'macd_short_window': trial.suggest_int('macd_short_window', 5, 50),
        'macd_long_window': trial.suggest_int('macd_long_window', 100, 300),
        'macd_signal_window': trial.suggest_int('macd_signal_window', 5, 50),
        'stop_loss': trial.suggest_float('stop_loss', 0.01, 0.15),
        'take_profit': trial.suggest_float('take_profit', 0.01, 0.15),
        'n_shares': trial.suggest_float('n_shares', 0.5, 10.0),
        'max_short_pct': trial.suggest_float('max_short_pct', 0.1, 0.5)
    }


def objective(trial: optuna.trial.Trial, data: pd.DataFrame, n_splits: int):
    """Añadir documentacion"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    params = get_params_from_trial(trial)

    for _, test_index in tscv.split(data):
        validation_set = data.iloc[test_index]
        _, portfolio_df, trades_log, _ = run_backtest(validation_set, 1_000_000, 0.00125, params)
        if len(trades_log) < 10:
            results.append(-1.0)
            continue
        results.append(calculate_calmar_for_optimization(portfolio_df))

    return np.mean(results)


def run_optimization(train_df: pd.DataFrame, n_trials: int, n_splits: int):
    """Añadir documentacion"""
    print("\nIniciando optimización walk-forward...")
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, train_df, n_splits=n_splits),
        n_trials=n_trials,
        show_progress_bar=True
    )
    return study