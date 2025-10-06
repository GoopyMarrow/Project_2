import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from backtesting import run_backtest
from metrics import calculate_calmar_for_optimization


def get_params_from_trial(trial: optuna.trial.Trial) -> dict:
    """
    Define el espacio de búsqueda de hiperparámetros para una prueba de Optuna.

    Args:
        trial (optuna.trial.Trial): Objeto de Optuna que sugiere los valores.

    Returns:
        dict: Un diccionario con un conjunto de parámetros sugeridos para la prueba.
    """
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
        # 'n_shares': trial.suggest_float('n_shares', 0.5, 10.0), # No funciona tan bien
        'pct_cash': trial.suggest_float('pct_cash', 0.001, 0.250, step= 0.001), # Funciona mejor
        'max_short_pct': trial.suggest_float('max_short_pct', 0.1, 0.5)
    }


def objective(trial: optuna.trial.Trial, data: pd.DataFrame, n_splits: int) -> float:
    """
    Función objetivo que Optuna evalúa y maximiza, usando validación cruzada.

    Aplica la estrategia con los parámetros del `trial` sobre múltiples ventanas
    de tiempo y retorna el promedio del Calmar Ratio.

    Args:
        trial (optuna.trial.Trial): La prueba actual de Optuna.
        data (pd.DataFrame): El set de datos de entrenamiento.
        n_splits (int): Número de divisiones para la validación cruzada (walk-forward).

    Returns:
        float: El valor promedio de la métrica (Calmar Ratio) en todas las divisiones.
    """
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


def run_optimization(train_df: pd.DataFrame, n_trials: int, n_splits: int) -> optuna.study.Study:
    """
    Configura y ejecuta el estudio de optimización de hiperparámetros.

    Args:
        train_df (pd.DataFrame): El set de datos de entrenamiento.
        n_trials (int): Número de pruebas que realizará Optuna.
        n_splits (int): Número de divisiones para el walk-forward.

    Returns:
        optuna.study.Study: El objeto de estudio de Optuna con todos los resultados.
    """
    print("\nIniciando optimización walk-forward...")
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, train_df, n_splits=n_splits),
        n_trials=n_trials,
        show_progress_bar=True
    )
    return study

