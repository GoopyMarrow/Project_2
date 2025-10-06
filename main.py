import pandas as pd

from utils import load_and_split_data, display_final_results
from optimize import run_optimization
from backtesting import run_backtest
from plots import (
    plot_training_performance,
    plot_split_performance,
    plot_performance_vs_buy_and_hold
)


def main():
    """
    Función principal que orquesta todo el proceso de backtesting y optimización.
    """
    # --- 1. Configuración ---
    DATA_FILE = 'Binance_BTCUSDT_1h.csv'
    INITIAL_CASH = 1_000_000
    COMMISSION = 0.00125
    N_TRIALS = 250
    N_SPLITS = 5

    # --- 2. Carga y División de Datos ---
    train_df, test_df, validation_df = load_and_split_data(
        DATA_FILE, train_ratio=0.6, test_ratio=0.2
    )
    if train_df is None:
        return

    # --- 3. Optimización de Hiperparámetros ---
    study = run_optimization(train_df, n_trials=N_TRIALS, n_splits=N_SPLITS)
    print(f"\nOptimización completada. Mejor Calmar Ratio promedio: {study.best_value:.4f}")
    best_params = study.best_params

    # --- 4. Ejecución y Reporte en cada Set de Datos ---
    print("\nEjecutando backtest final en cada set de datos...")

    # Ejecución para TRAINING SET
    _, train_portfolio, train_log, _ = run_backtest(
        train_df, INITIAL_CASH, COMMISSION, best_params
    )
    display_final_results("Train", INITIAL_CASH, train_portfolio, train_log, best_params)

    # Ejecución para TEST SET
    cash_after_test, test_portfolio, test_log, _ = run_backtest(
        test_df, INITIAL_CASH, COMMISSION, best_params
    )
    display_final_results("Test", INITIAL_CASH, test_portfolio, test_log, best_params)

    # Ejecución para VALIDATION SET
    _, validation_portfolio, validation_log, _ = run_backtest(
        validation_df, cash_after_test, COMMISSION, best_params
    )
    display_final_results("Validation", cash_after_test, validation_portfolio, validation_log, best_params)

    # --- 5. Generación de Gráficas Finales ---
    print("\nGenerando gráficas de rendimiento...")
    plot_training_performance(train_portfolio)
    plot_split_performance(test_portfolio, validation_portfolio)

    full_test_validation_portfolio = pd.concat([test_portfolio, validation_portfolio])
    full_prices = pd.concat([test_df['Close'], validation_df['Close']])
    plot_performance_vs_buy_and_hold(full_test_validation_portfolio, full_prices, INITIAL_CASH)


if __name__ == "__main__":
    main()