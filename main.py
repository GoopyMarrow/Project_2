import pandas as pd
from utils import load_and_split_data
from optimize import run_optimization
from backtesting import run_backtest
from metrics import calculate_full_performance_metrics
from plots import (
    plot_training_performance,
    plot_split_performance,
    plot_performance_vs_buy_and_hold
)


def display_final_results(dataset_name: str, initial_cash: float, portfolio: pd.Series,
                          log: list, open_positions: list, time_frame_minutes: int):
    """
    Imprime un reporte completo para un set de datos específico (Train, Test o Validation).
    """
    final_metrics = calculate_full_performance_metrics(portfolio, log, time_frame_minutes)
    print(f"\n--- Métricas de Desempeño ({dataset_name}) ---")
    final_value = portfolio.iloc[-1]
    print(f"- Valor Inicial del Periodo: ${initial_cash:,.2f} USD")
    print(f"- Valor Final del Portafolio:   ${final_value:,.2f} USD")
    open_longs = sum(1 for p in open_positions if p.type == 'LONG')
    open_shorts = sum(1 for p in open_positions if p.type == 'SHORT')
    print(f"- Posiciones Abiertas (Long):   {open_longs}")
    print(f"- Posiciones Abiertas (Short):  {open_shorts}")
    print("-" * 50)
    for key, value in final_metrics.items():
        print(f"- {key}: {value}")


def main():
    """
    Función principal que orquesta todo el proceso de backtesting y optimización.
    """
    # --- 1. Configuración ---
    DATA_FILE = 'Binance_BTCUSDT_1h.csv'
    INITIAL_CASH = 1_000_000
    COMMISSION = 0.00125
    TIME_FRAME_MINUTES = 60
    N_TRIALS = 50
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
    print(f"Mejores parámetros: {best_params}")

    # --- 4. Ejecución y Reporte en cada Set de Datos ---
    print("\nEjecutando backtest final en cada set de datos...")

    # Ejecución y reporte para TRAINING SET
    _, train_portfolio, train_log, train_open = run_backtest(
        train_df, INITIAL_CASH, COMMISSION, best_params
    )
    display_final_results("Train", INITIAL_CASH, train_portfolio, train_log, train_open, TIME_FRAME_MINUTES)

    # Ejecución y reporte para TEST SET
    cash_after_test, test_portfolio, test_log, test_open = run_backtest(
        test_df, INITIAL_CASH, COMMISSION, best_params
    )
    display_final_results("Test", INITIAL_CASH, test_portfolio, test_log, test_open, TIME_FRAME_MINUTES)

    # Ejecución y reporte para VALIDATION SET
    # Inicia con el capital original para una evaluación justa e independiente
    _, validation_portfolio, validation_log, validation_open = run_backtest(
        validation_df, INITIAL_CASH, COMMISSION, best_params
    )
    display_final_results("Validation", INITIAL_CASH, validation_portfolio, validation_log, validation_open,
                          TIME_FRAME_MINUTES)

    # --- 5. Generación de Gráficas Finales ---
    print("\nGenerando gráficas de rendimiento...")
    plot_training_performance(train_portfolio)
    plot_split_performance(test_portfolio, validation_portfolio)

    full_test_validation_portfolio = pd.concat([test_portfolio, validation_portfolio])
    full_prices = pd.concat([test_df['Close'], validation_df['Close']])
    plot_performance_vs_buy_and_hold(full_test_validation_portfolio, full_prices, INITIAL_CASH)


if __name__ == "__main__":
    main()