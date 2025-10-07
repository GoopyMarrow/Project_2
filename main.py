import pandas as pd
from utils import load_and_split_data, display_final_results
from optimize import run_optimization
from backtesting import run_backtest
from plots import (
    plot_single_period_performance,
    plot_split_performance,
    plot_performance_vs_buy_and_hold
)

# Entrypoint

def main():
    """
    Ejecuta el proceso completo de backtesting y optimización de la estrategia.

    El flujo de ejecución es el siguiente:
    1.  Define las configuraciones iniciales (archivo de datos, capital, comisiones).
    2.  Carga y divide los datos históricos en sets de entrenamiento, prueba y validación.
    3.  Ejecuta la optimización de hiperparámetros en el set de entrenamiento
        utilizando Optuna con validación cruzada (walk-forward).
    4.  Imprime los mejores parámetros encontrados.
    5.  Ejecuta el backtest final con los parámetros óptimos en cada uno de los
        tres sets de datos (Train, Test, Validation).
    6.  Muestra un reporte de métricas detallado para cada set.
    7.  Genera y muestra las gráficas de rendimiento correspondientes.
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

    print("\n--- Mejores Hiperparámetros Encontrados ---")
    for key, value in best_params.items():
        if isinstance(value, float):
            print(f"  - {key}: {value:.4f}")
        else:
            print(f"  - {key}: {value}")

    # --- 4. Ejecución y Reporte en cada Set de Datos ---
    print("\nEjecutando backtest final en cada set de datos...")

    _, train_portfolio, train_log, _ = run_backtest(
        train_df, INITIAL_CASH, COMMISSION, best_params
    )
    display_final_results("Train", INITIAL_CASH, train_portfolio, train_log, best_params)

    cash_after_test, test_portfolio, test_log, _ = run_backtest(
        test_df, INITIAL_CASH, COMMISSION, best_params
    )
    display_final_results("Test", INITIAL_CASH, test_portfolio, test_log, best_params)

    _, validation_portfolio, validation_log, _ = run_backtest(
        validation_df, cash_after_test, COMMISSION, best_params
    )
    display_final_results("Validation", cash_after_test, validation_portfolio, validation_log, best_params)

    # --- 5. Generación de Gráficas Finales ---
    print("\nGenerando gráficas de rendimiento...")

    # 1. Gráfica del periodo Train
    plot_single_period_performance(train_portfolio, title="Rendimiento en Periodo de Entrenamiento")

    # 2. Gráfica del periodo Test
    plot_single_period_performance(test_portfolio, title="Rendimiento en Periodo de Test")

    # 3. Gráfica del periodo Validation
    plot_single_period_performance(validation_portfolio, title="Rendimiento en Periodo de Validation")

    # 4. Gráfica combinada Test y Validation
    plot_split_performance(test_portfolio, validation_portfolio)

    # 5. Gráfica comparativa vs. Buy & Hold
    full_test_validation_portfolio = pd.concat([test_portfolio, validation_portfolio])
    full_prices = pd.concat([test_df['Close'], validation_df['Close']])
    plot_performance_vs_buy_and_hold(full_test_validation_portfolio, full_prices, INITIAL_CASH)


if __name__ == "__main__":
    main()

