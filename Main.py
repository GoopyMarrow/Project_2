# --- 5. FUNCIÓN PRINCIPAL ---
def main():
    DATA_FILE = 'Binance_BTCUSDT_1h.csv'
    INITIAL_CASH = 1_000_000;
    COMMISSION = 0.00125;
    TIME_FRAME_MINUTES = 60

    try:
        data = pd.read_csv(DATA_FILE, skiprows=1)
    except FileNotFoundError:
        print(f"Error: No se encontró '{DATA_FILE}'."); return
    data['Date'] = pd.to_datetime(data['Date'], format='mixed');
    data = data.set_index('Date').sort_index()

    train_size = int(len(data) * 0.6);
    test_size = int(len(data) * 0.2)
    train_df, test_df, validation_df = data.iloc[:train_size], data.iloc[train_size:train_size + test_size], data.iloc[
        train_size + test_size:]

    print(f"Datos cargados: {len(data)} velas.");
    print("\nIniciando optimización walk-forward...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: walk_forward(train_df, trial, n_splits=5), n_trials=100, show_progress_bar=True)

    print(f"\nOptimización completada. Mejor Calmar Ratio promedio: {study.best_value:.4f}")
    best_params = study.best_params;
    print(f"Mejores parámetros: {best_params}")

    print("\nEjecutando backtest final...")
    # Ejecución en Test
    cash_after_test, test_portfolio, test_log, _ = run_backtest(test_df, INITIAL_CASH, COMMISSION, best_params)
    # Ejecución en Validation
    final_cash, validation_portfolio, validation_log, open_positions = run_backtest(validation_df, cash_after_test,
                                                                                    COMMISSION, best_params)

    # Combinar resultados para el reporte y gráficas finales
    full_test_validation_portfolio = pd.concat([test_portfolio, validation_portfolio])
    full_test_validation_log = test_log + validation_log

    # --- Reporte de Métricas Final ---
    final_metrics = calculate_performance_metrics(full_test_validation_portfolio, full_test_validation_log,
                                                  TIME_FRAME_MINUTES)
    print("\n--- Métricas de Desempeño Finales (Test + Validation) ---")
    final_value = full_test_validation_portfolio.iloc[-1]
    print(f"- Valor Inicial del Periodo: ${INITIAL_CASH:,.2f} USD")
    print(f"- Valor Final del Portafolio:   ${final_value:,.2f} USD")
    open_longs = sum(1 for p in open_positions if p.type == 'LONG');
    open_shorts = sum(1 for p in open_positions if p.type == 'SHORT')
    print(f"- Posiciones Abiertas (Long):   {open_longs}")
    print(f"- Posiciones Abiertas (Short):  {open_shorts}")
    print("-" * 50)
    for key, value in final_metrics.items(): print(f"- {key}: {value}")

    # --- Gráficas Finales ---
    print("\nGenerando gráficas de rendimiento...")
    # Gráfica de Train
    _, train_portfolio, _, _ = run_backtest(train_df, INITIAL_CASH, COMMISSION, best_params)
    plt.figure(figsize=(15, 7));
    train_portfolio.plot(title='Rendimiento en Entrenamiento');
    plt.show()
    # Gráfica de Test + Validation
    plot_split_performance(test_portfolio, validation_portfolio)
    # Gráfica vs. Buy & Hold (solo en periodo Test + Validation)
    full_test_validation_prices = pd.concat([test_df['Close'], validation_df['Close']])
    plot_vs_buy_and_hold(full_test_validation_portfolio, full_test_validation_prices, INITIAL_CASH)

    if __name__ == "__main__":
        main()