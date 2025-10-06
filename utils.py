import pandas as pd
from metrics import calculate_full_performance_metrics


def load_and_split_data(file_path: str, train_ratio: float, test_ratio: float):
    """
    Carga datos desde un archivo CSV, procesa la columna de fecha y los divide
    en sets de entrenamiento, prueba y validación.

    Args:
        file_path (str): Ruta al archivo CSV de datos históricos.
        train_ratio (float): Proporción de datos para el set de entrenamiento (ej. 0.6).
        test_ratio (float): Proporción de datos para el set de prueba (ej. 0.2).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Una tupla con los
        DataFrames de entrenamiento, prueba y validación. Retorna (None, None, None)
        si el archivo no se encuentra.
    """
    try:
        data = pd.read_csv(file_path, skiprows=1)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{file_path}'.")
        return None, None, None

    data['Date'] = pd.to_datetime(data['Date'], format='mixed')
    data = data.set_index('Date').sort_index()

    train_size = int(len(data) * train_ratio)
    test_size = int(len(data) * test_ratio)

    train_df = data.iloc[:train_size]
    test_df = data.iloc[train_size: train_size + test_size]
    validation_df = data.iloc[train_size + test_size:]

    print(f"Datos cargados: {len(data)} velas en total.")
    return train_df, test_df, validation_df


def display_final_results(dataset_name: str, initial_cash: float, portfolio: pd.Series,
                          log: list, best_params: dict):
    """
    Imprime un reporte formateado con los resultados del backtest para un
    set de datos específico (Train, Test, o Validation).

    Args:
        dataset_name (str): Nombre del set de datos (ej. "Train").
        initial_cash (float): Capital inicial para ese periodo.
        portfolio (pd.Series): Serie temporal del valor del portafolio.
        log (list): Lista de operaciones cerradas.
        best_params (dict): Diccionario con los hiperparámetros óptimos utilizados.
    """
    final_metrics = calculate_full_performance_metrics(portfolio, log, 60)

    print(f"\n--- Métricas con Parámetros Óptimos ({dataset_name}) ---")

    # Imprimir los hiperparámetros solo para el primer reporte (Train)
    if dataset_name == "Train":
        print("\nMejores Hiperparámetros Utilizados:")
        for key, value in best_params.items():
            if isinstance(value, float):
                print(f"  - {key}: {value:.4f}")
            else:
                print(f"  - {key}: {value}")
        print("-" * 50)

    # Imprimir los resultados del portafolio
    final_value = portfolio.iloc[-1]
    net_return = (final_value - initial_cash) / initial_cash

    print(f"- Valor Inicial del Periodo:     ${initial_cash:,.2f} USD")
    print(f"- Valor Final del Portafolio:   ${final_value:,.2f} USD")
    print(f"- Rendimiento Neto del Periodo: {net_return:.2%}")
    print("-" * 50)

    # Imprimir el resto de las métricas
    for key, value in final_metrics.items():
        print(f"- {key}: {value}")