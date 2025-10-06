import pandas as pd


def load_and_split_data(file_path: str, train_ratio: float, test_ratio: float):
    """
    Carga datos de un archivo CSV, procesa la columna de fecha y los divide
    en sets de entrenamiento, prueba y validación.

    Args:
        file_path (str): Ruta al archivo CSV.
        train_ratio (float): Proporción de datos para el set de entrenamiento (ej. 0.6).
        test_ratio (float): Proporción de datos para el set de prueba (ej. 0.2).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Una tupla con los
        DataFrames de entrenamiento, prueba y validación. Retorna None si el
        archivo no se encuentra.
    """
    data = pd.read_csv(file_path, skiprows=1)

    data['Date'] = pd.to_datetime(data['Date'], format='mixed')
    data = data.set_index('Date').sort_index()

    train_size = int(len(data) * train_ratio)
    test_size = int(len(data) * test_ratio)

    train_df = data.iloc[:train_size]
    test_df = data.iloc[train_size: train_size + test_size]
    validation_df = data.iloc[train_size + test_size:]

    print(f"Datos cargados: {len(data)} velas en total.")
    return train_df, test_df, validation_df