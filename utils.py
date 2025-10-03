import pandas as pd

def load_and_split_data(file_path: str, train_ratio: float, test_ratio: float):
    """Añadir documentacion"""
    try:
        data = pd.read_csv(file_path, skiprows=1) # Para no leer la primera fila que es un link
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