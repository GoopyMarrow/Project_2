import pandas as pd
import ta


def calculate_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Calcula un conjunto de indicadores técnicos y los añade como columnas a un DataFrame.

    Indicadores calculados:
    - RSI (Relative Strength Index)
    - Bandas de Bollinger (Bollinger Bands)
    - Oscilador Estocástico (Stochastic Oscillator)
    - MACD (Moving Average Convergence Divergence)

    Args:
        df (pd.DataFrame): DataFrame con datos de precios (OHLC). Debe contener
                           las columnas 'High', 'Low', y 'Close'.
        params (dict): Diccionario con los parámetros (ventanas, etc.) para cada indicador.

    Returns:
        pd.DataFrame: Una copia del DataFrame original con las nuevas columnas de
                      indicadores, eliminando las filas que puedan tener valores NaN
                      resultantes del cálculo de las ventanas.
    """
    df_copy = df.copy()

    # RSI
    df_copy['rsi'] = ta.momentum.RSIIndicator(
        close=df_copy['Close'], window=params['rsi_window']
    ).rsi()

    # Bandas de Bollinger
    bollinger = ta.volatility.BollingerBands(
        close=df_copy['Close'], window=params['bb_window'], window_dev=2
    )
    df_copy['bb_high'] = bollinger.bollinger_hband()
    df_copy['bb_low'] = bollinger.bollinger_lband()

    # Oscilador Estocástico
    stochastic = ta.momentum.StochasticOscillator(
        high=df_copy['High'], low=df_copy['Low'], close=df_copy['Close'],
        window=params['stoch_window'], smooth_window=params['stoch_smooth_k']
    )
    df_copy['stoch_k'] = stochastic.stoch()

    # MACD
    macd = ta.trend.MACD(
        close=df_copy['Close'], window_slow=params['macd_long_window'],
        window_fast=params['macd_short_window'], window_sign=params['macd_signal_window']
    )
    df_copy['macd_line'] = macd.macd()
    df_copy['macd_signal_line'] = macd.macd_signal()

    return df_copy.dropna()


def generate_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Genera señales de compra ('buy_signal') y venta ('sell_signal') basadas
    en el consenso de múltiples indicadores.

    Una señal se activa si al menos 2 de los 4 indicadores (RSI, BB, Stoch, MACD)
    están de acuerdo en la dirección del mercado.

    Args:
        df (pd.DataFrame): DataFrame que ya contiene las columnas de indicadores
                           calculadas por `calculate_indicators`.
        params (dict): Diccionario con los umbrales para generar las señales
                       de cada indicador (ej. 'rsi_lower', 'stoch_buy_th').

    Returns:
        pd.DataFrame: El DataFrame con dos nuevas columnas booleanas:
                      'buy_signal' y 'sell_signal', que son `True` en la vela
                      donde se activa una señal.
    """
    df_copy = df.copy()

    # --- Señales Individuales de cada Indicador ---
    rsi_buy = df_copy['rsi'] < params['rsi_lower']
    rsi_sell = df_copy['rsi'] > params['rsi_upper']

    bb_buy = df_copy['Close'] < df_copy['bb_low']
    bb_sell = df_copy['Close'] > df_copy['bb_high']

    stoch_buy = df_copy['stoch_k'] < params['stoch_buy_th']
    stoch_sell = df_copy['stoch_k'] > params['stoch_sell_th']

    macd_buy = df_copy['macd_line'] > df_copy['macd_signal_line']
    macd_sell = df_copy['macd_line'] < df_copy['macd_signal_line']

    # --- Consenso de Señales (2 de 4) ---
    buy_signals = (
            rsi_buy.astype(int) +
            bb_buy.astype(int) +
            stoch_buy.astype(int) +
            macd_buy.astype(int)
    )
    sell_signals = (
            rsi_sell.astype(int) +
            bb_sell.astype(int) +
            stoch_sell.astype(int) +
            macd_sell.astype(int)
    )

    df_copy['buy_signal'] = buy_signals >= 2
    df_copy['sell_signal'] = sell_signals >= 2

    return df_copy

