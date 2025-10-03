import pandas as pd
import ta

def calculate_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Añadir documentacion"""
    df_copy = df.copy()
    df_copy['rsi'] = ta.momentum.RSIIndicator(df_copy['Close'], window=params['rsi_window']).rsi()
    bollinger = ta.volatility.BollingerBands(df_copy['Close'], window=params['bb_window'], window_dev=2)
    df_copy['bb_high'] = bollinger.bollinger_hband()
    df_copy['bb_low'] = bollinger.bollinger_lband()
    stochastic = ta.momentum.StochasticOscillator(
        df_copy['High'], df_copy['Low'], df_copy['Close'],
        window=params['stoch_window'], smooth_window=params['stoch_smooth_k']
    )
    df_copy['stoch_k'] = stochastic.stoch()
    macd = ta.trend.MACD(
        df_copy['Close'], window_slow=params['macd_long_window'],
        window_fast=params['macd_short_window'], window_sign=params['macd_signal_window']
    )
    df_copy['macd_line'] = macd.macd()
    df_copy['macd_signal_line'] = macd.macd_signal()
    return df_copy.dropna()

def generate_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Añadir documentacion"""
    df_copy = df.copy()
    rsi_buy = df_copy['rsi'] < params['rsi_lower']
    rsi_sell = df_copy['rsi'] > params['rsi_upper']
    bb_buy = df_copy['Close'] < df_copy['bb_low']
    bb_sell = df_copy['Close'] > df_copy['bb_high']
    stoch_buy = df_copy['stoch_k'] < params['stoch_buy_th']
    stoch_sell = df_copy['stoch_k'] > params['stoch_sell_th']
    macd_buy = df_copy['macd_line'] > df_copy['macd_signal_line']
    macd_sell = df_copy['macd_line'] < df_copy['macd_signal_line']

    buy_signals = (rsi_buy.astype(int) + bb_buy.astype(int) + stoch_buy.astype(int) + macd_buy.astype(int))
    sell_signals = (rsi_sell.astype(int) + bb_sell.astype(int) + stoch_sell.astype(int) + macd_sell.astype(int))

    df_copy['buy_signal'] = buy_signals >= 2
    df_copy['sell_signal'] = sell_signals >= 2
    return df_copy