import pandas as pd
from dataclasses import dataclass

@dataclass
class Operation:
    """
    Representa una única operación de trading, almacenando todos sus detalles
    desde la apertura hasta el cierre.

    Attributes:
        open_time (pd.Timestamp): Fecha y hora de apertura de la operación.
        open_price (float): Precio de entrada.
        n_shares (float): Cantidad de activo comprado/vendido.
        type (str): Tipo de operación, 'LONG' o 'SHORT'.
        stop_loss (float): Precio al que se activa el stop-loss.
        take_profit (float): Precio al que se activa el take-profit.
        status (str): Estado actual de la operación, 'OPEN' o 'CLOSED'.
        close_time (pd.Timestamp): Fecha y hora de cierre.
        close_price (float): Precio de salida.
        pnl (float): Ganancia o pérdida neta de la operación.
    """
    open_time: pd.Timestamp
    open_price: float
    n_shares: float
    type: str
    stop_loss: float
    take_profit: float
    status: str = 'OPEN'
    close_time: pd.Timestamp = None
    close_price: float = None
    pnl: float = 0.0

