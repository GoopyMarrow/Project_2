import pandas as pd
from dataclasses import dataclass

@dataclass
class Operation:
    """Añadir documentacion"""
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