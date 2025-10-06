import pandas as pd
from config import Operation
from get_signals import calculate_indicators, generate_signals


def run_backtest(df: pd.DataFrame, initial_cash: float, commission: float, params: dict):
    """
    Ejecuta una simulación de trading (backtest) sobre un set de datos.

    Args:
        df (pd.DataFrame): Datos históricos de precios.
        initial_cash (float): Capital inicial.
        commission (float): Costo de comisión por operación (ej. 0.00125).
        params (dict): Diccionario completo con los parámetros de la estrategia.

    Returns:
        tuple: Una tupla conteniendo:
            - cash (float): El efectivo final.
            - portfolio_df (pd.Series): La serie del valor del portafolio a lo largo del tiempo.
            - closed_trades_log (list): Una lista de objetos `Operation` cerrados.
            - active_positions (list): Una lista de objetos `Operation` que quedaron abiertos.
    """
    cash = initial_cash
    active_positions: list[Operation] = []
    closed_trades_log: list[Operation] = []
    portfolio_history = [{'timestamp': df.index[0], 'value': initial_cash}]

    df_with_signals = generate_signals(calculate_indicators(df, params), params)

    for timestamp, row in df_with_signals.iterrows():
        current_price = row['Close']

        # --- Cierre de Posiciones ---
        for position in active_positions[:]:
            if position.type == 'LONG' and (
                    current_price >= position.take_profit or current_price <= position.stop_loss):
                cash += current_price * position.n_shares * (1 - commission)
                position.status = 'CLOSED'
                position.pnl = (current_price - position.open_price) * position.n_shares
                closed_trades_log.append(position)
                active_positions.remove(position)
            elif position.type == 'SHORT' and (
                    current_price <= position.take_profit or current_price >= position.stop_loss):
                pnl = (position.open_price - current_price) * position.n_shares * (1 - commission)
                initial_margin = position.open_price * position.n_shares * (1 + commission)
                cash += pnl + initial_margin
                position.status = 'CLOSED'
                position.pnl = pnl  # PnL ya incluye comisión en este caso
                closed_trades_log.append(position)
                active_positions.remove(position)

        # --- Chequeo Portafolio ---
        long_value = sum(p.n_shares * current_price for p in active_positions if p.type == 'LONG')
        current_equity = cash + long_value
        if current_equity <= 0:
            portfolio_history.append({'timestamp': timestamp, 'value': current_equity})
            continue

        # --- Apertura de Posiciones ---
        if params['n_shares'] > 0:
            cost_of_long = current_price * params['n_shares'] * (1 + commission)
            if row['buy_signal'] and cash >= cost_of_long:
                cash -= cost_of_long
                active_positions.append(Operation(
                    open_time=timestamp, open_price=current_price, n_shares=params['n_shares'], type='LONG',
                    stop_loss=current_price * (1 - params['stop_loss']),
                    take_profit=current_price * (1 + params['take_profit'])
                ))
            elif row['sell_signal']:
                position_margin = current_price * params['n_shares'] * (1 + commission)
                if cash >= position_margin:
                    cash -= position_margin
                    active_positions.append(Operation(
                        open_time=timestamp, open_price=current_price, n_shares=params['n_shares'], type='SHORT',
                        stop_loss=current_price * (1 + params['stop_loss']),
                        take_profit=current_price * (1 - params['take_profit'])
                    ))

        final_long_value = sum(p.n_shares * current_price for p in active_positions if p.type == 'LONG')
        short_equity = 0
        for pos in active_positions:
            if pos.type == 'SHORT':
                pnl = (pos.open_price - current_price) * pos.n_shares
                margin = pos.open_price * pos.n_shares
                short_equity += margin + pnl

        portfolio_value = cash + final_long_value + short_equity
        portfolio_history.append({'timestamp': timestamp, 'value': portfolio_value})

    # Liquidación al final del periodo
    last_price = df['Close'].iloc[-1]
    for pos in active_positions[:]:
        if pos.type == 'LONG':
            cash += last_price * pos.n_shares * (1 - commission)
        elif pos.type == 'SHORT':
            pnl = (pos.open_price - last_price) * pos.n_shares * (1 - commission)
            initial_margin = pos.open_price * pos.n_shares * (1 + commission)
            cash += pnl + initial_margin

    if portfolio_history:
        portfolio_history[-1]['value'] = cash

    portfolio_df = pd.DataFrame(portfolio_history).set_index('timestamp')['value']
    return cash, portfolio_df, closed_trades_log, active_positions