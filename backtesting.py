import pandas as pd
from config import Operation
from get_signals import calculate_indicators, generate_signals


def run_backtest(df: pd.DataFrame, initial_cash: float, commission: float, params: dict):
    """
    Ejecuta una simulación de trading (backtest) sobre un set de datos, aplicando
    una estrategia basada en los parámetros proporcionados.

    Args:
        df (pd.DataFrame): Datos históricos de precios con un índice de tipo Datetime.
        initial_cash (float): Capital inicial para la simulación.
        commission (float): Costo de comisión por operación (ej. 0.00125 para 0.125%).
        params (dict): Diccionario completo con los hiperparámetros de la estrategia
                     (ventanas de indicadores, SL/TP, pct_cash, etc.).

    Returns:
        tuple: Una tupla conteniendo:
            - cash (float): El efectivo final después de la simulación.
            - portfolio_df (pd.Series): La serie temporal del valor del portafolio.
            - closed_trades_log (list): Una lista de objetos `Operation` cerrados.
            - active_positions (list): Una lista de objetos `Operation` que quedaron abiertos.
    """
    # --- 1. Inicialización de variables ---
    cash = initial_cash
    active_positions: list[Operation] = []
    closed_trades_log: list[Operation] = []
    portfolio_history = [{'timestamp': df.index[0], 'value': initial_cash}]

    # --- 2. Pre-cálculo de señales para eficiencia ---
    df_with_signals = generate_signals(calculate_indicators(df, params), params)
    pct_cash = params['pct_cash']

    # --- 3. Simulación ---
    for timestamp, row in df_with_signals.iterrows():
        current_price = row['Close']

        # --- 3.1. Cierre de Posiciones Abiertas ---
        # Se itera sobre una copia para poder modificar la lista original.
        for position in active_positions[:]:
            # Cierre de posiciones LONG por SL/TP
            if position.type == 'LONG' and (
                    current_price >= position.take_profit or current_price <= position.stop_loss):
                cash += current_price * position.n_shares * (1 - commission)
                position.status = 'CLOSED'
                position.pnl = (current_price - position.open_price) * position.n_shares
                closed_trades_log.append(position)
                active_positions.remove(position)
            # Cierre de posiciones SHORT por SL/TP
            elif position.type == 'SHORT' and (
                    current_price <= position.take_profit or current_price >= position.stop_loss):
                pnl = (position.open_price - current_price) * position.n_shares * (1 - commission)
                initial_margin = position.open_price * position.n_shares * (1 + commission)
                cash += pnl + initial_margin
                position.status = 'CLOSED'
                position.pnl = pnl
                closed_trades_log.append(position)
                active_positions.remove(position)

        # --- 3.2. Chequeo de Salud del Portafolio ---
        long_value = sum(p.n_shares * current_price for p in active_positions if p.type == 'LONG')
        current_equity = cash + long_value
        if current_equity <= 0:
            portfolio_history.append({'timestamp': timestamp, 'value': current_equity})
            continue

        # --- 3.3. Apertura de Nuevas Posiciones ---
        n_shares = (cash * pct_cash) / current_price
        if n_shares > 0:
            # Apertura de posición LONG
            cost_of_long = current_price * n_shares * (1 + commission)
            if row['buy_signal'] and cash >= cost_of_long:
                cash -= cost_of_long
                active_positions.append(Operation(
                    open_time=timestamp, open_price=current_price, n_shares=n_shares, type='LONG',
                    stop_loss=current_price * (1 - params['stop_loss']),
                    take_profit=current_price * (1 + params['take_profit'])
                ))
            # Apertura de posición SHORT
            elif row['sell_signal']:
                position_margin = current_price * n_shares * (1 + commission)
                if cash >= position_margin:
                    cash -= position_margin
                    active_positions.append(Operation(
                        open_time=timestamp, open_price=current_price, n_shares=n_shares, type='SHORT',
                        stop_loss=current_price * (1 + params['stop_loss']),
                        take_profit=current_price * (1 - params['take_profit'])
                    ))

        # --- 3.4. Cálculo del Valor del Portafolio para el Histórico ---
        final_long_value = sum(p.n_shares * current_price for p in active_positions if p.type == 'LONG')
        short_equity = sum(
            (pos.open_price * pos.n_shares) + ((pos.open_price - current_price) * pos.n_shares)
            for pos in active_positions if pos.type == 'SHORT'
        )
        portfolio_value = cash + final_long_value + short_equity
        portfolio_history.append({'timestamp': timestamp, 'value': portfolio_value})

    # --- 4. Liquidación Final de Posiciones Abiertas ---
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

    # --- 5. Retorno de Resultados ---
    portfolio_df = pd.DataFrame(portfolio_history).set_index('timestamp')['value']
    return cash, portfolio_df, closed_trades_log, active_positions