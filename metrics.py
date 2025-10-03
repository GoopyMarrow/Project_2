import pandas as pd
import numpy as np


def calculate_calmar_for_optimization(portfolio_values: pd.Series) -> float:
    """Añadir documentacion"""
    if portfolio_values.empty or portfolio_values.nunique() < 2:
        return -1.0
    returns = portfolio_values.pct_change().dropna()
    if returns.empty:
        return -1.0

    annualized_return = returns.mean() * (24 * 365)
    cumulative_max = portfolio_values.cummax()
    drawdown = (cumulative_max - portfolio_values) / cumulative_max
    max_drawdown = drawdown.max()
    return annualized_return / max_drawdown if max_drawdown > 0 else -1.0


def calculate_full_performance_metrics(portfolio_values: pd.Series, trades_log: list, time_frame_minutes: int) -> dict:
    """Añadir documentacion"""
    if portfolio_values.empty or len(trades_log) == 0:
        return {'Calmar Ratio': 0.0, 'Sharpe Ratio': 0.0, 'Sortino Ratio': 0.0, 'Max Drawdown': 0.0, 'Win Rate': 0.0,
                'Total Trades': 0, 'Annualized Return': 0.0}

    returns = portfolio_values.pct_change().dropna()
    if returns.empty:
        return {'Calmar Ratio': 0.0, 'Sharpe Ratio': 0.0, 'Sortino Ratio': 0.0, 'Max Drawdown': 0.0, 'Win Rate': 0.0,
                'Total Trades': len(trades_log), 'Annualized Return': 0.0}

    bars_per_year = (24 * 60 / time_frame_minutes) * 365
    annualized_return = returns.mean() * bars_per_year
    annualized_std_dev = returns.std() * np.sqrt(bars_per_year)
    sharpe_ratio = annualized_return / annualized_std_dev if annualized_std_dev != 0 else 0.0
    downside_returns = returns[returns < 0]
    annualized_downside_risk = (downside_returns.std() if not downside_returns.empty else 0) * np.sqrt(bars_per_year)
    sortino_ratio = annualized_return / annualized_downside_risk if annualized_downside_risk != 0 else 0.0
    cumulative_max = portfolio_values.cummax()
    drawdown = (cumulative_max - portfolio_values) / cumulative_max
    max_drawdown = drawdown.max()
    win_rate = sum(1 for trade in trades_log if trade.pnl > 0) / len(trades_log) if trades_log else 0.0
    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0

    metrics = {
        'Annualized Return': annualized_return, 'Calmar Ratio': calmar_ratio,
        'Sharpe Ratio': sharpe_ratio, 'Sortino Ratio': sortino_ratio,
        'Max Drawdown': max_drawdown, 'Win Rate': win_rate, 'Total Trades': len(trades_log)
    }
    return {key: round(value, 4) for key, value in metrics.items()}