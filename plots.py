# plots.py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Configuración global de estilo para todos los gráficos del módulo
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 5]
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.prop_cycle'] = plt.cycler(
    color=["skyblue", "steelblue", "navy", "royalblue", "mediumslateblue"]
)


def plot_performance_vs_buy_and_hold(strategy_portfolio: pd.Series, price_data: pd.Series, initial_cash: float):
    """
    Grafica el rendimiento de la estrategia optimizada contra la estrategia
    pasiva de Comprar y Mantener (Buy and Hold).

    Args:
        strategy_portfolio (pd.Series): Serie del valor del portafolio de la estrategia.
        price_data (pd.Series): Serie de precios de cierre del activo para el mismo periodo.
        initial_cash (float): Capital inicial del periodo para calcular la base del Buy and Hold.
    """
    buy_and_hold_shares = initial_cash / price_data.iloc[0]
    buy_and_hold_portfolio = buy_and_hold_shares * price_data

    plt.figure()
    strategy_portfolio.plot(label='Estrategia Optimizada')
    buy_and_hold_portfolio.plot(label='Comprar y Mantener (Buy and Hold)', color='gray', linestyle='--')
    plt.title('Estrategia Optimizada vs. Comprar y Mantener (Test + Validation)')
    plt.xlabel('Fecha')
    plt.ylabel('Valor del Portafolio (USD)')
    plt.legend()
    plt.show()


def plot_split_performance(test_portfolio: pd.Series, validation_portfolio: pd.Series):
    """
    Grafica el rendimiento del portafolio en los periodos de Test y Validation
    en un mismo gráfico para visualizar la continuidad.

    Args:
        test_portfolio (pd.Series): Serie del valor del portafolio en el set de prueba.
        validation_portfolio (pd.Series): Serie del valor del portafolio en el set de validación.
    """
    plt.figure()
    test_portfolio.plot(label='Test')
    validation_portfolio.plot(label='Validation')
    plt.title('Rendimiento del Portafolio (Test + Validation)')
    plt.xlabel('Fecha')
    plt.ylabel('Valor del Portafolio (USD)')
    plt.legend()
    plt.show()


def plot_training_performance(portfolio_values: pd.Series):
    """
    Grafica el rendimiento del portafolio durante el periodo de entrenamiento.

    Args:
        portfolio_values (pd.Series): Serie del valor del portafolio en el set de entrenamiento.
    """
    plt.figure()
    portfolio_values.plot(title='Rendimiento en Entrenamiento')
    plt.xlabel('Fecha')
    plt.ylabel('Valor del Portafolio (USD)')
    plt.show()

