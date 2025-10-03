import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_performance_vs_buy_and_hold(strategy_portfolio, price_data, initial_cash):
    """Añadir documentacion"""
    buy_and_hold_shares = initial_cash / price_data.iloc[0]
    buy_and_hold_portfolio = buy_and_hold_shares * price_data

    plt.figure(figsize=(15, 7))
    strategy_portfolio.plot(label='Estrategia Optimizada', color='blue')
    buy_and_hold_portfolio.plot(label='Comprar y Mantener (Buy and Hold)', color='gray', linestyle='--')
    plt.title('Estrategia Optimizada vs. Comprar y Mantener (Test + Validation)')
    plt.xlabel('Fecha')
    plt.ylabel('Valor del Portafolio (USD)')
    plt.legend()
    plt.show()


def plot_split_performance(test_portfolio, validation_portfolio):
    """Añadir documentacion"""
    plt.figure(figsize=(15, 7))
    test_portfolio.plot(label='Test', color='orange')
    validation_portfolio.plot(label='Validation', color='green')
    plt.title('Rendimiento del Portafolio (Test + Validation)')
    plt.xlabel('Fecha')
    plt.ylabel('Valor del Portafolio (USD)')
    plt.legend()
    plt.show()


def plot_training_performance(portfolio_values):
    """Añadir documentacion"""
    plt.figure(figsize=(15, 7))
    portfolio_values.plot(title='Rendimiento en Entrenamiento')
    plt.xlabel('Fecha')
    plt.ylabel('Valor del Portafolio (USD)')
    plt.show()