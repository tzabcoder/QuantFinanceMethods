"""
# Brownian Motion Overview
#
# 1. Drift = (mean - 0.5 * Var)
# 2. Volatility = sigma * Z * [Rand(0,1)]
#
# Thus the time series equation is
# P[t] = P[t-1] * e ^ u
#
# Where,
# u = Drift + Volatility
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import norm, gmean, cauchy

warnings.filterwarnings('ignore')

# Calculate the daily logarithmic returns
def LogReturns(data):
    return np.log(1 + data.pct_change())

# Calculate the daily simple returns
def SimpleReturns(data):
    return data.pct_change()

# Calculate the drift component for the Brownian Motion
def Drift(data, r_type='log'):
    if r_type == 'log':
        returns = LogReturns(data)

    elif r_type == 'simple':
        returns = SimpleReturns(data)

    # Invalid return type calculation
    else:
        return -1

    u = returns.mean()
    var = returns.var()

    drift = u - (0.5 * var)
    return drift

# Simulate the daily returns for each day into the future
# days = number of days into the future we want to predict (number of rows)
# itter = number of predictions to compute (columns)
def SimulatedDailyReturns(data, days, itter, r_type='log'):
    drift = Drift(data, r_type)

    if r_type == 'log':
        stddev = LogReturns(data).std()

    elif r_type == 'simple':
        stddev = SimpleReturns(data).std()

    # Invalid return type calculation
    else:
        return -1

    ret = np.exp(drift + stddev * norm.ppf(np.random.rand(days, itter)))
    return ret

# Calculates the probability that a stock will be above a certian threshold
# predicted = predicted prices
# threshold = threshold to compute the probability that the price will be above or below
# on = 'value' to calculate probability on prices, 'return' to calculate probability on returns
def Probability(predicted, threshold, on='value'):
    if on == 'return':
        p_0 = predicted.iloc[0,0]
        p = predicted.iloc[-1]
        p_list = list(p)

        above = []
        below = []

        for p_n in p_list:
            r = (p_n - p_0) / p_0

            if r >= threshold:
                above.append(r)
            else:
                below.append(r)

    elif on == 'value':
        p = predicted.iloc[-1]
        p_list = list(p)

        above = [i for i in p_list if i >= threshold]
        below = [i for i in p_list if i < threshold]

    # Invalid prediction method
    else:
        return -1

    return (len(above) / (len(above) + len(below)))

def SimulateMonteCarlo(data, days, iterations, r_type, plot=True, log=True):
    # Generate the simulated daily returns
    returns = SimulatedDailyReturns(data, days, iterations, r_type)

    prices = np.zeros_like(returns) # Create empty price matrix with same dimensionality as returns
    prices[0] = data.iloc[-1]       # Put last actual price in first row

    # Calculate daily prices
    for t in range(1, days):
        prices[t] = prices[t-1] * returns[t]

    priceDf = pd.DataFrame(prices)

    # Calculate expected data
    startPrice = priceDf.iloc[0,1]
    expectedPrice = priceDf.iloc[-1].mean()
    expectedReturn = ((expectedPrice - startPrice) / startPrice)
    expectedReturn_100 = expectedReturn * 100

    # Plot the end price distribution
    # Plot the cummulate distribution
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(14,4))
        sns.distplot(priceDf, ax=ax[0])
        sns.distplot(priceDf, hist_kws={'cumulative':True},kde_kws={'cumulative':True},ax=ax[1])
        plt.xlabel("Stock Price")
        plt.show()

        for name, data in priceDf.items():
            plt.plot(data)

        plt.show()

    # Print the information
    if log:
        print(priceDf)

        prob_0 = Probability(priceDf, 0, 'return')
        prob_5 = Probability(priceDf, 0.05, 'return')
        prob_10 = Probability(priceDf, 0.10, 'return')
        prob_25 = Probability(priceDf, 0.25, 'return')
        prob_50 = Probability(priceDf, 0.5, 'return')

        print(f"\n__________ Simulation {data.name} __________")
        print(f"Days:             {days}")
        print(f"Expected Price:   {expectedPrice}")
        print(f"Expected Return:  {round(expectedReturn_100, 4)}%")
        print(f"Probaility (0%):  {prob_0}")
        print(f"Probaility (5%):  {prob_5}")
        print(f"Probaility (10%): {prob_10}")
        print(f"Probaility (25%): {prob_25}")
        print(f"Probaility (50%): {prob_50}")

    return expectedPrice, expectedReturn

def main():
    TICKER = 'SPY'
    PERIOD = '5y'
    INTERVAL = '1d'

    data = yf.download(TICKER, period=PERIOD, interval=INTERVAL)['Adj Close']

    #SimulateMonteCarlo(data, 30, 1000, 'log', plot=True, log=True)

    signCorrect = []
    signIncorrect = []
    deviations = []

    # Rolling prediction vs actual performance
    for i in range(len(data)):
        if i >= 252 and i <= len(data)+30:
            X_train, Y_train = data.iloc[i-252:i], data.iloc[i:i+30]

            # Actual return and price data
            a_price, a_return = Y_train.iloc[-1], ((Y_train.iloc[-1] - Y_train.iloc[0]) / Y_train.iloc[0])

            # predicted return and price data
            p_price, p_return = SimulateMonteCarlo(X_train, 30, 1000, 'log', False, False)

            #print(f"Actual Ret: {a_return} || Predicted Ret: {p_return}")

            # Compare predicted return SIGN to actual return SIGN
            # This measures the "correctness" of the return direction prediction
            # Answers: How frequently are we predicting the direction?
            if np.sign(a_return) == np.sign(p_return):
                signCorrect.append(1)
            else:
                signIncorrect.append(1)

            # Measure the difference bewtween the predicted retun and actual return
            # This measures the "precision" of the return prediction
            # Answers: How accurate is the prediction?
            # NOTE:
            #   Negative deviation => predicted is HIGHER than actual
            #   Positive deviation => predicted is LOWER than actual
            deviation = a_return - p_return
            deviations.append(deviation)

    # Directionality measures the percent of correct SIGN predictions
    directionality = len(signCorrect) / (len(signCorrect) + len(signIncorrect))

    # Measures the average deviation between the actual and predicted returns
    mean_deviation = np.mean(deviations)

    print(f'__________ Performance {TICKER} __________')
    print(f"Directionality (%): {round(directionality*100, 4)}%")
    print(f"Mean deviation (predicted vs actual):     {mean_deviation}")

if __name__ == '__main__':
    main()
