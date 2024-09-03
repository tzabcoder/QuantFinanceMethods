import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

def main():
    data = yf.download(tickers=['SCHD'], period='max', interval='1d')
    data['Daily_Return'] = data['Adj Close'].pct_change()
    data.dropna(inplace=True)

    # Calculate and plot the cummulative return for SCHD
    data['Cumm_Return'] = (1 + data['Daily_Return']).cumprod() - 1

    plt.figure(figsize=(10, 6))
    plt.plot(data['Cumm_Return'])

    plt.title('SCHD Cummulative Returns')
    plt.xlabel('year')
    plt.ylabel('cummulative return')

    plt.show()

    years = (data.index.max() - data.index.min()).days / 365.25
    cagr = (data['Adj Close'].iloc[-1] / data['Adj Close'].iloc[0]) ** (1 / years) - 1

    print(f'Compound Annual Growth Rate: {cagr:.2%}')

    # Calculate and plot the monthly returns for SCHD
    monthly_returns = data['Adj Close'].resample('M').last()
    monthly_returns = monthly_returns.pct_change()
    monthly_returns.dropna(inplace=True)

    plt.figure(figsize=(10, 6))
    monthly_returns.plot(kind='bar', color='skyblue', edgecolor='black')

    plt.title('SCHD Monthly Returns')
    plt.xlabel('month')
    plt.ylabel('return')

    plt.xticks(rotation=45)

    plt.show()

    # Calculate the monthly return distribution (histogram)
    print(monthly_returns.describe())

    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(1, 1, 1)

    monthly_returns.hist(bins=25, ax=ax)
    ax.set_xlabel('return')
    ax.set_ylabel('frequency')
    ax.set_title('SCHD Monthly Return Distribution')

    plt.show()

    # Calculate the Probability Plot
    """
    * The probability plot assesses whether a dataset follows a given theoretical distribution (in our case
    * a normal distribution). The theoretical distribution is a straight line. Variability/departures from the
    * straight line are departures from the specified distribution. The correlation measure is the goodness of
    * fit measure for the data and the theoretical distribution.
    *
    * The Shapiro Test is a hypothesis test for normally distributed data. The null hypothesis is that the data
    * is normally distributed. A large p-value indicates that the data is normally distributed, whereas a low
    * p-value indicates that the data is not normally distributed.
    *
    * The result of the Shapiro test is:
    *       (test statistic, p-value)
    *
    * Generally, if the p-value is lower than 0.05 (5%), we reject the null hypothesis (that the data is normally
    * distributed).
    """
    print(f"Monthly Return Shapiro Test: {stats.shapiro(monthly_returns)}")
    print(f"Daily Return Shapiro Test: {stats.shapiro(data['Daily_Return'])}")

    # Monthly Return Probability Plot
    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(1, 1, 1)
    stats.probplot(monthly_returns, dist='norm', plot=ax)
    plt.title('Monthly Return: Probability Plot')
    plt.show()

    # Daily Return Probability Plot
    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(1, 1, 1)
    stats.probplot(data['Daily_Return'], dist='norm', plot=ax)
    plt.title('Daily Return: Probability Plot')
    plt.show()

if __name__ == '__main__':
    main()