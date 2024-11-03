import warnings
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

def create_lags(data, lags):
    """
    * Creates the lagged columns for the data. Eached lagged column is
    * saved in the dataframe. The lagged column names are returned for later
    * use.
    """

    cols = []
    for lag in range(1, lags+1):
        col = f'lag_{lag}'
        data[col] = data['Log_Returns'].shift(lag)
        cols.append(col)

    return cols

def main():
    data = yf.download('SPY', period='max', interval='1d')

    data['Returns'] = data['Adj Close'].pct_change()
    data['Log_Returns'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    data.dropna(inplace=True)

    # Obtain the sign of the daily return direction
    data['Direction'] = np.sign(data['Log_Returns']).astype(int)

    # Create a frequency distribution of daily log returns
    plt.hist(data['Log_Returns'], bins=40)
    plt.title('SPY Log Return Distribution')
    plt.show()

    # Create the lags for the autoregression
    cols = create_lags(data, 2)

    # Clean the lagged data
    data.dropna(inplace=True)

    # ----------------------------------------------------------------
    # Autoregression on the directions and lagged returns
    model = LinearRegression()

    N = 0.8
    T = int(N * len(data))

    # Split data (using a sequential data split)
    train, test = data.iloc[0:T+1], data.iloc[T+1:len(data)+1]

    # .fit() fits the LinearRegression model to the data
    # X = Lagged log returns are the independet variables
    # Y = Direction (-1 or +1) of the daily returns

    # .predict() predicts the dependent variable using the fitted model

    direction_ols = model.fit(train[cols], train['Direction'])

    test['direction_ols'] = direction_ols.predict(test[cols])

    test['direction_ols'] = np.where(test['direction_ols'] > 0, 1, -1)

    # Calculate the strategy returns
    test['direction_strat'] = test['direction_ols'] * test['Log_Returns']

    # Calculate cummulative returns
    test['cumm_reg_strat'] = (1 + test['direction_strat']).cumprod() - 1
    test['cumm_market'] = (1 + test['Log_Returns']).cumprod() - 1

    # ----------------------------------------------------------------
    # K-means clustering on the directions and lagged returns
    model = KMeans(n_clusters=2, random_state=0)

    model.fit(train[cols])

    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
        n_clusters=2, n_init=10, random_state=0, tol=0.0001, verbose=0)

    # Predict the direction based on the K-means model
    test['pos_clus'] = model.predict(test[cols])
    test['pos_clus'] = np.where(test['pos_clus'] == 1, 1, -1)

    # Plot the 2 groups
    plt.figure(figsize=(10, 6))
    plt.scatter(test[cols].iloc[:, 0], test[cols].iloc[:, 1],
        c=test['pos_clus'], cmap='coolwarm')
    plt.show()

    # Calculate the strategy's returns
    test['cluster_strat'] = test['pos_clus'] * test['Log_Returns']
    test['cumm_cluster_strat'] = (1 + test['cluster_strat']).cumprod() - 1

    # Plot the strategy's results vs market value
    plt.plot(test['cumm_reg_strat'], label='Regression')
    plt.plot(test['cumm_cluster_strat'], label='Cluster')
    plt.plot(test['cumm_market'], label='Market')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
