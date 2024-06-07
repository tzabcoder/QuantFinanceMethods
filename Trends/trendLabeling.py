"""
* This program implements the "Trend-Scanning Method" resultslined in the book
* "Machine Learning for Asset Managers" by Marcos M. Lopez de Prado.
*
* File Description:
* The idea of this proposed dataset labeling technique is to identify trends
* and let them run as long as they persist, withresults setting explicit barriers.
*
* Given a series {x}t->T, where xt may represent the price of a sercurity under
* observation, we want to assign a label yt in {-1, 0, 1} for every observation in xt.
* yt = -1 => downtrend
* yt = 0  => no trend
* yt = 1  => uptrend
*
* The proposed method in the book was to compute the t-value associated with the estimated
* regressor coefficient in a linear time-trend model. We pick the value L that maximizes
* the t-value (label xt according to the most statistically significant trend observed in
* the future), results of possible look-forward periods.
"""

# File Imports
import yfinance as yf
import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt

"""
* tValLinReg()
*
* Description:
* This function calculates the linear regression for the data provided. The
* function returns the t-statistics from the linear model.
* @param[in] close - dataframe of data to fit the linear model
* @return tvalues(np.array) float t-statistics
"""
def tValLinReg(close):
    # Obtain the t-stat from a linear trend
    x = np.ones((close.shape[0], 2))   # Create matrix of ones
    x[:,1] = np.arange(close.shape[0]) # Evenly space values

    ols = sm.OLS(close, x).fit() # Calculate linear regression

    return ols.tvalues[1] # return the t-statistics

"""
* trendScanning()
*
* Description:
* Derives the labels from the sign of the computed t-value (see file description above)
* Given the set of values L (span), the algorithm will compute the maximum abs(t-value.
* @param[in] obsIdx - index of observations to be labeled
* @param[in] close  - time series of {x}t
* @param[in] span   - set of values L
* @return results(pd.DataFrame) -
*           index = timestamp of {x}t
*           t1    = reports timestamp of farthest observation used to find the most significant trend
*           tVal  = reports the t-value associated with the most significant linear trend
*           tSign = Sign of the trend
"""
def trendScanning(obsIdx, close, span):
    dfColumns = ['t1', 'tVal', 'tSign', 'windowSize']

    # Construct the results dataframe
    results = pd.DataFrame(index=obsIdx, columns=dfColumns)

    horizons = range(*span) # unpack span (start, stop, step)

    windowSize = span[1] - span[0] # stop - start
    maxWindow = span[1] - 1        # stop - 1
    minWindow = span[0]            # start

    for idx in close.index:
        idx += maxWindow

        # Only process when the index is in frame
        if idx < len(close):
            dfTVal = pd.Series(dtype='float64')
            closeIloc = close.index.get_loc(idx)

            for horizon in horizons:
                dt1 = close.index[closeIloc - horizon + 1]
                df1 = close.iloc[dt1 : idx]

                dfTVal.loc[dt1] = tValLinReg(df1.values) # Calculates the T-statistic on the period data

            dt1 = dfTVal.replace([-np.inf, np.inf, np.nan], 0).abs().idxmax() # Obtain the largtest T-stat over period

            results.loc[idx, dfColumns] = dfTVal.index[-1], dfTVal[dt1], np.sign(dfTVal[dt1]), abs(dfTVal.values).argmax() + minWindow

        # Index is results of frame
        else:
            break

    results['t1'] = pd.to_datetime(results['t1'])
    results['tSign'] = pd.to_numeric(results['tSign'], downcast='signed')

    # Remove t-stat resultsliers
    tMax = 20
    tValVariance = results['tVal'].values.var()

    if tValVariance < tMax:
        tMax = tValVariance

    # Remove max t-stats from resultsput
    results.loc[results['tVal'] > tMax, 'tVal'] = tMax               # Cutoff => tStats > tMax
    results.loc[results['tVal'] < (-1) * tMax, 'tVal'] = (-1) * tMax # Cutoff => tStats < -tMax

    return results.dropna(subset=['tSign'])

def main():
    START = '2023-01-01'
    END = '2024-01-01'
    TICKER = 'AAPL'

    # Download ticker data (using only close data)
    data = yf.download(TICKER, start=START, end=END)
    data.reset_index(inplace=True)
    dates = data['Date']
    data = data['Close']

    # Create future period span
    idxFrom = 3
    idxTo = 10
    span = [idxFrom, idxTo, 1]

    labeledData = trendScanning(data.index, data, span) # Label the data
    tStatistics = labeledData['tVal'].values

    # Plot labeled data
    plt.scatter(labeledData.index, data.loc[labeledData.index].values, c=tStatistics, cmap='viridis')
    plt.plot(data.index, data.values, color='gray')
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    main()
