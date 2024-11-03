# Trends
This repo contans quantitative methods for time-series trends.

## trendLabeling.py
This program implements the trend labeling technique from the book ***Machine Learning for Asset Managers*** by Marcos M. Lopex de Prado.
The labeling method uses linear regression and a ***span*** (look-forward window) to calculate the linear regression and extract the t-statistics for the period. Since different values of L lead to different t-stats,
different values of L are chosen to maximize the t-stat.

This program labels SPY data over a one year peiod. As seen in the graph, the results are sub-optimal due to the high volatility over the period.

Daily trend over the previous year<br/>
![TrendLabeling](https://github.com/tzabcoder/QuantFinanceMethods/assets/60833046/55cea28c-248f-4111-a22d-2137c9d652c0)

Daily trend on minute data<br/>
![SPY_Minute_Tick_Trend](https://github.com/tzabcoder/QuantFinanceMethods/assets/60833046/9deaf82d-5f47-4c8a-a4b3-3fe71dfee65b)

Distribution of minute t-values<br/>
![TValDistribution](https://github.com/tzabcoder/QuantFinanceMethods/assets/60833046/368dee9c-92e3-4f24-bd63-e2d6b96454fa)

---
## trendPredictions.py
This program implements a vectorized backtest of a trend prediction strategy using OLS regression. The data us extracted for ***SPY***, which is the ETF modeling the S&P 500. Log returns are calculated for the index to be used direcly in the autoregression.
The data is lagged N times (in this example 2). Each lag is the indepent variable for the multi-autoregression. Direction is determined as (+1, 0, -1) for a positive, neutral, or negative daily return, respectively. Given the lagged inputs and the direction as
the dependent variable, the model is fitted as a multiple autoregression. The train data is then used to predict the directionality of the test data.

Distribution of daily log PY returns<br/>
!
