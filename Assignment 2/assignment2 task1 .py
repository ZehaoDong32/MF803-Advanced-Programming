"""
Assignment2 task1
Title:Sector ETF Factor Modeling
Author: Zehao Dong
Email: zehao@bu.edu
"""
import pandas as pd
import pandas_datareader.data as web
import statsmodels.api as sm
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns


def daily_return(df):
    r = df.pct_change()
    return r


def cov_daily(df):
    """
    Daily covariance
    """
    covariance = df.cov()
    return covariance


def correlation(df):
    cor = df.corr()
    return cor


def rolling_correlation(days, df):
    i = 0
    result = pd.DataFrame()
    while i < 2:
        j = i + 1
        while j < 3:
            corr = df.iloc[:, j].rolling(days).corr(df.iloc[:, i]).dropna()
            result = pd.concat([result, corr], axis=1)
            j += 1
        i += 1
    result.columns = ['Mkt-RF&SMB', 'Mkt-RF&HML', 'SML&HML']
    return result


def beta(y, df1, df2):
    x1 = df1
    y1 = df2[y]
    model = sm.OLS(y1, x1, missing='drop')
    result = model.fit()
    return list(result.params[0:3].values)


def beta_all(df1, df2):
    beta_dict = {}
    for c in df2.columns:
        beta_dict[c] = beta(c, df1, df2)
    return pd.DataFrame.from_dict(beta_dict, orient='index')


def rolling_beta(y, days, df1, df2):
    i = 0
    result = []
    while i < len(df2) - days:
        result.append(beta(y, df1.iloc[i: i + days, :], df2.iloc[i: i + days, :]))
        i += 1
    return pd.DataFrame(result)


def residual(name, df, df2):
    model = sm.OLS(df2[name], df, missing='drop').fit()
    e = model.resid
    return e


def qqplot(name, d, d2):
    model = sm.OLS(d2[name], d, missing='drop').fit()
    sm.qqplot(model.resid, fit=True, line='s')
    plt.title('qqplot for ' + name)
    plt.show()


def residual_vs_fit(name, d, d2):
    model = sm.OLS(d2[name], d, missing='drop').fit()
    e = model.resid
    fit = model.fittedvalues
    df = pd.DataFrame()
    df = pd.concat([e, fit], axis=1)
    df.columns = ['Residuals', 'Fitted_values']
    plt.figure(figsize=(8, 6), dpi=120)
    sns.residplot('Fitted_values', 'Residuals', data=df, lowess=True, scatter_kws={'alpha': 0.5},
                  line_kws={'color': 'red', 'lw': 1.5})
    plt.title('Residuals versus Fits for ' + name)
    plt.show()


# Download data of FF_factors
data = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start='2010-01-01',
                      end='2019-07-31')
FF_factors = pd.DataFrame()
FF_factors = data[0].iloc[:, 0:3]

# Download data of ETFs
tickers = 'SPY XLB XLE XLF XLI XLK XLP XLU XLV XLY'
data2 = yf.download(tickers, '2010-01-01', '2019-07-31')
ETF = daily_return(data2['Close'])
ETF = ETF * 100

# Daily covariance matrix of the factor returns
cov = cov_daily(FF_factors)
print('Daily covariance matrix of the factor returns is:\n', cov)

# Correlations of the factor returns
corr = correlation(FF_factors)
print('Correlations of the factor returns are:\n', corr)

# Rolling correlation of the factor returns
rolling_corr = rolling_correlation(90, FF_factors)
print('Rolling correlation of the factor returns:\n', rolling_corr)
rolling_corr.plot()
plt.title('Rolling correlation')
plt.xlabel('date')
plt.ylabel('correlation')

# Check the factor returns for normality
for c in range(0, 3):
    sm.qqplot(FF_factors.iloc[:, c], fit=True, line='45')
    plt.show()

# Beta for each sector ETF
beta1 = beta_all(FF_factors, ETF)
beta1.columns = ['Mkt-RF', 'SMB', 'HML']
print('Betas for each sector ETF are:\n', beta1)

# Rolling beta for each sector ETF
SPY = rolling_beta('SPY', 90, FF_factors, ETF)
SPY.columns = ['Mkt-RF', 'SMB', 'HML']
SPY.plot(title='SPY')
XLB = rolling_beta('XLB', 90, FF_factors, ETF)
XLB.columns = ['Mkt-RF', 'SMB', 'HML']
XLB.plot(title='XLB')
XLE = rolling_beta('XLE', 90, FF_factors, ETF)
XLE.columns = ['Mkt-RF', 'SMB', 'HML']
XLE.plot(title='XLE')
XLF = rolling_beta('XLF', 90, FF_factors, ETF)
XLF.columns = ['Mkt-RF', 'SMB', 'HML']
XLF.plot(title='XLF')
XLI = rolling_beta('XLI', 90, FF_factors, ETF)
XLI.columns = ['Mkt-RF', 'SMB', 'HML']
XLI.plot(title='XLI')
XLK = rolling_beta('XLK', 90, FF_factors, ETF)
XLK.columns = ['Mkt-RF', 'SMB', 'HML']
XLK.plot(title='XLK')
XLP = rolling_beta('XLP', 90, FF_factors, ETF)
XLP.columns = ['Mkt-RF', 'SMB', 'HML']
XLP.plot(title='XLP')
XLU = rolling_beta('XLU', 90, FF_factors, ETF)
XLU.columns = ['Mkt-RF', 'SMB', 'HML']
XLU.plot(title='XLU')
XLV = rolling_beta('XLV', 90, FF_factors, ETF)
XLV.columns = ['Mkt-RF', 'SMB', 'HML']
XLV.plot(title='XLV')
XLY = rolling_beta('XLY', 90, FF_factors, ETF)
XLY.columns = ['Mkt-RF', 'SMB', 'HML']
XLY.plot(title='XLY')

# Analyze residuals
for c in ETF.columns:
    res = residual(c, FF_factors, ETF)
    print('mean of residual of', c, 'is', res.mean(), '\nvariance of residual of', c, 'is',
          res.var())

# qqplot for residuals
for c in ETF.columns:
    qqplot(c, FF_factors, ETF)

# residual_vs_fit plot
for c in ETF.columns:
    residual_vs_fit(c, FF_factors, ETF)
