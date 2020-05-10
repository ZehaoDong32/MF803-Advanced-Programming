#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assignment1 task1
Title:Exotic Option Pricing via Simulation
Author: Zehao Dong
Email: zehao@bu.edu
"""
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import yfinance as yf

class ETF(object):
    def __init__(self, tickers, start, end):
        self.tickers = tickers
        self.start = start
        self.end = end
        #download data from yahoo finance
        self.data = yf.download(self.tickers, self.start, self.end)
        self.price_data = self.data['Close']
        self.price_start = self.price_data.iloc[0, :]
        self.price_end = self.price_data.iloc[-1, :]
        self.days = len(self.price_data)

    def annualized_return(self):
        ar = (self.price_end / self.price_start) ** (252 / self.days) - 1
        return ar

    def standard_deviation(self):
        std = self.price_data.std()
        return std

    def daily_return(self):
        daily_return = self.price_data.pct_change()
        return daily_return

    def monthly_return(self):
        monthly_data = self.price_data.resample('M').last()
        monthly_return = monthly_data.pct_change()
        return monthly_return

    def cov_daily(self):
        '''
        Daily covariance
        '''
        cov_daily= self.daily_return().cov()
        return cov_daily

    def cov_monthly(self):
        '''
        Monthly covariance
        '''
        cov_monthly = self.monthly_return().cov()
        return cov_monthly

    def corr_daily(self):
        '''
        Daily correlation
        '''
        corr_daily = self.daily_return().corr()
        return corr_daily

    def corr_monthly(self):
        '''
        Monthly correlation
        '''
        corr_monthly = self.monthly_return().corr()
        return corr_monthly

    def rolling_correlation(self, days, columns='SPY'):
        corr = self.price_data.rolling(days).corr(self.price_data[columns]).dropna()
        return corr

    def beta(self, y, df):
        '''
        Beta calculated by linear regression
        '''
        x1 = df['SPY']
        y1 = df[y]
        model = sm.OLS(y1, x1, missing='drop')
        result = model.fit()
        return result.params[0]

    def beta_all(self, df):
        '''
        All beta's for ETFs in tickers
        '''
        beta = {}
        for c in df.columns:
            if c != 'SPY':
                beta[c] = self.beta(c, df)
        return pd.DataFrame.from_dict(beta, orient='index').T


    def rolling_beta(self, y, days, df):
        i = 0
        result = []
        while i < len(df) - days:
            result.append(self.beta(y, df.iloc[i: i + days, :]))
            i += 1
        return pd.DataFrame(result, columns=[y])

    def rolling_beta_all(self, days, df):
        '''
        All rolling-beta's for ETFs in tickers
        '''
        beta = pd.DataFrame()
        for c in df.columns:
            if c != 'SPY':
                a = self.rolling_beta(c, days, df)
                beta = pd.concat([beta, a], axis=1)
        return beta

    def auto_correlation(self, x, df):
        x1 = df[x]
        df = df.shift(1)
        y1 = df[x]
        model = sm.OLS(y1, x1, missing='drop')
        result = model.fit()
        return result.params[0]

    def auto_correlation_all(self, df):
        '''
        All auto-correlation for ETFs in tickers
        '''
        corr = {}
        for c in df.columns:
            corr[c] = self.auto_correlation(c, df)
        return pd.DataFrame.from_dict(corr, orient='index').T

    def plot(self, df, title, x_label, y_label):
        plt.figure(figsize=(12, 8), dpi=120)
        df.plot()
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc = 'right')



#test
tickers = "SPY XLB XLE XLF XLI XLK XLP XLU XLV XLY"
start = "2010-01-01"
end = "2019-09-15"
test1 = ETF(tickers, start, end)
print('annualized_return:\n', test1.annualized_return())
print('standard deviation:\n', test1.standard_deviation())
print('covariance of daily return:\n', test1.cov_daily())
print('covariance of monthly return:\n', test1.cov_monthly())
print('rolling correlation:\n', test1.rolling_correlation(90))
test1.plot(test1.rolling_correlation(90),
           'Rolling 90-day correlation for ETFs', 
           'Date', 'Correlation')
print('beta:\n', test1.beta_all(test1.daily_return()))
print('rolling beta\n', test1.rolling_beta_all(90, test1.daily_return()))
test1.plot(test1.rolling_beta_all(90, test1.daily_return()),
           'Rolling 90-day beta for ETFs',
           'Date', 'Beta')
print('auto-correlation:\n', test1.auto_correlation_all(test1.daily_return()))