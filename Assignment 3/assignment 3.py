import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


class Index:
    def __init__(self, df):
        self.df = df

    def mean(self):
        m = self.df.mean()
        return m

    def auto_correlation(self, x):
        x1 = self.df[x]
        df = self.df.shift(1)
        y1 = df[x]
        model = sm.OLS(y1, x1, missing='drop')
        result = model.fit()
        return result.params[0]

    def __add__(self, other):
        merge = pd.merge(self.df, other.df, left_index=True, on='Date')
        return Index(merge)

    def to_monthly(self):
        df_m = self.df.resample('M').last()
        return Index(df_m)

    def correlation(self):
        cor = self.df.corr()
        return cor

    def rolling_correlation(self, days):
        corr = self.df.iloc[:, 0].rolling(days).corr(self.df.iloc[:, 1]).dropna()
        return Index(corr.to_frame())

    def daily_return(self):
        r = self.df.pct_change().dropna()
        return r

    def realized_vol(self, df):
        r = np.log(df) - np.log(df.shift(1))
        result = np.sqrt((r ** 2).sum() * 252 / len(df)) * 100
        return result

    def rolling_vol(self, days):
        i = days
        df = self.df
        result = []
        while i < len(df):
            result.append(self.realized_vol(df.iloc[i - days: i, :]))
            i += 1
        return pd.DataFrame(result)


def plot(df, title, x_label, y_label):
    df.plot(legend=0)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def BS_put(S, K, sigma, t, r):
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * t) / (sigma * t ** 0.5)
    d2 = (np.log(S / K) + (r - sigma ** 2 / 2) * t) / (sigma * t ** 0.5)
    Nd1 = stats.norm.cdf(-d1)
    Nd2 = stats.norm.cdf(-d2)
    price = Nd2 * K * np.exp(-r * t) - S * Nd1
    return price


def BS_call(S, K, sigma, t, r):
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * t) / (sigma * t ** 0.5)
    d2 = (np.log(S / K) + (r - sigma ** 2 / 2) * t) / (sigma * t ** 0.5)
    Nd1 = stats.norm.cdf(d1)
    Nd2 = stats.norm.cdf(d2)
    price = -Nd2 * K * np.exp(-r * t) + S * Nd1
    return price


# Read VIX data
data = pd.read_csv('vix.csv', index_col=0)
v = data.loc[:, ['VIX Close']]
v.index = pd.to_datetime(v.index)
v.columns = ['Implied vol']
VIX = Index(v)

# Download S&P data
data2 = yf.download('SPY', '2004-01-01', '2019-09-27')
s = data2.loc[:, ['Close']]
s.columns = ['SPY']
SPY = Index(s)

print('Auto-correlation for S&P:', SPY.auto_correlation('SPY'))
print('auto-correlation for VIX', VIX.auto_correlation('Implied vol'))

# Change data to monthly basis
M_SPY = SPY.to_monthly()
M_VIX = VIX.to_monthly()

# Merge VIX and S&P to one DataFrame
merge_daily = SPY + VIX
merge_monthly = M_SPY + M_VIX

print('Daily correlation is:', merge_daily.correlation())
print('Monthly correlation is:', merge_monthly.correlation())
rolling_corr = merge_daily.rolling_correlation(90)
plot(rolling_corr.df, 'Rolling 90-day correlation', 'Date', 'Correlation')
# Plot standard deviation of correlation
mean = rolling_corr.mean()
st_deviation = np.sqrt((rolling_corr.df - mean) ** 2)
plot(st_deviation, 'Standard Deviation of correlation', 'Date', 'Deviation')

# Rolling 90-day realized volatility
vol = SPY.rolling_vol(90)
vol.index = SPY.df.iloc[90:, :].index
vol.columns = ['realized vol']
plot(vol, 'Realized volatility in S&P', 'Date', 'Volatility')

# Premium
vol_merge = pd.merge(VIX.df.iloc[90:, :], vol, left_index=True, on='Date')
premium = vol_merge.iloc[:, 0] - vol_merge.iloc[:, 1]
premium = pd.DataFrame(premium, columns=['premium'])
plot(premium, 'Premium of implied vol. over realized vol.', 'Date', 'Premium')

# Straddle
BS_put(SPY.df.iloc[:, 0], SPY.df.iloc[:, 0], VIX.df.iloc[:, 0] / 100, 21 / 252, 0)
straddle = pd.merge(SPY.df, VIX.df / 100, left_index=True, on='Date')
straddle['call price'] = straddle.apply(lambda x: BS_call(x[0], x[0], x[1], 21 / 252, 0), axis=1)
straddle['put price'] = straddle.apply(lambda x: BS_put(x[0], x[0], x[1], 21 / 252, 0), axis=1)
plot(straddle.loc[:, ['call price']], 'Option prices', 'Date', 'Prices')
# Payoff and profit
straddle[['SPY_1M_later']] = straddle[['SPY']].shift(-21)
straddle['payoff'] = np.abs(straddle.iloc[:, 4] - straddle.iloc[:, 0])
straddle['profit'] = straddle.iloc[:, 5] - straddle.iloc[:, 2] - straddle.iloc[:, 3]
plot(straddle.loc[:, ['profit']], 'P&L of straddles', 'Date', 'P&L')
print('Average P&L of straddles:', straddle.loc[:, ['profit']].mean())

compare = pd.merge(straddle, premium, left_index=True, on='Date')
plt.scatter(compare.iloc[:, 6], compare.iloc[:, 7])
plt.title('P&L against premium')
plt.xlabel('Premium')
plt.ylabel('P&L')
plt.show()
y = compare.iloc[:, 6]
X = compare.iloc[:, 7]
model = sm.OLS(y, X, missing='drop').fit()
model.summary()
