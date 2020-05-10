"""
Assignment2 task2
Title:Exotic Option Pricing via Simulation
Author: Zehao Dong
Email: zehao@bu.edu
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm


def random_num(paths, mean, variance, t):
    result = []
    while paths > 0:
        path = np.random.normal(mean, variance ** 0.5, size=t * 252)
        result.append(path)
        paths -= 1
    return result


class Option:
    def __init__(self, t, s, sigma, r, k):
        self.t = t
        self.s = s
        self.sigma = sigma
        self.r = r
        self.k = k

    def simulation(self, w, e):
        size = self.t * 252
        result = [[0 for col in range(size + 1)] for row in range(len(w))]
        i = 0
        while i < len(w):
            s2 = self.s + e
            j = 0
            result[i][j] = s2
            j += 1
            while j < size + 1:
                s2 = self.r * (1 / 252) + self.sigma * w[i][j - 1] + s2
                result[i][j] = s2
                j += 1
            i += 1
        return pd.DataFrame(result).T

    def histogram(self, series, title, x_label='x', y_label='y'):
        plt.figure(figsize=(8, 6), dpi=120)
        series.plot.hist(bins=15, rwidth=0.95, color='#607c8e')
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)


class Put(Option):
    def __init__(self, t, s, sigma, r, k):
        Option.__init__(self, t, s, sigma, r, k)

    def price(self, df):
        price_min = df.min(axis=0).reset_index(drop=True)
        result = []
        for c in price_min:
            if c < self.k:
                result.append(self.k - c)
            else:
                result.append(0)
        payoff = pd.Series(result)
        d = np.exp(self.r * self.t)
        price = np.mean(payoff) * d
        return price

    def delta(self, w, e):
        c1 = self.simulation(w, e)
        c2 = self.simulation(w, -e)
        price1 = self.price(c1)
        price2 = self.price(c2)
        delta = (price1 - price2) / (2 * e)
        return delta


class Call(Option):
    def __init__(self, t, s, sigma, r, k):
        Option.__init__(self, t, s, sigma, r, k)

    def price(self, df):
        price_max = df.max(axis=0).reset_index(drop=True)
        result = []
        for c in price_max:
            if c > self.k:
                result.append(c - self.k)
            else:
                result.append(0)
        payoff = pd.Series(result)
        d = np.exp(self.r * self.t)
        price = np.mean(payoff) * d
        return price

    def delta(self, w, e):
        c1 = self.simulation(w, e)
        c2 = self.simulation(w, -e)
        price1 = self.price(c1)
        price2 = self.price(c2)
        delta = (price1 - price2) / (2 * e)
        return delta


t = 3
w = random_num(1000, 0, 1 / 252, 3)
put = Put(t, 100, 10, 0, 100)
call = Call(t, 100, 10, 0, 100)

# simulation and results
p = put.simulation(w, 0)
p.plot(legend=0)
plt.title('Simulation price')
plt.xlabel('t')
plt.ylabel('Price')

# ending values of the underlying asset
ending_value = p.iloc[-1, :]
ending_value.reset_index(drop=True)

# histogram and normality test
put.histogram(ending_value, 'Ending values', 'values', 'Frequency')
sm.qqplot(ending_value, fit=True, line='s')
plt.show()

print('simulation price of put is:', put.price(p))

# change of delta
dict = {}
for c in np.arange(0.1, 6, 0.5):
    a = call.delta(w, c)
    dict[c] = [a]
df = pd.DataFrame(dict)
df = df.T
df.plot(legend=0)
plt.xlabel('e')
plt.ylabel('delta')
plt.show()

dict = {}
for c in np.arange(0.1, 6, 0.5):
    b = put.delta(w, c)
    dict[c] = [b]
df = pd.DataFrame(dict)
df = df.T
df.plot(legend=0)
plt.xlabel('e')
plt.ylabel('delta')
plt.show()
