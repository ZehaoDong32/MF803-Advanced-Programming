#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assignment1 task2
Title:Historical Analysis of Sector ETFs
Author: Zehao Dong
Email: zehao@bu.edu
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

#Parent class for options
class option(object):
    def __init__(self, r, S, K, sigma, t):
        self.r = r
        self.S = S
        self.K = K
        self.sigma = sigma
        self.t = t
    
    def simulation(self, mean, variance):
        '''
        Simulation for only for path
        '''
        w = np.random.normal(mean, variance ** 0.5, size=self.t * 252)
        result = []
        S = self.S
        result.append(S)
        for c in w:
            S = self.r * S * (1 / 252) + self.sigma * S * c + S
            result.append(S)
        return pd.DataFrame(result)
    
    def simulation_paths(self, paths, mean, variance):
        '''
        Simulation for a number of paths, 
        the number determines by the variable 'paths'
        '''
        result = pd.DataFrame()
        while paths > 0:
            a = self.simulation(mean, variance)
            result = pd.concat([result, a], axis = 1)
            paths -= 1
        return result
    
    def mean(self, list):
        mean = np.mean(list)
        return mean

    def variance(self, list):
        var = np.var(list)
        return var

    def standard_deviation(self, list):
        std = self.variance(list) ** 0.5
        return std

    def discount(self, price):
        '''
        Dicount factor for continuos compounding
        '''
        D = np.exp(-self.r * self.t)
        price = price * D
        return price

    def BSformula(self):
        '''
        Price calculated by Black-Scholes formula
        '''
        d1 = (np.log(self.S / self.K) + (self.r + self.sigma ** 2 / 2) * self.t) / (self.sigma * self.t ** 0.5)
        d2 = (np.log(self.S / self.K) + (self.r - self.sigma ** 2 / 2) * self.t) / (self.sigma * self.t ** 0.5)
        Nd1 = stats.norm.cdf(-d1)
        Nd2 = stats.norm.cdf(-d2)
        price = Nd2 * self.K * np.exp(-self.r * self.t) - self.S * Nd1
        return price
   
    def payoff(self, price):
        result = []
        for c in price:
            if c < self.K:
                result.append(self.K - c)
            else:
                result.append(0)
        return pd.Series(result)

    def histogram(self, series, title, x_label, y_label):
        plt.figure(figsize=(8, 6),dpi=120)
        series.plot.hist(bins = 15, rwidth = 0.8, color = '#607c8e')
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        
    def plot(self, df, title, x_label, y_label):
        plt.figure(figsize=(8, 3), dpi=120)
        df.plot(legend=False)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
    
#Child class for European options
class Euro_option(option):
    def __init__(self,r, S, K, sigma, t):
        option.__init__(self, r, S, K, sigma, t)
   
    def simulation_price(self, df):
        '''
        Terminal value
        '''
        value = df.iloc[-1, :]
        return value.reset_index(drop = True)      

#Child class for Lookback options  
class Lookback_option(option):
    def __init__(self, r, S, K, sigma, t):
        option.__init__(self, r, S, K, sigma, t)
    
    def simulation_price(self, df):
        '''
        Lowest value among each path
        '''
        value = df.min(axis=0)
        return value.reset_index(drop = True)
    
  
      
if __name__ == '__main__':        
#test
    euro = Euro_option(0, 100, 100, 0.25, 1)
    lookback = Lookback_option(0, 100, 100, 0.25, 1)
    simulation_result = euro.simulation_paths(1000, 0, 1 / 252)
    #terminal value
    euro_price = euro.simulation_price(simulation_result)
    print('mean of terminal values:\n', euro.mean(euro_price))
    print('standard deviation of terminal values:\n', 
          euro.standard_deviation(euro_price))
    #lowest price among each path
    lookback_price = lookback.simulation_price(simulation_result)
    euro_payoff = euro.payoff(euro_price)
    print('Payoffs for European options\n',
          euro_payoff)
    euro.histogram(euro_payoff,
                   'Payoffs for European options',
                   'Payoff', 'Frequency')
    print('mean of payoffs:\n', euro.mean(euro_payoff))
    print('standard deviation of payoffs:\n', 
          euro.standard_deviation(euro_payoff))
   #option price equal to average discounted payoff
    euro_option_price = euro.discount(euro.mean(euro_payoff))
    print('simulation price for the European option:\n',
          euro_option_price)
    print('price for the European option calculated by BS formula:\n',
          euro.BSformula())
    lookback_payoff = lookback.payoff(lookback_price)
    print('Payoffs for lookback options\n',
          lookback_payoff)
    lookback.histogram(lookback_payoff,
                   'Payoffs for lookback options',
                   'Payoff', 'Frequency')
    #option price equal to average discounted payoff
    lookback_option_price = lookback.discount(lookback.mean(lookback_payoff))
    print('simulation price for the lookback option:\n',
          lookback_option_price)
    print('the premium on lookback option:\n',
          lookback_option_price - euro_option_price)
    
    #how prices of these two options and the premium change when sigma changes
    dict = {}
    for c in np.linspace(0, 0.5, 51):
        option1 = Euro_option(0, 100, 100, c, 1)
        option2 = Lookback_option(0, 100, 100, c, 1)
        simulation = option1.simulation_paths(100, 0, 1 / 252)
        payoff1 = option1.payoff(option1.simulation_price(simulation))
        payoff2 = option2.payoff(option2.simulation_price(simulation))
        price1 = option1.mean(option1.discount(payoff1))
        price2 = option2.mean(option2.discount(payoff2))
        premium = price2 - price1
        dict[c] = [price1, price2, premium]  
        
    df = pd.DataFrame.from_dict(dict, orient='index', columns=['European', 'Lookback', 'Premium'])
    df.plot() 
    plt.xlabel('Sigma')
    plt.ylabel('Price')