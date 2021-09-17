#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 15:45:53 2021

@author: sadrachpierre
"""


import pandas_datareader.data as web
import datetime
import pandas as pd
from functools import reduce


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# start = datetime.datetime(2019,9,15)
# end = datetime.datetime(2021,9,15)

# def get_stock(ticker):
#     data = web.DataReader(f"{ticker}","yahoo",start,end)
#     data[f'{ticker}'] = data["Close"]#(data["Close"] - data["Open"])/data["Open"]
#     data = data[[f'{ticker}']] 
#     print(data.head())
#     return data 

# pfizer = get_stock("PFE")
# jnj = get_stock("JNJ")




# def combine_stocks(tickers):
#     data_frames = []
#     for i in tickers:
#         data_frames.append(get_stock(i))
        
#     df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['Date'],
#                                             how='outer'), data_frames)
#     print(df_merged.head())
#     return df_merged
          

# stocks = ["MRNA", "PFE", "JNJ", "GOOGL", 
#           "FB", "AAPL", "COST", "WMT", "KR", "JPM", 
#           "BAC", "HSBC"]




# portfolio = combine_stocks(stocks)



# portfolio.to_csv("portfolio.csv", index=False)


portfolio = pd.read_csv("portfolio.csv")
print(portfolio.head())

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage


mu = mean_historical_return(portfolio)
S = CovarianceShrinkage(portfolio).ledoit_wolf()


ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()

cleaned_weights = ef.clean_weights()
print(dict(cleaned_weights))

ef.portfolio_performance(verbose=True)



from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

latest_prices = get_latest_prices(portfolio)

da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=100000)

allocation, leftover = da.greedy_portfolio()
print("Discrete allocation:", allocation)
print("Funds remaining: ${:.2f}".format(leftover))




from pypfopt import HRPOpt
returns = portfolio.pct_change().dropna()
hrp = HRPOpt(returns)
hrp_weights = hrp.optimize()
hrp.portfolio_performance(verbose=True)
print(dict(hrp_weights))

da_hrp = DiscreteAllocation(hrp_weights, latest_prices, total_portfolio_value=100000)

allocation, leftover = da_hrp.greedy_portfolio()
print("Discrete allocation (HRP):", allocation)
print("Funds remaining (HRP): ${:.2f}".format(leftover))




from pypfopt.efficient_frontier import EfficientCVaR
S = portfolio.cov()
ef_cvar = EfficientCVaR(mu, S)
cvar_weights = ef_cvar.min_cvar()

cleaned_weights = ef_cvar.clean_weights()
print(dict(cleaned_weights))

ef_cvar.portfolio_performance(verbose=True)

da_cvar = DiscreteAllocation(cvar_weights, latest_prices, total_portfolio_value=100000)

allocation, leftover = da_cvar.greedy_portfolio()
print("Discrete allocation (CVAR):", allocation)
print("Funds remaining (CVAR): ${:.2f}".format(leftover))
