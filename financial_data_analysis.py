#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 12:19:13 2021

@author: sadrachpierre
"""
import pandas_datareader.data as web
import datetime
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

sns.set()
start = datetime.datetime(2019,6,23)
end = datetime.datetime(2021,6,23)

# amzn = web.DataReader('AMZN','yahoo',start,end)
# amzn['Date'] = amzn.index
# print(amzn.head())
# amzn.to_csv(f"amzn_{start}_{end}.csv", index=False)


# googl = web.DataReader('GOOGL','yahoo',start,end)
# googl['Date'] = googl.index
# print(googl.head())
# googl.to_csv(f"googl_{start}_{end}.csv", index=False)

# aapl = web.DataReader('AAPL','yahoo',start,end)
# aapl['Date'] = aapl.index
# print(aapl.head())
# aapl.to_csv(f"aapl_{start}_{end}.csv", index=False)


amzn_df = pd.read_csv(f'amzn_{start}_{end}.csv')
googl_df = pd.read_csv(f'googl_{start}_{end}.csv')   
aapl_df = pd.read_csv(f'aapl_{start}_{end}.csv')   

# print(aapl_df.head())

amzn_df['Returns'] = (amzn_df['Open'] - amzn_df['Close'])/amzn_df['Open']
amzn_df['Returns'].hist()
import numpy as np
mean_amnz_returns = np.round(amzn_df['Returns'].mean(), 5)
std_amnz_returns = np.round(amzn_df['Returns'].std(), 2)
plt.title(f'AMZN Stock Price Returns Distribution; Mean {mean_amnz_returns}, STD: {std_amnz_returns}')
plt.show()


googl_df['Returns'] = (googl_df['Open'] - googl_df['Close'])/googl_df['Open']
googl_df['Returns'].hist()
mean_googl_returns = np.round(googl_df['Returns'].mean(), 5)
std_googl_returns = np.round(googl_df['Returns'].std(), 2)
plt.title(f'GOOGL Stock Price Returns Distribution; Mean {mean_googl_returns}, STD: {std_googl_returns}')
plt.show()

aapl_df['Returns'] = (aapl_df['Open'] - aapl_df['Close'])/aapl_df['Open']
aapl_df['Returns'].hist()
mean_aapl_returns = np.round(aapl_df['Returns'].mean(), 5)
std_aapl_returns = np.round(aapl_df['Returns'].std(), 2)
plt.title(f'AAPL Stock Price Returns Distribution; Mean {mean_aapl_returns}, STD: {std_aapl_returns}')
plt.show()


amzn_df['Ticker'] =  'AMZN'
googl_df['Ticker'] =  'GOOGL'
aapl_df['Ticker'] =  'AAPL'

df = pd.concat([amzn_df, googl_df, aapl_df])
df = df[['Ticker', 'Returns']]
print(df.head())

sns.boxplot(x= df['Ticker'], y = df['Returns'])
plt.title('Box Plot for AMZN, GOOGL and AAPL Returns')
plt.show()

df_corr = pd.DataFrame({'AMZN':amzn_df['Returns'], 'GOOGL':googl_df['Returns'], 'AAPL':aapl_df['Returns']})
print(df_corr.head())
corr = df_corr.corr()
sns.heatmap(corr, annot= True)
plt.show()


cutoff = datetime.datetime(2021,1,23)
amzn_df['Date'] = pd.to_datetime(amzn_df['Date'], format='%Y/%m/%d')
amzn_df = amzn_df[amzn_df['Date'] > cutoff]
amzn_df['SMA_10'] = amzn_df['Close'].rolling(window=10).mean()
print(amzn_df.head())
plt.plot(amzn_df['Date'], amzn_df['SMA_10'])
plt.plot(amzn_df['Date'], amzn_df['Adj Close'])
plt.title("Moving average and Adj Close price for AMZN")
plt.ylabel('Adj Close Price')
plt.xlabel('Date')
plt.show()


googl_df['Date'] = pd.to_datetime(googl_df['Date'], format='%Y/%m/%d')
googl_df = googl_df[googl_df['Date'] > cutoff]
googl_df['SMA_10'] = googl_df['Close'].rolling(window=10).mean()
print(googl_df.head())
plt.plot(googl_df['Date'], googl_df['SMA_10'])
plt.plot(googl_df['Date'], googl_df['Adj Close'])
plt.title("Moving average and Adj Close price for GOOGL")
plt.ylabel('Adj Close Price')
plt.xlabel('Date')
plt.show()


aapl_df['Date'] = pd.to_datetime(aapl_df['Date'], format='%Y/%m/%d')
aapl_df = aapl_df[aapl_df['Date'] > cutoff]
aapl_df['SMA_10'] = aapl_df['Close'].rolling(window=10).mean()
print(googl_df.head())
plt.plot(aapl_df['Date'], aapl_df['SMA_10'])
plt.plot(aapl_df['Date'], aapl_df['Adj Close'])
plt.title("Moving average and Adj Close price for AAPL")
plt.ylabel('Adj Close Price')
plt.xlabel('Date')
plt.show()

amzn_df['SMA_10_STD'] = amzn_df['Adj Close'].rolling(window=10).std() 
amzn_df['Upper Band'] = amzn_df['SMA_10'] + (amzn_df['SMA_10_STD'] * 2)
amzn_df['Lower Band'] = amzn_df['SMA_10'] - (amzn_df['SMA_10_STD'] * 2)
amzn_df.index = amzn_df['Date']
amzn_df[['Adj Close', 'SMA_10', 'Upper Band', 'Lower Band']].plot(figsize=(12,6))
plt.title('10 Day Bollinger Band for Amazon')
plt.ylabel('Adjusted Close Price')
plt.show()



googl_df['SMA_10_STD'] = googl_df['Adj Close'].rolling(window=10).std() 
googl_df['Upper Band'] = googl_df['SMA_10'] + (googl_df['SMA_10_STD'] * 2)
googl_df['Lower Band'] = googl_df['SMA_10'] - (googl_df['SMA_10_STD'] * 2)
googl_df.index = googl_df['Date']
googl_df[['Adj Close', 'SMA_10', 'Upper Band', 'Lower Band']].plot(figsize=(12,6))
plt.title('10 Day Bollinger Band for Google')
plt.ylabel('Adjusted Close Price')
plt.show()


aapl_df['SMA_10_STD'] = aapl_df['Adj Close'].rolling(window=10).std() 
aapl_df['Upper Band'] = aapl_df['SMA_10'] + (aapl_df['SMA_10_STD'] * 2)
aapl_df['Lower Band'] = aapl_df['SMA_10'] - (aapl_df['SMA_10_STD'] * 2)
aapl_df.index = aapl_df['Date']
aapl_df[['Adj Close', 'SMA_10', 'Upper Band', 'Lower Band']].plot(figsize=(12,6))
plt.title('10 Day Bollinger Band for Apple')
plt.ylabel('Adjusted Close Price')
plt.show()
