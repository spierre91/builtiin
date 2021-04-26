#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 11:58:01 2021

@author: sadrachpierre
"""

import pandas as pd 
import numpy as np
from collections import Counter


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


df_new = pd.read_csv("covid.csv")



df_new.dropna(inplace=True)


print(Counter(df_new['journal']).most_common(100))


df_plos = df_new[df_new['journal'] == 'PLoS One'].copy()
print(df_plos.head())
df_infect = df_new[df_new['journal'].str.contains('Infect Dis', regex=False)].copy()
df_microbial = df_new[df_new['journal'].str.contains('Microbial', regex=False)].copy()

df_abstract_microbiome = df_new[df_new['abstract'].str.contains('microbiome', regex=False)].copy()
print("Number of Microbiome Studies: ", len(df_abstract_microbiome))


print(df_abstract_microbiome.head())

df_abstract_microbiome['publish_time'] = pd.to_datetime(df_abstract_microbiome['publish_time'], format='%Y/%m/%d')
df_abstract_microbiome['year'] = df_abstract_microbiome['publish_time'].dt.year
print(df_abstract_microbiome.head())
print(set(df_abstract_microbiome['year']))
print(df_abstract_microbiome['abstract'].iloc[20])

from textblob import TextBlob
df_abstract_microbiome['abstract_sentiment'] = df_abstract_microbiome['abstract'].apply(lambda abstract: TextBlob(abstract).sentiment.polarity)
print(df_abstract_microbiome.head())


df_micro_group = df_abstract_microbiome.groupby(['year'])['abstract_sentiment'].mean()


import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
plt.xlabel('Year')
plt.ylabel('Sentiment')
plt.title('Research Sentiment in Gut Microbiome Studies')
plt.plot(df_micro_group.index, df_micro_group.values)
#plt.show()



df_plos['publish_time'] = pd.to_datetime(df_plos['publish_time'], format='%Y/%m/%d')
df_plos['year'] = df_plos['publish_time'].dt.year
df_plos['abstract_sentiment'] = df_plos['abstract'].apply(lambda abstract: TextBlob(abstract).sentiment.polarity)



df_plos_group = df_plos.groupby(['year'])['abstract_sentiment'].mean()


import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
plt.xlabel('Year')
plt.ylabel('Sentiment')
#plt.title('Research Sentiment in PLoS One Publications')
plt.plot(df_plos_group.index, df_plos_group.values)



df_nature = df_new[df_new['journal'] == 'Nature'].copy()

df_nature['publish_time'] = pd.to_datetime(df_nature['publish_time'], format='%Y/%m/%d')
df_nature['year'] = df_nature['publish_time'].dt.year
df_nature['abstract_sentiment'] = df_nature['abstract'].apply(lambda abstract: TextBlob(abstract).sentiment.polarity)



df_nature = df_nature.groupby(['year'])['abstract_sentiment'].mean()


import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
plt.xlabel('Year')
plt.ylabel('Sentiment')
plt.title('Research Sentiment in Publications')
plt.plot(df_nature.index, df_nature.values)
