#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 12:14:38 2022

@author: sadrachpierre
"""
import pandas as pd

df = pd.read_csv("creditcard_downsampled5000.csv")


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# df = df.sample(30000, random_state=42)

# df.to_csv("creditcard_downsampled5000.csv", index=False)
print(df.head())


import seaborn as sns
import matplotlib.pyplot as plt 

sns.set()

sns.boxplot(y = df['V14'])
plt.show()



Q1=df['V13'].quantile(0.25)
print("Q1:", Q1)

Q3=df['V13'].quantile(0.75)
print("Q3:", Q3)

IQR=Q3-Q1
print("IQR: ", IQR)

lower_bound = Q1 - 1.5*IQR
print("Lower Bound:", lower_bound)

upper_bound = Q3 + 1.5*IQR
print("Upper Bound:", upper_bound)

df_clean = df[(df['V13']>lower_bound)&(df['V13']<upper_bound)]


sns.boxplot(y = df_clean['V13'])
plt.show()

sns.scatterplot(df['V13'], df['V14'])

from sklearn.cluster import DBSCAN

X_train = df[['V13', 'V14']]
model = DBSCAN()
model.fit(X_train)


cluster_labels = model.labels_
plt.scatter(df["V13"], df["V14"], c = cluster_labels)

plt.show()

df['labels'] = cluster_labels
df_cluster_clean = df[df['labels'] != -1]

plt.scatter(df_cluster_clean["V13"], df_cluster_clean["V14"], c = 'r')
plt.xlabel('V13')
plt.ylabel('V14')
plt.show()


