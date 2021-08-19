#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 17:32:26 2021

@author: sadrachpierre
"""
import pandas as pd 

df = pd.read_csv("banknotes.csv")


print(df.head())

from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

def boxplot(column):
    sns.boxplot(data=df,x=df[f"{column}"])
    plt.title(f"Boxplot of Swiss Banknote {column}")
    plt.show()

df_outlier1 = df[df['Length']> 216].copy()
print(Counter(df_outlier1['conterfeit']))




df_outlier2 = df[df['Length']> 215.5].copy()
print(Counter(df_outlier2['conterfeit']))


boxplot('Length')
boxplot('Right')
boxplot('Left')
boxplot('Bottom')
boxplot('Top')
boxplot('Diagonal')


df_outlier3 = df[(df['Length']> 215)&(df['Right']> 130)&(df['Left']> 130)&(df['Bottom']> 10)].copy()
print(Counter(df_outlier3['conterfeit']))
print(Counter(df['conterfeit']))

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.svm import OneClassSVM

X = df[['Length', 'Left', 'Right', 'Bottom', 'Top', 'Diagonal']]
y = df['conterfeit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = IsolationForest(random_state=0)
clf.fit(X_train)
y_pred = clf.predict(X_test)

import numpy as np
pred = pd.DataFrame({'pred': y_pred})
pred['y_pred'] = np.where(pred['pred'] == -1, 1, 0)

y_pred = pred['y_pred'] 
print("Precision:", precision_score(y_test, y_pred))



clf_svm = OneClassSVM(gamma='auto')
clf_svm.fit(X_train)
y_pred_svm = clf_svm.predict(X_test)

pred['svm'] = y_pred_svm
pred['svm_pred'] = np.where(pred['svm'] == -1, 1, 0)

y_pred_svm = pred['svm_pred']
print("SVM Precision:", precision_score(y_test, y_pred_svm))
