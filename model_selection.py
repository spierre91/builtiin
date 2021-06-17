#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 13:11:36 2021

@author: sadrachpierre
"""

import pandas as pd 
import numpy  as np
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv("telco_churn.csv")

print(df.head())


df['Churn_binary'] = np.where(df['Churn'] == 'Yes', 1, 0)


X = df[['tenure', 'MonthlyCharges']]

y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= 42)




# folds = KFold(n_splits=5)
# folds.get_n_splits(X)



# from sklearn.metrics import accuracy_score
# fold = 0
# for train_index, test_index in folds.split(X):
#     X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
#     model = RandomForestClassifier()
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     fold+=1
#     print(f"Accuracy in fold {fold}:", accuracy_score(y_pred, y_test))
# loo = LeaveOneOut()
# for train_index, test_index in loo.split(X):
#     X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    


import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, chi2
import matplotlib.pyplot as plt


df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df['TotalCharges'].fillna(0, inplace = True)
df['TotalCharges']  = df['TotalCharges'].astype(float)


X = df[['tenure', 'MonthlyCharges', 'TotalCharges']]

y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= 42)


numerical_predictors = ["MonthlyCharges", "TotalCharges", "tenure" ]
numerical_selector = SelectKBest(f_classif, k=3)
numerical_selector.fit(X_train[numerical_predictors], y_train)



num_scores = -np.log10(numerical_selector.pvalues_)


plt.bar(range(len(numerical_predictors)), num_scores)
plt.xticks(range(len(numerical_predictors)), numerical_predictors, rotation='vertical')
plt.xlabel("Feature")
plt.ylabel("Score")
plt.show()

from sklearn.model_selection import RandomizedSearchCV


n_estimators = [50, 100, 200]
max_features = ['auto', 'sqrt', 'log2']
max_depth = [int(x) for x in np.linspace(10, 30, num = 5)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]



random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

model = RandomForestClassifier()

rf_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid,
                                n_iter = 3, cv =3, verbose=2, random_state=42)
rf_random.fit(X_train, y_train)

parameters = rf_random.best_params_
print(parameters)

