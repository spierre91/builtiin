#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 00:34:19 2022

@author: sadrachpierre
"""
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np 
from scipy.optimize import differential_evolution


df = pd.read_csv("Concrete_Data_Yeh.csv")

print(df.head())


X = df[['cement', 'slag', 'flyash', 'water', 'superplasticizer',
       'coarseaggregate', 'fineaggregate', 'age']]

y = df['csMPa']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 42)

model = RandomForestRegressor(n_estimators=100, max_depth=100, random_state =42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("RMSE: ", rmse)


import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.title("Actual vs. Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")


model_full= RandomForestRegressor(n_estimators=100, max_depth=100, random_state =42)
model_full.fit(X, y)


def obj_fun(X):
    X = [X]
    results = model_full.predict(X)
    obj_fun.counter += 1
    print(obj_fun.counter)
    return -results



boundaries = [(df['cement'].min(), df['cement'].max()), (df['slag'].min(), df['slag'].max()), (df['flyash'].min(), df['flyash'].max()), 
                (df['water'].min(), df['water'].max()), (df['superplasticizer'].min(), df['superplasticizer'].max()),
       (df['coarseaggregate'].min(), df['coarseaggregate'].max()), (df['fineaggregate'].min(), df['fineaggregate'].max()), (df['age'].min(), df['age'].max())]


obj_fun.counter = 0

if __name__ == '__main__':
    
    
    opt_results = differential_evolution(obj_fun, boundaries)
    
        
    print('cement:', opt_results.x[0])
    print('slag:', opt_results.x[1])
    print('flyash:', opt_results.x[2])
    print('water:', opt_results.x[3])
    print('superplasticizer:', opt_results.x[4])
    print('coarseaggregate:', opt_results.x[5])
    print('fineaggregate:', opt_results.x[6])
    print('age:', opt_results.x[7])
    
    
    print("Max Strength: ", -opt_results.fun)


import dlib 

lbounds = [df['cement'].min(), df['slag'].min(), df['flyash'].min(), df['water'].min(), df['superplasticizer'].min(), df['coarseaggregate'].min(), 
           df['fineaggregate'].min(), df['age'].min()]
ubounds = [df['cement'].max(), df['slag'].max(), df['flyash'].max(), df['water'].max(), df['superplasticizer'].max(), df['coarseaggregate'].max(), 
           df['fineaggregate'].max(), df['age'].max()]
max_fun_calls = 1000

def maxlip_obj_fun(X1, X2, X3, X4, X5, X6, X7, X8):
    X = [[X1, X2, X3, X4, X5, X6, X7, X8]]
    results = model_full.predict(X)
    return results


sol, obj_val = dlib.find_max_global(maxlip_obj_fun, lbounds, ubounds, max_fun_calls)

print("MAXLIPO Results: ")
print('cement:', sol[0])
print('slag:', sol[1])
print('flyash:', sol[2])
print('water:', sol[3])
print('superplasticizer:', sol[4])
print('coarseaggregate:', sol[5])
print('fineaggregate:', sol[6])
print('age:', sol[7])


print("Max Strength: ", obj_val)


