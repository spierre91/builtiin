#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 18:02:29 2022

@author: sadrachpierre
"""

from memory_profiler import profile
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from catboost import CatBoostClassifier

from timeit import default_timer as timer

import cProfile, pstats, io
from pstats import SortKey


'''
df = pd.read_csv("creditcard.csv")

print(df.head())

print("Number of rows: ", len(df))

print("Number of columns: ", len(df.columns))

df = df.sample(10000, random_state=42)

df.to_csv("creditcard_subsample10000.csv", index=False)
'''

@profile
def read_data(filename):
    df = pd.read_csv(filename)
    return df

@profile
def data_prep(dataframe, columns):
    df_select = dataframe[columns]
    return df_select

@profile
def feature_engineering(dataframe, inputs, output):
    X = dataframe[inputs]
    y = dataframe[output]
    return X, y
@profile
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size = 0.33)
    return X_train, X_test, y_train, y_test

@profile
def model_training(X_train, y_train, model_type):
    if model_type == 'Logistic Regression':
        model = LogisticRegression()
        model.fit(X_train, y_train)
    elif model_type == 'CatBoost':
        model = CatBoostClassifier()
        model.fit(X_train, y_train)        
    return model

@profile
def predict(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred
@profile
def evaluate(y_pred, y_test):
    precision = average_precision_score(y_test, y_pred)
    print("Precision: ", precision)

def main():
    runtime_metrics = dict()
    #read in data
    start = timer()
    data = read_data('creditcard.csv')
    end = timer()
    read_time = end - start
    runtime_metrics['read_time'] = read_time
    
    #slect relevant columns
    start = timer()
    columns = ['V1', 'V2', 'V3', 'Amount', 'Class']
    df_select = data_prep(data, columns)
    end = timer()
    select_time = end - start
    runtime_metrics['select_time'] = select_time
    
    
    #define input and output
    start = timer()
    inputs = ['V1', 'V2', 'V3']
    output = 'Class'
    X, y = feature_engineering(df_select, inputs, output)
    end = timer()
    data_prep_time = end - start
    runtime_metrics['data_prep_time'] = data_prep_time
    
    
    #split data for training and testing
    start = timer()
    X_train, X_test, y_train, y_test = split_data(X, y)
    end = timer()
    split_time = end - start
    runtime_metrics['split_time'] = split_time
    
    
    #fit model
    start = timer()
    model_type = 'CatBoost'
    model = model_training(X_train, y_train, model_type)
    end = timer()
    fit_time = end - start
    runtime_metrics['fit_time'] = fit_time
    
    #make predictions
    start = timer()
    y_pred = predict(model, X_test)
    end = timer()
    pred_time = end - start
    runtime_metrics['pred_time'] = pred_time
    
    #evaluate model predictions
    start = timer()
    evaluate(y_pred, y_test)
    end = timer()
    pred_time = end - start
    runtime_metrics['pred_time'] = pred_time
    
    print(runtime_metrics)
    
    
if __name__ == "__main__":
    main()

        
