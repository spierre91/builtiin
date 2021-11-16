#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 14:55:53 2021

@author: sadrachpierre
"""
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("telco_churn.csv")
df.fillna(0,inplace=True)
print(df.head())

import numpy as np 
df['Churn'] = np.where(df['Churn'] == 'Yes', 1, 0)

def convert_categories(cat_list):
    for col in cat_list:
        df[col] = df[col].astype('category')
        df[f'{col}_cat'] = df[col].cat.codes
        df[f'{col}_cat'] = df[f'{col}_cat'].astype(float)
    
category_list = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                  'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                  'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
convert_categories(category_list)

print(df.head())
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(0, inplace=True)
cols = ['gender_cat', 'Partner_cat', 'Dependents_cat', 'PhoneService_cat', 'MultipleLines_cat', 'InternetService_cat',
                  'OnlineSecurity_cat', 'OnlineBackup_cat', 'DeviceProtection_cat', 'TechSupport_cat', 'StreamingTV_cat',
                  'StreamingMovies_cat', 'Contract_cat', 'PaperlessBilling_cat', 'PaymentMethod_cat','MonthlyCharges',
                  'TotalCharges', 'SeniorCitizen', 'tenure']

X = df[cols]
print(X.head())
df['Churn'] = df['Churn'].astype(int)
y = df['Churn']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.metrics import accuracy_score


model = Sequential()
model.add(Dense(len(cols),input_shape=(len(cols),), kernel_initializer='normal', activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer = 'adam',loss='binary_crossentropy', metrics =['accuracy'])
model.fit(X_train, y_train,epochs =20)

y_pred = model.predict(X_test)
y_pred = np.where(y_pred > 0.5, 1, 0)
print("Accuracy: ", accuracy_score(y_pred, y_test))


from tensorflow.keras import regularizers

model_lasso = Sequential()
model_lasso.add(Dense(len(cols),input_shape=(len(cols),), kernel_initializer='normal', activation='relu', kernel_regularizer = regularizers.l1(1e-6)))
model_lasso.add(Dense(32, activation='relu'))
model_lasso.add(Dense(32, activation='relu'))
model_lasso.add(Dense(1, activation='sigmoid'))
model_lasso.compile(optimizer = 'adam',loss='binary_crossentropy', metrics =['accuracy'])
model_lasso.fit(X_train, y_train,epochs =20)

y_pred = model_lasso.predict(X_test)
y_pred = np.where(y_pred > 0.5, 1, 0)
print("Accuracy With Lasso: ", accuracy_score(y_pred, y_test))


model_ridge = Sequential()
model_ridge.add(Dense(len(cols),input_shape=(len(cols),), kernel_initializer='normal', activation='relu', kernel_regularizer = regularizers.l2(1e-6)))
model_ridge.add(Dense(32, activation='relu'))
model_ridge.add(Dense(32, activation='relu'))
model_ridge.add(Dense(1, activation='sigmoid'))
model_ridge.compile(optimizer = 'adam',loss='binary_crossentropy', metrics =['accuracy'])
model_ridge.fit(X_train, y_train,epochs =20)

y_pred = model_ridge.predict(X_test)
y_pred = np.where(y_pred > 0.5, 1, 0)
print("Accuracy With Ridge: ", accuracy_score(y_pred, y_test))
