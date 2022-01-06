#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 17:40:26 2022

@author: sadrachpierre
"""
import pandas as pd 
import numpy as np 
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

df = pd.read_csv('telco_churn.csv')

print(df.head())



df['Churn'] = np.where(df['Churn'] == 'Yes', 1, 0)




def convert_categories(cat_list):
    for col in cat_list:
        df[col] = df[col].astype('category')
        df[f'{col}_cat'] = df[f'{col}'].cat.codes


category_list = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                  'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                  'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

convert_categories(category_list)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(0, inplace=True)

cols = ['gender_cat', 'Partner_cat', 'Dependents_cat', 'PhoneService_cat', 'MultipleLines_cat', 'InternetService_cat',
                  'OnlineSecurity_cat', 'OnlineBackup_cat', 'DeviceProtection_cat', 'TechSupport_cat', 'StreamingTV_cat',
                  'StreamingMovies_cat', 'Contract_cat', 'PaperlessBilling_cat', 'PaymentMethod_cat','MonthlyCharges',
                  'TotalCharges', 'SeniorCitizen']

X = df[cols]

y= df['Churn']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

model_bce = Sequential()
model_bce.add(Dense(len(cols),input_shape=(len(cols),), kernel_initializer='normal', activation='relu'))
model_bce.add(Dense(32, activation='relu'))
model_bce.add(Dense(32, activation='relu'))
model_bce.add(Dense(32, activation='relu'))
model_bce.add(Dense(1, activation='softmax'))
model_bce.compile(optimizer = 'adam',loss='binary_crossentropy', metrics =['accuracy'])
model_bce.fit(X_train, y_train,epochs =10)







(X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = mnist.load_data()



plt.imshow(X_train_mnist[0])
plt.show()


plt.imshow(X_train_mnist[1])
plt.show()

plt.imshow(X_train_mnist[4])
plt.show()

X_train_mnist = X_train_mnist.reshape((X_train_mnist.shape[0], 28, 28, 1))
X_test_mnist = X_test_mnist.reshape((X_test_mnist.shape[0], 28, 28, 1))

y_train_mnist = np.where(y_train_mnist == 9, 1, 0)
y_test_mnist = np.where(y_test_mnist == 9, 1, 0)


model_cce = Sequential()
model_cce.add(Conv2D(16, (3, 3), activation='relu', kernel_initializer='normal', input_shape=(28, 28, 1)))
model_cce.add(MaxPooling2D((2, 2)))
model_cce.add(Flatten())
model_cce.add(Dense(16, activation='relu', kernel_initializer='normal'))
model_cce.add(Dense(2, activation='softmax'))
model_cce.compile(optimizer = 'SGD',loss='sparse_categorical_crossentropy', metrics =['accuracy'])
model_cce.fit(X_train_mnist, y_train_mnist, epochs =5)
