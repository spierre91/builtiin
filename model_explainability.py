#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 12:31:27 2021

@author: sadrachpierre
"""
import pandas as pd 
import numpy as np 


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv("telco_churn.csv")

df['gender_cat'] = df['gender'].astype('category')
df['gender_cat'] = df['gender_cat'].cat.codes

df['PaperlessBilling_cat'] = df['PaperlessBilling'].astype('category')
df['PaperlessBilling_cat'] = df['PaperlessBilling_cat'].cat.codes



df['Contract_cat'] = df['Contract'].astype('category')
df['Contract_cat'] = df['Contract_cat'].cat.codes


df['PaymentMethod_cat'] = df['PaymentMethod'].astype('category')
df['PaymentMethod_cat'] = df['PaymentMethod_cat'].cat.codes


df['Partner_cat'] = df['Partner'].astype('category')
df['Partner_cat'] = df['Partner_cat'].cat.codes



df['Dependents_cat'] = df['Dependents'].astype('category')
df['Dependents_cat'] = df['Dependents_cat'].cat.codes


df['DeviceProtection_cat'] = df['DeviceProtection'].astype('category')
df['DeviceProtection_cat'] = df['DeviceProtection_cat'].cat.codes


print(df.head())

df['churn_score'] = np.where(df['Churn']=='Yes', 1, 0)

X = df[[ 'tenure', 'MonthlyCharges', 'gender_cat', 'PaperlessBilling_cat',
        'Contract_cat','PaymentMethod_cat', 'Partner_cat', 'Dependents_cat', 'DeviceProtection_cat' ]]
y = df['churn_score']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)


from sklearn.metrics import confusion_matrix
conmat = confusion_matrix(y_test, y_pred)

val = np.mat(conmat) 
classnames = list(set(y_train))

df_cm = pd.DataFrame(
        val, index=classnames, columns=classnames, 
    )
df_cm = df_cm.astype('float') / df_cm.sum(axis=1)[:, np.newaxis]  

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure()
heatmap = sns.heatmap(df_cm, annot=True, cmap="Blues")
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Churn Logistic Regression Model Results')
plt.show()

from sklearn.inspection import plot_partial_dependence
features = [0, 1, (1, 0)]
plot_partial_dependence(lr_model, X_train, features, target=1) 


from sklearn.ensemble import RandomForestClassifier

rf_model =  RandomForestClassifier(n_estimators=100, max_depth=10)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)




conmat = confusion_matrix(y_test, y_pred_rf)

val = np.mat(conmat) 
classnames = list(set(y_train))

df_cm_rf = pd.DataFrame(
        val, index=classnames, columns=classnames, 
    )
df_cm_rf = df_cm_rf.astype('float') / df_cm_rf.sum(axis=1)[:, np.newaxis]  



plt.figure()
heatmap = sns.heatmap(df_cm_rf, annot=True, cmap="Blues")
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Churn Random Forest Model Results')
plt.show()

features = ['tenure', 'MonthlyCharges', 'gender_cat', 'PaperlessBilling_cat',
        'Contract_cat','PaymentMethod_cat', 'Partner_cat', 'Dependents_cat', 'DeviceProtection_cat' ]

print(rf_model.feature_importances_)
feature_df = pd.DataFrame({'Importance':rf_model.feature_importances_, 'Features': features })

sns.set()
plt.bar(feature_df['Features'], feature_df['Importance'])
plt.xticks(rotation=90)
plt.title('Random Forest Model Feature Importance')
plt.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(8, input_shape = (len(features),)))
model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs = 1)

y_pred_nn = [round(float(x)) for x in model.predict(X_test)]

conmat = confusion_matrix(y_test, y_pred_nn)

val = np.mat(conmat) 
classnames = list(set(y_train))

df_cm_nn = pd.DataFrame(
        val, index=classnames, columns=classnames, 
    )
df_cm_nn = df_cm_nn.astype('float') / df_cm_nn.sum(axis=1)[:, np.newaxis]  


plt.figure()
heatmap = sns.heatmap(df_cm_nn, annot=True, cmap="Blues")
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Churn Neural Network Model Results')
plt.show()

import shap

f = lambda x: model.predict(x)
med = X_train.median().values.reshape((1,X_train.shape[1]))

explainer = shap.Explainer(f, med)
shap_values = explainer(X_test.iloc[0:1000,:])

shap.plots.beeswarm(shap_values)


import lime
from lime import lime_tabular

explainer = lime_tabular.LimeTabularExplainer(training_data=np.array(X_train),feature_names=X_train.columns,class_names=['Yes', 'No'],
    mode='classification')

exp = explainer.explain_instance(data_row=X_test.iloc[1], predict_fn=model.predict, labels=(0,))

exp.show_in_notebook(show_table=True)
