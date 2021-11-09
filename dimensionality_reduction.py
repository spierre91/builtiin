#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:10:03 2021

@author: sadrachpierre
"""
import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


df = pd.read_csv("Loan_status_2007-2020Q3.gzip")



print("Number of Columns: ", len(list(df.columns)))
print("Number of rows: ", len(df))


print(df.head())


df = df[df['purpose'] == 'credit_card']


columns = ['loan_amnt', 'loan_status', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate','mths_since_recent_revol_delinq','home_ownership', 'verification_status',
 'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util', 'chargeoff_within_12_mths', 'delinq_amnt', 'last_fico_range_low', 'last_fico_range_high']


df = df[columns]
df.to_csv("credit_card_loan.csv", index=False)



df_credit = pd.read_csv("credit_card_loan.csv")

print("Number of Columns: ", len(list(df_credit.columns)))
print("Number of rows: ", len(df_credit))


def fill_na(numerical_column):
    df_credit[numerical_column].fillna(df_credit[numerical_column].mean(), inplace=True)
    
fill_na('mths_since_recent_revol_delinq')
fill_na('num_accts_ever_120_pd')
fill_na('num_actv_bc_tl')
fill_na('num_actv_rev_tl')
fill_na('avg_cur_bal')
fill_na('bc_open_to_buy')
fill_na('bc_util')




def convert_categories(categorical_columnn):
     df_credit[categorical_columnn] = df_credit[categorical_columnn].astype('category')
     df_credit[f'{categorical_columnn}_cat'] = df_credit[categorical_columnn].cat.codes
     
convert_categories('home_ownership')     
convert_categories('verification_status')   
convert_categories('term')  


print(set(df_credit['loan_status']))

df_credit = df_credit[df_credit['loan_status'].isin(['Fully Paid', 'Default', 'Charged Off'])]

print(df_credit.head())   



df_credit['loan_status_label'] = np.where(df_credit['loan_status'] == 'Fully Paid', 0, 1)
columns2 = ['loan_amnt', 'loan_status_label', 'funded_amnt', 'funded_amnt_inv', 'term_cat', 'int_rate','mths_since_recent_revol_delinq','home_ownership_cat', 'verification_status_cat',
 'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util', 'chargeoff_within_12_mths', 'delinq_amnt', 'last_fico_range_low', 'last_fico_range_high']
df_credit = df_credit[columns2]
print(df_credit.head()) 

df_credit['int_rate'] = df_credit['int_rate'].str.rstrip('%')
df_credit['int_rate'] = df_credit['int_rate'].astype(float)
df_credit.fillna(0, inplace=True)


X = df_credit[['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term_cat', 'int_rate','mths_since_recent_revol_delinq','home_ownership_cat', 'verification_status_cat',
 'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util', 'chargeoff_within_12_mths', 'delinq_amnt', 'last_fico_range_low', 'last_fico_range_high']]
y = df_credit['loan_status_label']

X_train, X_test, y_train, y_test = train_test_split(X, y , random_state=42, test_size = 0.33)


import seaborn as sns
import matplotlib.pyplot as plt
model = RandomForestClassifier()
model.fit(X_train, y_train)

features = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term_cat', 'int_rate','mths_since_recent_revol_delinq','home_ownership_cat', 'verification_status_cat',
 'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util', 'chargeoff_within_12_mths', 'delinq_amnt', 'last_fico_range_low', 'last_fico_range_high']
    
    
feature_df = pd.DataFrame({"Importance":model.feature_importances_, "Features": features })
sns.set()
plt.bar(feature_df["Features"], feature_df["Importance"])
plt.xticks(rotation=90)
plt.title("Random Forest Model Feature Importance")
plt.show()

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

features2 = ['loan_amnt', 'loan_status_label', 'funded_amnt', 'funded_amnt_inv', 'term_cat', 'int_rate','mths_since_recent_revol_delinq','home_ownership_cat', 'verification_status_cat',
 'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util', 'chargeoff_within_12_mths', 'delinq_amnt', 'last_fico_range_low', 'last_fico_range_high']
    



X = df_credit[features2]
scaler = StandardScaler()

scaler.fit(X)
X_scaled=scaler.transform(X)

pca=PCA(n_components=4) 
pca.fit(X_scaled)
X_components=pca.transform(X_scaled) 

components_df = pd.DataFrame({'component_one': list(X_components[:,0]), 'component_two': list(X_components[:,1]),
                              'component_three': list(X_components[:,2]), 'component_four': list(X_components[:,3])})

print(components_df.head())




labels=X.loan_status_label
color_dict={0:'Red',1:'Blue'}



fig,ax=plt.subplots(figsize=(7,5))

sns.set()
for i in np.unique(labels):   
 index=np.where(labels==i)
 ax.scatter(components_df['component_one'].loc[index],components_df['component_two'].loc[index],c=color_dict[i],s=10,
           label=i)
 
 
plt.xlabel("1st Component",fontsize=14)
plt.ylabel("2nd Component",fontsize=14)
plt.title('Scatter Plot of Principal Components')
plt.legend()
plt.show()



