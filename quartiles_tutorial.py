import pandas as pd 

df = pd.read_csv('telco_churn.csv')

print(df.head())

print("First Quartile (Q1) for Tenure: ", df['tenure'].quantile(0.25))
print("Second Quartile (Q2) for Tenure: ", df['tenure'].quantile(0.50))
print("Third Quartile (Q3) for Tenure: ", df['tenure'].quantile(0.75))

print("Third Quartile (Q3) for Tenure: ", df['tenure'].quantile(0.75))



print("Ninth Decile for Tenure: ", df['tenure'].quantile(0.9))

df_dsl = df[df['InternetService'] == 'DSL']
df_fiberoptic = df[df['InternetService'] == 'Fiber optic']


print("Third Quartile (Q3) for Tenure - DSL: ", df_dsl['tenure'].quantile(0.75))
print("Third Quartile (Q3) for Tenure - Fiber Optic: ", df_fiberoptic['tenure'].quantile(0.75))

print("Ninth Decile for Tenure - DSL: ", df_dsl['tenure'].quantile(0.9))
print("Ninth Decile for Tenure - Fiber Optic: ", df_fiberoptic['tenure'].quantile(0.9))


df_churn_yes = df[df['Churn'] == 'Yes']
df_churn_no = df[df['Churn'] == 'No']


print("Third Quartile (Q3) for Tenure - Churn: ", df_churn_yes['tenure'].quantile(0.75))
print("Third Quartile (Q3) for Tenure - No Churn: ", df_churn_no['tenure'].quantile(0.75))


print("Third Quartile (Q3) for Tenure - Churn: ", df_churn_yes['MonthlyCharges'].quantile(0.75))
print("Third Quartile (Q3) for Tenure - No Churn: ", df_churn_no['MonthlyCharges'].quantile(0.75))


import numpy as np 


print("Numpy Third Quartile (Q3) for Tenure - Churn: ", np.percentile(df_churn_yes['MonthlyCharges'], 75))

import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

sns.boxplot(df['tenure'])
plt.show()


from collections import Counter 
def get_boxplot_of_categories(data_frame, categorical_column, numerical_column, limit):

    keys = []
    for i in dict(Counter(df[categorical_column].values).most_common(limit)):
        keys.append(i)
    
    df_new = df[df[categorical_column].isin(keys)]
    sns.boxplot(x = df_new[categorical_column], y = df_new[numerical_column])
    plt.show()
    
    
get_boxplot_of_categories(df, 'Churn', 'tenure', 5)
    
get_boxplot_of_categories(df, 'Churn', 'MonthlyCharges', 5)
    
