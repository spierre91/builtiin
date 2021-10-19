import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt
sns.set()


df_original = pd.read_csv("HousingData.csv")

df = df_original.copy()

print(df.head())

df.info()
plt.boxplot(df['RM'])
plt.title("Box Plot of Number of Rooms")
plt.xlabel('RM')
plt.ylabel('Number of Rooms')
plt.show()

print("Before removing missing values:", len(df))
df['AGE'].fillna(df['AGE'].mean(),inplace=True)
print("After removing missing values:", len(df))
df['CRIM'].fillna(df['CRIM'].mean(),inplace=True)
df['ZN'].fillna(df['ZN'].mean(),inplace=True)
df['INDUS'].fillna(df['INDUS'].mean(),inplace=True)
df['CHAS'].fillna(df['CHAS'].mean(),inplace=True)
df['LSTAT'].fillna(df['LSTAT'].mean(),inplace=True)

df.info()

df_bad = df_original 
print("Mean age: ", df_bad['AGE'].mean())
df_bad['AGE'] = [str(x) if x > df_bad['AGE'].mean() else x for x in list(df_bad['AGE']) ]

df_bad['AGE'] = pd.to_numeric(df_bad['AGE'], errors = 'coerce')
print(df_bad['AGE'].mean())



from scipy import stats
import numpy as np 
print("Length before removing RM outlier:",  len(df_bad))
df_bad['RM_zscore'] = np.abs(stats.zscore(df['RM']))
df_clean1 = df_bad[df_bad['RM_zscore']< 3]
print("Length after removing RM outlier:",  len(df_clean1))


def remove_outliers(column_name, df_in):
    print(f"Length before removing {column_name} outlier:",  len(df_in))
    df_in[f'{column_name}_zscore'] = np.abs(stats.zscore(df_in[f'{column_name}']))
    df_clean = df_in[df_in[f'{column_name}_zscore']< 3]
    print(f"Length after removing {column_name} outlier:",  len(df_clean))
    return df_clean

df1 = remove_outliers('DIS', df_bad)

