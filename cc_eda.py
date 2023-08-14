import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_models import LinearRegression


df = pd.read_csv("synthetic_transaction_data_Dining.csv")


df['log_transaction_amnt'] = np.log(df['transaction_amount'])

print(df.head())


df['transaction_date'] = pd.to_datetime(df['transaction_date'])
df= df[df['merchant_name'] == "Cheesecake Factory"]

z_scores = np.abs((df['transaction_amount'] - df['transaction_amount'].mean()) / df['transaction_amount'].std())
# Define a threshold (e.g., Z-score > 3) to identify outliers
threshold = 3
# Remove outliers from the DataFrame
df = df[z_scores < threshold]


df['year'] = df['transaction_date'].dt.year
df['year'] = df['year'].astype(str)
df['month'] = df['transaction_date'].dt.month
df = df[df['month'] <= 12]
df = df[df['month'] >= 1]
df['month'] = df['month'].astype(str)
df['month_year'] =  df['year'] + "-"+  df['month']
df['month_year'] = pd.to_datetime(df['month_year'])


df_grouped = df.groupby('month_year')['transaction_amount'].sum().reset_index()
df_grouped = df_grouped.set_index('month_year').sort_index()
df_grouped.index = pd.to_datetime(df_grouped.index)
plt.plot(df_grouped.index, df_grouped['transaction_amount'])


data = df[['transaction_amount', 'log_transaction_amnt']]
sns.pairplot(data)
plt.show()

state_abbreviations = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}


df['merchant_state_abbr'] = df['merchant_state'].map(state_abbreviations)
state_counts = df.groupby('merchant_state_abbr')['cardholder_name'].nunique().reset_index()
state_counts.columns = ['State', 'Customer_Count']

fig = px.choropleth(state_counts, locations='State', locationmode='USA-states',
                    color='Customer_Count', scope='usa',
                    color_continuous_scale='Blues',
                    title='Number of Customers by State')

fig.show()


model = LinearRegression()
