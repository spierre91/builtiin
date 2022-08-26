age1 = 35
name1 =  "Fred Philips"
income1= 55250.15
senior_citizen1 = False


age2 = 42
name2 =  "Josh Rogers"
income2=65240.25
senior_citizen2 = False


age3 = 28
name3 =  "Bill Hanson"
income3=79250.65
senior_citizen3 = False


#age4 = ""
#name4 =  100
#income4 = 45250.65
#senior_citizen4 = True

#age4 = None
#name4 =  None
#income4 = 45250.65
#senior_citizen4 = True


import numpy as np 

age4 = np.nan
name4 =  np.nan
income4 = 45250.65
senior_citizen4 = np.nan

avg_age = (age1 + age2 + age3 + age4)/4

print(avg_age)


ages = []
names = []
incomes = []
senior_citizen = []

ages.append(age1)
ages.append(age2)
ages.append(age3)
ages.append(age4)
print("List of ages: ", ages)

names.append(name1)
names.append(name2)
names.append(name3)
names.append(name4)

print("List of names: ", names)


incomes.append(income1)
incomes.append(income2)
incomes.append(income3)
incomes.append(income4)


print("List of incomes: ", incomes)


senior_citizen.append(senior_citizen1)
senior_citizen.append(senior_citizen2)
senior_citizen.append(senior_citizen3)
senior_citizen.append(senior_citizen4)


print("List of senior citizen status: ", senior_citizen)

demo_dict = {}

demo_dict['age'] = ages
demo_dict['name'] = names
demo_dict['income'] = incomes
demo_dict['senior_citizen'] = senior_citizen

print("Demographics Dictionary")
print(demo_dict)


import pandas as pd 

demo_df = pd.DataFrame()

demo_df['age'] = ages
demo_df['name'] = names
demo_df['income'] = incomes
demo_df['senior_citizen'] = senior_citizen

print("Demographics Dataframe")
print(demo_df)




def income_after_tax(income, after_tax = np.nan):
    if income is float:
        after_tax = income - 0.22*income
    return after_tax



after_tax1 = income_after_tax(income1)
print("Before: ", income1)

print("After: ", after_tax1)







after_tax_invalid1 = income_after_tax('')
after_tax_invalid2 = income_after_tax(None)
after_tax_invalid3 = income_after_tax("income")
after_tax_invalid4 = income_after_tax(True)
after_tax_invalid5 = income_after_tax({})

print("after_tax_invalid1: ", after_tax_invalid1)
print("after_tax_invalid2: ", after_tax_invalid2)
print("after_tax_invalid3: ", after_tax_invalid3)
print("after_tax_invalid4: ", after_tax_invalid4)
print("after_tax_invalid5: ", after_tax_invalid5)


def get_after_tax_list(input_list, out_list = []):
    if type(input_list) is list:
        out_list = [x - 0.22*x for x in input_list]
    print("After Tax Incomes: ", out_list)
    return out_list


    
out_list1 = get_after_tax_list(incomes)
out_list2 = get_after_tax_list(5)


def get_income_truth_values(input_dict, output_dict={'avg_income': np.nan}):
    if type(input_dict) is dict and 'income' in input_dict:
        output_dict= {'avg_income': np.mean(input_dict['income'])}
    print(output_dict)
    return output_dict 

get_income_truth_values(demo_dict)      
get_income_truth_values(10000) 

demo_df['state'] = ['NY', 'MA', 'NY', 'CA']
demo_df['age'].fillna(demo_df['age'].mean(), inplace=True)
demo_df['income'].fillna(demo_df['income'].mean(), inplace=True)

def income_age_groupby(input_df, output_df = pd.DataFrame({'state': [np.nan], 'age': [np.nan], 'income':[np.nan]})):
    if type(input_df) is type(pd.DataFrame()) and  set(['age', 'income', 'state']).issubset(input_df.columns):
        output_df = input_df.groupby(['state'])['age', 'income'].mean().reset_index()
    print(output_df)
    return output_df
    
income_age_groupby(demo_df)
income_age_groupby([1,2,3])
