#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 18:59:33 2022

@author: sadrachpierre
"""

import numpy as np 


tech_company_names = ['Facebook', 'Apple', 'Amazon', 'Netflix', 'Google']

tech_company_employees = [58604, 147000, 950000, 11300, 135301]

tech_company_revenue = [117, 378, 470, 30, 257]


tech_company_employee_bool = [x > 60000 for x in tech_company_employees ]


sort_company = sorted(tech_company_names)
sort_employee = sorted(tech_company_employees)

print(sort_company)
print(sort_employee)

new_company_info = ['Microsoft', 163000, 877, True]

tech_company_names.append(new_company_info[0])
tech_company_employees.append(new_company_info[1])
tech_company_revenue.append(new_company_info[2])
tech_company_employee_bool.append(new_company_info[3])


print('Company: ', tech_company_names)
print('Employees: ', tech_company_employees)
print("Revenue: ", tech_company_revenue)
print("Employee_threshold: ", tech_company_employee_bool)


mu, sigma = 80, 40
n_values = len(tech_company_names)
np.random.seed(21)
net_income_normal = np.random.normal(mu, sigma, n_values)
print(net_income_normal)

np.random.seed(64)
net_income_fat_tail = np.random.gumbel(mu, sigma, n_values)
print(net_income_fat_tail)


company_data_dict = {'company_name': tech_company_names,
                     'number_of_employees': tech_company_employees,
                     'company_revenue': tech_company_revenue,
                     'employee_threshold': tech_company_employee_bool, 
                     'net_income_normal': list(net_income_normal), 
                     'net_income_fat_tail': list(net_income_fat_tail)}

print(company_data_dict)

import json 
with open('company_data.json', 'w') as fp:
    json.dump(company_data_dict, fp)

f = open('company_data.json')
company_json = json.loads(f.read()) 
print(company_json)

import pandas as pd 
company_df = pd.DataFrame(company_data_dict)
print(company_df)

company_df.to_csv("comapany_csv_file.csv", index=False)

read_company_df = pd.read_csv("comapany_csv_file.csv")
print(read_company_df)
