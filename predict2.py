# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 15:25:57 2019
@author: resistance
"""

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_csv('trainJobTitleSplit.csv')

y = df[['Income']]

x = df[['Age', 'Gender', 'Year of Record', 'Country', 'Size of City', 'Profession_Word1', 'Profession_Word2', 'Profession_Word3', 'Profession_Word4', 'Profession_Word5', 'University Degree', 'Wears Glasses', 'Hair Color', 'Height' ]]
cat_columns = ["Gender", "Country", 'Profession_Word1', 'Profession_Word2', 'Profession_Word3', 'Profession_Word4', 'Profession_Word5', "University Degree", "Wears Glasses", "Hair Color"]
x = pd.get_dummies(df, prefix_sep="__", columns=cat_columns)

# save x to CSV and then use existing code online
# x.to_csv(r'encodedJobTitleSplit.csv')
x.dropna()
y.dropna()


# run
x.fillna(x.mean())

x.drop(x.tail(2).index,inplace=True)
y.drop(y.tail(2).index,inplace=True)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)



regressor = LinearRegression()  
regressor.fit(X_train, y_train)

coeff_df = pd.DataFrame([[regressor.coef_]], x.columns, columns=['Coefficient'])  
# print(coeff_df)

y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': [y_test], 'Predicted': [y_pred]})

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



# run on prediction data

df = pd.read_csv('submissionData.csv')

xp = df[['Age', 'Gender', 'Year of Record', 'Country', 'Size of City', 'Profession_Word1', 'Profession_Word2', 'Profession_Word3', 'Profession_Word4', 'Profession_Word5', 'University Degree', 'Wears Glasses', 'Height' ]]
cat_columns = ["Gender", "Country", 'Profession_Word1', 'Profession_Word2', 'Profession_Word3', 'Profession_Word4', 'Profession_Word5', "University Degree", "Wears Glasses"]
xp = pd.get_dummies(xp, prefix_sep="__", columns=cat_columns)

# x.to_csv(r'x_temp6.csv')

x = pd.read_csv('x_temp6.csv')
xp.fillna(xp.mean(), inplace=True)
y_pred = regressor.predict(xp)



dataset = pd.read_csv('submissionData.csv')
dataset.fillna("unknown", inplace=True)
cat_columns = ["Gender", "Country", 'Profession_Word1', 'Profession_Word2', 'Profession_Word3', 'Profession_Word4', 'Profession_Word5', "University Degree", "Wears Glasses"]
dataset = pd.get_dummies(dataset, prefix_sep="__", columns=cat_columns)


y_pred = regressor.predict(dataset)
print(y_pred)



