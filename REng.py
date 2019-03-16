#!Recommendation Engine
'''
Using Logistic regression to predict the value of 
categorical variables using predictor variables

A linear model does not output probabilities, but it treats the classes as numbers (0 and 1) and fits the best hyperplane (for a single feature, it is a line) that minimizes the distances between the points and the hyperplane. So it simply interpolates between the points, and you cannot interpret it as probabilities.

'''
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


from pandas import Series, DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
dummy_values = False



#print("hi")
df = pd.read_csv('bank_full_w_dummy_vars.csv')


#preprocessing data - clearing missing values as we have almost 50,000 rows
df.isnull().sum()
print(df.shape)
df.dropna(inplace=True)
print('hi')
#changing categorical data to numerical data if needed
if dummy_values == True:
	X = df.iloc[ : , :-1].values
	X[ : ,0] = label_encoder.fit_transform(X[ :,0])
	onehotencoder = OneHotEncoder(categorical_features=[0])
	X = onehotencoder.fit_transform(X)
	label_encoder = LabelEncoder()

#!Dummy Variables Already exist in dataset!!!!



X = df.ix[:,(18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36)].values
y = df.ix[:,17].values


LogReg = LogisticRegression()
LogReg.fit(X,y)

#!Have function that lets you choose users
user_1 = [[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1]]

y_pred = LogReg.predict(user_1)
print("hi")
print(y_pred)

'''

#!Need to have prediction function