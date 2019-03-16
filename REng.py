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
from tkinter import * 
from tkinter.ttk import *

dummy_values = False

#Declaring variables
#!Need to have sample descriptions - deal in tkkinter
user_1 = [[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1]]
user_2 = [[0,0,1,0,0,1,0,1,0,0,1,0,1,0,0,0,0,1,1]]
user_3 = [[1,1,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1]]

global df
df = pd.read_csv('bank_full_w_dummy_vars.csv')

def preprocess_data():
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

def display_gui():
	window = Tk() 
	window.title("Welcome to LikeGeeks app")
	window.geometry('350x200')
	 
	selected = IntVar() 
	rad1 = Radiobutton(window,text='First', value=1, variable=selected) 
	rad2 = Radiobutton(window,text='Second', value=2, variable=selected) 
	rad3 = Radiobutton(window,text='Third', value=3, variable=selected)
	 
	 
	btn = Button(window, text="Click Me", command=clicked) 
	rad1.grid(column=0, row=0) 
	rad2.grid(column=1, row=0) 
	rad3.grid(column=2, row=0) 
	btn.grid(column=3, row=0)
	 
	window.mainloop()
	return selected.get()




#!Need to have prediction function



#main program
preprocess_data()

X = df.ix[:,(18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36)].values
y = df.ix[:,17].values
LogReg = LogisticRegression()
LogReg.fit(X,y)
test_pred = LogReg.predict(X)
user_pred = LogReg.predict(user_1)

report = classification_report(y,test_pred)
print(report)
print(user_pred)

