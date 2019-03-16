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




#Declaring variables
#!Need to have sample descriptions - deal in tkkinter
user_1 = [[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1]]
user_2 = [[0,0,1,0,0,1,0,1,0,0,1,0,1,0,0,0,0,1,1]]
user_3 = [[1,1,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1]]

global df
df = pd.read_csv('bank_full_w_dummy_vars.csv')
global dummy_values
dummy_values = False

#preprocesing data <-> Dummy Variables Already exist in dataset!!!!
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


#displays tkinter box
def display_gui():
	window = Tk() 
	window.title("Welcome to FirstWords")
	window.geometry('350x200')
	 
	selected = IntVar() 
	rad1 = Radiobutton(window,text='First', value=1, variable=selected) 
	rad2 = Radiobutton(window,text='Second', value=2, variable=selected) 
	rad3 = Radiobutton(window,text='Third', value=3, variable=selected)
	def clicked():
 		print(selected.get())
	 
	 
	btn = Button(window, text="Click Me", command=clicked) 
	rad1.grid(column=0, row=0) 
	rad2.grid(column=1, row=0) 
	rad3.grid(column=2, row=0) 
	btn.grid(column=3, row=0)
	 
	window.mainloop()
	return selected.get()

def display_gui_2(report):
	window = Tk()
	window.title("Welcome to FirstWorld")
	window.geometry('350x200')
	def clicked():
	 	messagebox.showinfo('Hi!', report)
	btn = Button(window,text='Click here', command=clicked)
	btn.grid(column=0,row=0)
	window.mainloop()





#main code!!!

#preprocess data
preprocess_data()
#get which user we want to test
user_name = display_gui()#this is an integer

#declaring feeatures I am going to look at
X = df.ix[:,(18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36)].values
#declaring features I am trying to predict
y = df.ix[:,17].values

#fiting data
LogReg = LogisticRegression()
LogReg.fit(X,y)

#doing test on accuracy
test_pred = LogReg.predict(X)
report = classification_report(y,test_pred)
#predicting which user will do this
if user_name == 1:
	user_pred = LogReg.predict(user_1)
if user_name == 2:
	user_pred = LogReg.predict(user_2)
if user_name == 3:
	user_pred = LogReg.predict(user_3)


display_gui_2(report)









