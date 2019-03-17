#!Recommendation Engine!
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
import threading
from tkinter import messagebox



#Declaring variables
#!Need to have sample descriptions - deal in tkkinter
user_1 = [[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1]]
user_2 = [[0,0,1,0,0,1,0,1,0,0,1,0,1,0,0,0,0,1,1]]
user_3 = [[1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]]

global df
df = pd.read_csv('DatasetType.csv')
global dummy_values
dummy_values = False

#preprocesing data <-> Dummy Variables Already exist in dataset!!!!
def preprocess_data(type):
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
def display_gui(type):
	window = Tk() 
	window.title("Choose one of 3 test Users")
	window.geometry('350x300')
	 
	selected = IntVar() 
	rad1 = Radiobutton(window,text='First', value=1, variable=selected) 
	rad2 = Radiobutton(window,text='Second', value=2, variable=selected) 
	rad3 = Radiobutton(window,text='Third', value=3, variable=selected)
	def clicked():
 		messagebox.showinfo('FYI JUDGE!', "Just so you :) know each user has different attributes") 

	 
	 
	btn = Button(window, text="Click Me", command=clicked) 
	rad1.grid(column=0, row=0) 
	rad2.grid(column=1, row=0) 
	rad3.grid(column=2, row=0) 
	btn.grid(column=3, row=0)
	 
	window.mainloop()
	global user_name
	user_name =  selected.get()

def display_gui_2(text_to_display):
	window = Tk()
	window.title("Welcome to FirstWorld")
	window.geometry('350x300')
	def clicked():
	 	messagebox.showinfo('Accuracy Report', report)
	btn = Button(window,text='Click here', command=clicked)
	btn.grid(column=0,row=0)
	window.mainloop()


def display_gui_3(text_to_display):
	window = Tk() 
	window.title("Click For Your result!") 
	window.geometry('350x400') 
	def clicked(): 
	    messagebox.showinfo('YourResult!', text_to_display) 
	btn = Button(window,text='Click here', command=clicked) 
	btn.grid(column=0,row=0) 
	window.mainloop()







#main code!!!
# Start new Threads to preprocess data and get inputs at the same time
#We are using multithreading cause its fast and cool!
thread1 = threading.Thread(target=display_gui, args=("",))
thread2 = threading.Thread(target=preprocess_data, args=("",))


# Start new Threads
thread1.start()
thread2.start()



# Wait for all threads to complete
thread1.join()
thread2.join()



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

print (user_name)
#predicting which user will do this

if user_name == 1:
	user_pred = LogReg.predict(user_1)
if user_name == 2:
	user_pred = LogReg.predict(user_2)
if user_name == 3:
	user_pred = LogReg.predict(user_3)



value_add = user_pred[0]

if value_add == 0:
	display_gui_3("This user will not be interested in this material")
if value_add == 1:
	display_gui_3("This user will not be interested in this material,but let's assume the user is intersted in business courses")


#Displays model acuracy report!!
display_gui_2(report)


'''
Now is the time to leverage open data
'''
def open_data():
	global df1
	df1 = pd.read_csv('OpenData.csv', encoding='latin-1')
	df1 = pd.DataFrame(df1, columns = ['Type of Contact', 'TypeofOrganization','Name of Organization','City'])
	df1 = df1[df1["TypeofOrganization"] == "Business and Economic Development"]
	df1 = df1[df1["City"] == "Vancouver"]
	df1 = pd.DataFrame(df1, columns = ['Name of Organization'])
	print(df1.head())
	dfToList = df1['Name of Organization'].tolist()
	dfList = list(df1['Name of Organization'])
	display_gui_5(dfList)


def display_gui_5(text_to_display):
	window = Tk()
	window.title("Here's a list of the organizations that you might be interested in!")
	window.geometry('350x300')
	def clicked():
	 	messagebox.showinfo('Hope you like it!', text_to_display)
	btn = Button(window,text='Click here', command=clicked)
	btn.grid(column=0,row=0)
	window.mainloop()


open_data()