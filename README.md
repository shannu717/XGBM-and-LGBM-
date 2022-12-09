# XGBoost 
### XGBoost

!pip install xgboost 

# First XGBoost model for Pima Indians dataset
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pandas import read_csv


# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8] 
dataframe   

# split data into train and test sets
seed = 42
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed) 
 

# fit model no training data
model = XGBClassifier(max_depth =6, n_estimators=500, learning_rate=0.50,gamma=0.5, objective='binary:logistic')
model.fit(X_train, y_train)       


# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]  

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f" % (accuracy * 100.0))        

predictions 

***Light GBM***

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8] 
dataframe 

# Splitting the dataset into the Training set and Test set

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


!pip install lightgbm

import lightgbm as lgb
d_train = lgb.Dataset(x_train, label=y_train)

params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 50
params['max_depth'] = 10 

clf = lgb.train(params, d_train, 500) 

#Prediction
y_pred=clf.predict(x_test)

predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)


accuracy

predictions 
