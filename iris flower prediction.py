# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 00:15:46 2022

@author: user
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
data=pd.read_csv(r"C:\Users\user\Downloads\IRIS.csv")
data
data.columns
sns.pairplot(data, hue="species")
df=data.values
x=df[:,0:4]
y=df[:,4]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(y_test)
#support vector machine algorithm
from sklearn.svm import SVC
model_svc=SVC()
model_svc.fit(x_train,y_train)
SVC()
prediction1=model_svc.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,prediction1))
#logistic regression
from sklearn.linear_model import LogisticRegression
model_LR=LogisticRegression()
model_LR.fit(x_train,y_train)
prediction2=model_LR.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,prediction2)*100)
#decision tree classifier
from sklearn.tree import DecisionTreeClassifier
model_DT=DecisionTreeClassifier()
model_DT.fit(x_train,y_train)
DecisionTreeClassifier()
prediction3=model_svc.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,prediction3)*100)
from sklearn.metrics import classification_report
print(classification_report(y_test,prediction2))

x_new=np.array([[3,2,1,0.2],[4.9,2.2,3.8,1.1],[5.3,2.5,4.6,1.9]])
prediction=model_svc.predict(x_new)
print("prediction of species:{}".format(prediction))













