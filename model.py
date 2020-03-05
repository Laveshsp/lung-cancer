# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 02:48:17 2020

@author: Lavesh
"""

import pandas as pd
import numpy as np
import pickle
df = pd.read_csv('data/ThoracicSurgery.csv')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split, GridSearchCV
X = df.drop(['Death_1yr', 'MI_6mo', 'Asthma'], axis=1)
#print(X)
# Attributes of Significance from Hypothesis Testing
#X2 = df[['Performance', 'Dyspnoea', 'Cough', 'Tumor_Size', 'Diabetes_Mellitus']]
y = df['Death_1yr']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=1111, stratify=y)
#print(X_train) 
#print(y_train)   
clf = LogisticRegression(class_weight=None, random_state=1111)
    
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#print("hello")

pickle.dump(clf, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
#pred=model.predict([np.array([3.0,2.88,2.16,1.0,0.0,0.0,0,1.0,1.0,4.0,0.0,0.0,1.0,60.0])])
#print(pred)
#if(pred[0]==0):
 #   print("Hello")
