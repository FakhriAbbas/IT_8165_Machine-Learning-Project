#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 17:47:31 2017

@author: fabbas1
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
  

# reading training set
df_train = pd.read_csv("/Users/fabbas1/Google Drive/study/Phd/Machine Learning/assignment/ITCS6156_SLProject/DigitRecognition/optdigits_raining.csv", 
                 header=None)
# reading test set
df_test = pd.read_csv("/Users/fabbas1/Google Drive/study/Phd/Machine Learning/assignment/ITCS6156_SLProject/DigitRecognition/optdigits_test.csv", 
                 header=None)

X_train =  df_train.iloc[:, :64]
y_train = df_train.iloc[: , 64:]


X_test = df_test.iloc[:, :64]
y_test = df_test.iloc[: , 64:]

scaler = StandardScaler()  
scaler.fit(X_train)  

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(5, 8), random_state=1)


clf.fit(X_train, y_train)

print(clf.score(X_train,y_train))
print(clf.score(X_test,y_test))

print(clf.predict([[0,0,8,15,16,13,0,0,0,1,11,9,11,16,1,0,0,0,0,0,7,14,0,0,0,0,3,4,14,12,2,0,0,1,16,16,16,16,10,0,0,2,12,16,10,0,0,0,0,0,2,16,4,0,0,0,0,0,9,14,0,0,0,0
]]))

  

    
    