#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 13:21:23 2017

@author: fabbas1
"""
from sklearn.cross_validation import cross_val_predict
from sklearn import metrics
from time import time
from sklearn.model_selection import KFold, cross_val_score
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
import pydotplus





# read training set
df = pd.read_csv("/Users/fabbas1/Google Drive/study/Phd/Machine Learning/assignment/ITCS6156_SLProject/DigitRecognition/optdigits_raining.csv", 
                 index_col=0)

# features
X =  df.iloc[:, :63]


# drop column 38. all zeros
X.drop(X.columns[38])

# target variable
y = df.iloc[:, 63:]

# reading test set
test_df = pd.read_csv("/Users/fabbas1/Google Drive/study/Phd/Machine Learning/assignment/ITCS6156_SLProject/DigitRecognition/optdigits_test.csv", 
                 index_col=0)
# split as before
X_Test =  test_df.iloc[:, :63]
X_Test.drop(X_Test.columns[38])
Y_Label = test_df.iloc[:,63:]

splitting_criterion = ["entropy","gini"]

for criteria in splitting_criterion:
    #build & fit Decision Tree
    dt = DecisionTreeClassifier(criterion=criteria,min_samples_split=3, 
                            random_state=99,splitter="best",
                            max_features=None,
                            max_depth=None)
    print(criteria)    
    start = time()
    dt.fit(X, y)
    print(time() - start)
    
    print(dt.score(X,y))
    print(dt.score(X_Test,Y_Label))

#reshape for cross validation
labels = y.values
c, r = labels.shape
labels = labels.reshape(c,)

max_depth= range(1,20)
for depth in max_depth:
    #build & fit Decision Tree
    dt = DecisionTreeClassifier(criterion="entropy",min_samples_split=3, 
                            random_state=99,splitter="best",
                            max_features=None,
                            max_depth=depth)
   
    start = time()
    dt.fit(X, y)
    print(depth,",",time()-start, "," , dt.score(X,y),",",dt.score(X_Test,Y_Label), end=",") 
    scores = cross_val_score(dt, X, labels, cv=10)
    print(scores.mean(), "," , scores.std())

# pruning by specifiying maximum leaf nodes
leaf_nodes = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160]
for leaf_nodes in leaf_nodes:
    #build & fit Decision Tree
    dt = DecisionTreeClassifier(criterion="entropy",min_samples_split=3, 
                            random_state=99,splitter="best",
                            max_features=None,
                            max_depth=11,
                            max_leaf_nodes=leaf_nodes)
   
    start = time()
    dt.fit(X, y)
    print(leaf_nodes,",",time()-start, "," , dt.score(X,y),",",dt.score(X_Test,Y_Label), end=",") 
    scores = cross_val_score(dt, X, labels, cv=10)
    print(scores.mean(), "," , scores.std())

dt = DecisionTreeClassifier(criterion="entropy",min_samples_split=3, 
                            random_state=99,splitter="best",
                            max_features=None,
                            max_depth=11,
                            max_leaf_nodes=10)
dt.fit(X, y)

    
dot_data = tree.export_graphviz(dt,out_file='/Users/fabbas1/Desktop/tree',
                     filled=True, rounded=True)




