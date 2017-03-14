#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 21:21:32 2017

@author: fabbas1
"""

import pandas as pd
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from time import time

end = time()
data = pd.read_csv("/Users/fabbas1/Google Drive/study/Phd/Machine Learning/assignment/ITCS6156_SLProject/AmazonReviews/amazon_baby_train_clean.csv")
data_test = pd.read_csv("/Users/fabbas1/Google Drive/study/Phd/Machine Learning/assignment/ITCS6156_SLProject/AmazonReviews/amazon_baby_test_clean.csv")


data['merged'] = data['name_processed'] + " " +  data['review_processed']
data_test['merged'] = data_test['name_processed'] + " " +  data_test['review_processed']


count_vect = CountVectorizer()
print("Building CountVectorizer ...")
X = count_vect.fit_transform(data.merged.values.astype('U'))
x_test = count_vect.transform(data_test["merged"])


# test different depth performance
#for i in range (1,20):
#    dt = DecisionTreeClassifier(criterion="gini",min_samples_split=3, 
#                                random_state=99,splitter="best",
#                                max_features=None,
#                                max_depth=i)
#    start = time()
#    dt.fit(X, data["rating"])
#    print(i,",",dt.score(X,data["rating"]),",",dt.score(x_test,data_test['rating']),
#                         ",",time()-start , end="\n" )
#    start = time()
#    scores = cross_val_score(dt, X,data["rating"] , cv=10)
#    print(scores.mean(),",",time()-start)
    
# test different depth performance
#for i in [10,20,30,40,50,60,70,80,90,100]:
#    count_vect = CountVectorizer(min_df=i)
#    X = count_vect.fit_transform(data.merged.values.astype('U'))
#    x_test = count_vect.transform(data_test["merged"])
#    print(X.shape)
#    dt = DecisionTreeClassifier(criterion="gini",min_samples_split=3, 
#                                random_state=99,splitter="best",
#                                max_features=None,
#                                max_depth=11)
#    start = time()
#    dt.fit(X, data["rating"])
#    print(i,",",dt.score(X,data["rating"]),",",dt.score(x_test,data_test['rating']),
#                         ",",time()-start , end="\n" )

dt = DecisionTreeClassifier(criterion="gini",min_samples_split=3, 
                                random_state=99,splitter="best",
                                max_features=None,
                                max_depth=11)
dt.fit(X, data["rating"])
dot_data = tree.export_graphviz(dt,out_file='/Users/fabbas1/Desktop/AmazonTree',
                     filled=True, rounded=True)





