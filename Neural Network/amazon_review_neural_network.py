#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 15:07:45 2017

@author: fabbas1
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler


data = pd.read_csv("/Users/fabbas1/Google Drive/study/Phd/Machine Learning/assignment/ITCS6156_SLProject/AmazonReviews/amazon_baby_train_clean.csv")
data_test = pd.read_csv("/Users/fabbas1/Google Drive/study/Phd/Machine Learning/assignment/ITCS6156_SLProject/AmazonReviews/amazon_baby_test_clean.csv")

data['merged'] = data['name_processed'] + " " +  data['review_processed']
data_test['merged'] = data_test['name_processed'] + " " +  data_test['review_processed']

del data["Unnamed: 0"]
del data["name_processed"]
del data["review_processed"]
del data_test["Unnamed: 0"]
del data_test["name_processed"]
del data_test["review_processed"]

print("Build Counter ... ")
count_vect = CountVectorizer(min_df=50)
X_train = count_vect.fit_transform(data.merged.values.astype('U'))
y_train = data['rating']

X_test = count_vect.transform(data_test["merged"])
y_test = data_test["rating"]

scaler = StandardScaler(with_mean=False)  
scaler.fit(X_train)  

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)


print(X_train.shape)

print("Train NN ... ")
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(5, 20), random_state=1)

clf.fit(X_train, data['rating'])

print("Accuracy ... ")
print(clf.score(X_train,y_train))
print(clf.score(X_test,y_test))
