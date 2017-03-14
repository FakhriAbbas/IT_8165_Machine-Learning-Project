#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 12:56:51 2017

@author: fabbas1
"""
import pandas as pd
import nltk
from nltk.stem.porter import *


STOPWORDS_LIST = nltk.corpus.stopwords.words('english')
PUNCTUATION_LIST = [',','.',"'",'``','(',')' , '-' , '+' , '//' , '*' , ':','%', "''"]

def remove_stopword(row):
    if row:
        stemmer = PorterStemmer()
        tokenized_text = nltk.word_tokenize(row)
        row_stem = [stemmer.stem(word) for word in tokenized_text 
                        if word not in STOPWORDS_LIST and 
                            word not in PUNCTUATION_LIST and
                            len(word) > 2 ]
        return " ".join(row_stem)
    else:
        return ""

df = pd.read_csv("/Users/fabbas1/Google Drive/study/Phd/Machine Learning/assignment/ITCS6156_SLProject/AmazonReviews/amazon_baby_train.csv")
df = df.dropna()
print(df.shape)

i = df.shape[1]-1
series = df[[i]].stack().value_counts()
mean = df[[i]].mean()[0]
std = df[[i]].std()[0]
print( i , "," , mean , "," , std )
print(series)


df["name_processed"] = df.apply(lambda row: remove_stopword(row["name"]),1)
df["review_processed"] = df.apply(lambda row: remove_stopword(row["review"]),1)

del df["name"]
del df["review"]

df['merged'] = df['name_processed'] + " " +  df['review_processed']

df.to_csv("/Users/fabbas1/Google Drive/study/Phd/Machine Learning/assignment/ITCS6156_SLProject/AmazonReviews/amazon_baby_train_clean.csv")