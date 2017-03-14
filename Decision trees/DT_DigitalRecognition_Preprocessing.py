#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 13:23:15 2017

@author: fabbas1
"""

import pandas as pd

df = pd.read_csv("/Users/fabbas1/Google Drive/study/Phd/Machine Learning/assignment/ITCS6156_SLProject/DigitRecognition/optdigits_raining.csv", 
                 index_col=0)

print("Number of (Rows, Columns) : ", df.shape)

X =  df.iloc[:, :63]

print(df[[0]].mean())

for i in range(0,(df.shape[1])):
    mean = df[[i]].mean()[0]
    std = df[[i]].std()[0]
    print( i , "," , mean , "," , std )
    

i = df.shape[1]-1
series = df[[i]].stack().value_counts()

print(series)
    

    
    