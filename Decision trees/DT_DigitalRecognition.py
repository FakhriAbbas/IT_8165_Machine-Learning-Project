# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from sklearn.cross_validation import cross_val_score
import pydotplus 
from sklearn.grid_search import GridSearchCV
from operator import itemgetter
import numpy as np


def run_gridsearch(X, y, clf, param_grid, x_test,y_test, cv=5):
    """Run a grid search for best Decision Tree parameters.

    Args
    ----
    X -- features
    y -- targets (classes)
    cf -- scikit-learn Decision Tree
    param_grid -- [dict] parameter settings to test
    cv -- fold of cross-validation, default 5

    Returns
    -------
    top_params -- [dict] from report()
    """
    grid_search = GridSearchCV(clf,
                               param_grid=param_grid,
                               cv=cv)
    
    start = time()
    grid_search.fit(X, y)

    print("xxxxxxxxxxxx")
    print(grid_search.score(X,y))
    print(grid_search.score(x_test,y_test))
    print("xxxxxxxxxxxx")
    
    print(("\nGridSearchCV took {:.2f} "
           "seconds for {:d} candidate "
           "parameter settings.").format(time() - start,
                len(grid_search.grid_scores_)))

    top_params = report(grid_search.grid_scores_, 3)
    return  top_params

def report(grid_scores, n_top=3):
    """Report top n_top parameters settings, default n_top=3.

    Args
    ----
    grid_scores -- output from grid or random search
    n_top -- how many to report, of top models

    Returns
    -------
    top_params -- [dict] top parameter settings found in
                  search
    """
    top_scores = sorted(grid_scores,
                        key=itemgetter(1),
                        reverse=True)[:n_top]
    parameters = [x[0] for x in grid_scores]
    scores = [x[1] for x in grid_scores]
    
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print(("Mean validation score: "
               "{0:.3f} (std: {1:.3f})").format(
               score.mean_validation_score,
               np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

    return top_scores[0].parameters



# read training set
df = pd.read_csv("/Users/fabbas1/Google Drive/study/Phd/Machine Learning/assignment/ITCS6156_SLProject/DigitRecognition/optdigits_raining.csv", 
                 index_col=0)

# features
X =  df.iloc[:, :63]

# drop column 38. all zeros
X.drop(X.columns[38])

# target variable
y = df.iloc[:, 63:]

#build & fit Decision Tree
dt = DecisionTreeClassifier(criterion='entropy',min_samples_split=3, random_state=99,splitter="best",max_features=None,presort=False,max_depth=10)
dt.fit(X, y)

# reading test set
test_df = pd.read_csv("/Users/fabbas1/Google Drive/study/Phd/Machine Learning/assignment/ITCS6156_SLProject/DigitRecognition/optdigits_test.csv", 
                 index_col=0)

# split as before
X_Test =  test_df.iloc[:, :63]
X_Test.drop(X_Test.columns[38])
Y_Label = test_df.iloc[:,63:]


# print results
print(dt.score(X,y))
print(dt.score(X_Test,Y_Label))

#reshape for cross validation
labels = y.values
c, r = labels.shape
labels = labels.reshape(c,)

#cross validation
#scores = cross_val_score(dt, X, labels, cv=10)
#print("mean: {:.3f} (std: {:.3f})".format(scores.mean(),scores.std()),end="\n\n" )
#print(scores)

#list of parameters to check the best combination for DT
param_grid = {"criterion": ["gini", "entropy"],
              "min_samples_split": [1,2,3,4,5],
              "max_depth": [ 1,2,3,4,5],
              "min_samples_leaf": [1,2,3],
              "max_leaf_nodes": [2,3]
              }
              
              
#simulation
ts_gs = run_gridsearch(X, labels, dt, param_grid, X_Test,Y_Label, cv=10)
print("\n-- Best Parameters:")
for k, v in ts_gs.items():
    print("parameter: {:<20s} setting: {}".format(k, v))

print("\n\n-- Testing best parameters [Grid]...")
dt_ts_gs = DecisionTreeClassifier(**ts_gs)
scores = cross_val_score(dt_ts_gs, X, labels, cv=10)
print("mean: {:.3f} (std: {:.3f})".format(scores.mean(),
                                          scores.std()),
                                          end="\n\n" )
dt_ts_gs.fit(X,y)    
print("Best estimate : ")
print(dt_ts_gs.score(X,y)) 
print(dt_ts_gs.score(X_Test,Y_Label))  


 
#generate dot file
#dot_data = tree.export_graphviz(dt, out_file="/Users/fabbas1/Desktop/iris.dot") 
#graph = pydotplus.graph_from_dot_data(dot_data) 
#graph.write_pdf("/Users/fabbas1/Desktop/iris.pdf") 

     
