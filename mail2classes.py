import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.grid_search import GridSearchCV

dataSets=['../kaminiski_unbalanced_600_50.csv','../all_balanced.csv','../all_unbalanced_100.csv','../kaminski_unbalanced_50.csv', '../all_multiplePerson.csv','../all_unbalanced_2600_1000.csv']

for dataSet in dataSets:
    #######################DATASET##################################################
    df = pd.read_csv(dataSet, na_values=['?'],header=0)
    df["features"] = df["X-From"].map(str)
    X=df['features']
    y=df['class']


    ########################DESCRIPTORS############################################
    X=tf_idf(X)

    #######################MODEL##################################################

    grid = decision_tree(X,y)

    #######################RESULTS##################################################

    print("The best model for: "+dataSets)
    print("- best score  : {}\n".format(grid.best_score_))
    print("- best params : {}\n".format(grid.best_params_))
    print("- best estimator : {}\n".format(grid.best_estimator_))
