import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import GridSearchCV
from sklearn.grid_search import GridSearchCV
import imports
from imports import descriptors, model

#dataSets=['../kaminski_unbalanced_600_50.csv','../all_balanced_100_predictable.csv','../kaminski_balanced_50.csv', '../all_multiplePerson.csv','../all_unbalanced_2600_1000.csv']
dataSets=['../kaminski_unbalanced_600_50.csv']
models_list=['neural_network']

for dataSet in dataSets:
    #######################DATASET##################################################
    df = pd.read_csv(dataSet, na_values=['?'],header=0)
    df["features"] = df[field].map(str)
    X=df['features']
    y=df['class']


    ########################DESCRIPTORS############################################
    X=descriptors.tf_idf(X)
    best_grid = model.decision_tree(X,y)

    #######################MODELS##################################################
    for m in models_list:
        if(m == "neural_network"):
            grid = model.neural_network(X,y)

    #######################RESULTS##################################################
        print("The best model for: {}\n".format(dataSet))
        print("- best score  : {}\n".format(best_grid.best_score_))
        print("- best params : {}\n".format(best_grid.best_params_))
        print("- best estimator : {}\n".format(best_grid.best_estimator_))
