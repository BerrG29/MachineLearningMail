import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import GridSearchCV
from sklearn.grid_search import GridSearchCV
import imports
from imports import descriptors, model

#dataSets=['../kaminski_unbalanced_600_50.csv','../all_balanced_100_predictable.csv','../kaminski_balanced_50.csv', '../all_multiplePerson.csv','../all_unbalanced_2600_1000.csv']
dataSets=['../kaminski_unbalanced_600_50.csv']
fieldToTest=['Content','X-From']
descriptors_list=['tf_idf','frequence']
models_list=['neural_network']

for dataSet in dataSets:
    #######################DATASET##################################################
    df = pd.read_csv(dataSet, na_values=['?'],header=0)
    for field in fieldToTest:
        df["features"] = df[field].map(str)
        X=df['features']
        y=df['class']
        
        best_descriptor = ""
        current_descriptor = ""
        
        best_model = ""
        current_model = ""
        
        best_grid = {}
        current_grid = {}

        ########################INITIALISATION############################################
        best_descriptor = "tf_idf"
        best_model = "decision_tree"
        X=descriptors.tf_idf(X)
        best_grid = model.decision_tree(X,y)

        ########################DESCRIPTORS############################################
        for descriptor in descriptors_list:
            if(descriptor == "frequence"):
                X=descriptors.frequence(X)
            if(descriptor == "tf_idf"):
                X=descriptors.tf_idf(X)

            #######################MODELS##################################################
            for m in models_list:
                if(m == "neural_network"):
                    current_model = m
                    current_grid = model.neural_network(X,y)
                    
                    if(current_grid.best_score_ > best_grid.best_score_ ):
                        best_grid.best_score_ = current_grid.best_score_
                        best_model = m
                        best_descriptor = descriptor
                        
                        #######################RESULTS##################################################
                        print("The best model for: {}\n".format(dataSet))
                        print("- best score  : {}\n".format(best_grid.best_score_))
                        print("- best params : {}\n".format(best_grid.best_params_))
                        print("- best estimator : {}\n".format(best_grid.best_estimator_))
