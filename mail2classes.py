import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.grid_search import GridSearchCV
import imports
from imports import descriptors, model


#######################DATASET##################################################

df = pd.read_csv('./americasOrCalendar.csv', na_values=['?'],header=0)
df["features"] = df["X-From"].map(str)
X=df['features']
y=df['class']


########################DESCRIPTORS############################################

X=descriptors.tf_idf(X)

#######################MODEL##################################################

grid = model.decision_tree(X,y)

#######################RESULTS##################################################

print("The best model : ")
print("- best score  : {}\n".format(grid.best_score_))
print("- best params : {}\n".format(grid.best_params_))
print("- best estimator : {}\n".format(grid.best_estimator_))
