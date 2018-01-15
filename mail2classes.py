import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.grid_search import GridSearchCV
import model.py
import desciptors.py

#######################DATASET##################################################

df = pd.read_csv('../shorted_email_1800_500.csv', na_values=['?'],header=0)
df["features"] = df["X-From"].map(str)
X=df['features']
y=df['class']


########################DESCRIPTORS############################################

X=tf_idf(X)

#######################MODEL##################################################

grid = decision_tree(X,y)

#######################RESULTS##################################################

print("The best model : ")
print("- best score  : {}\n".format(grid.best_score_))
print("- best params : {}\n".format(grid.best_params_))
print("- best estimator : {}\n".format(grid.best_estimator_))
