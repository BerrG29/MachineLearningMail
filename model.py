from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
########################MODEL###################################################
def naiveBayes():
    #Naive bayes Gaussian
    grid = GridSearchCV(GaussianNB(), cv=10, param_grid=param_grid)
    return grid

def decision_tree(X,y):
    param_grid = {
        'criterion': ['gini','entropy'], 
        'splitter': ['best','random']
    }
    dtc = DecisionTreeClassifier(random_state=0)
    grid = GridSearchCV(dtc, cv=10, param_grid=param_grid)
    return  grid.fit(X,y)

def neural_network():
    #parameters = {'hidden_layer_sizes ': [(100,),(100,100), (100,100,100)], 'alpha':[10.0 ** -np.arange(1, 7)], 'activation':('identity', 'logist$
    parameters = {'activation':('relu','logistic')}
    clf=MLPClassifier()
    clf = GridSearchCV(nn, parameters)
    grid = GridSearchCV(pipe, cv=3, n_jobs=1, param_grid=param_grid)
    return grid

