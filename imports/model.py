from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD


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

def SVM():
     #param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5,],
     #         'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    param_grid= {'kernel':['rbf','linear','sigmoid','poly'],'degree':[1,2,3,4],'C':[1,10,100,1000],'gamma': [0.1,0.01,1,0.001]}
    #param_grid = {'C': [1e2],
              #'gamma': [0.01], }
    return GridSearchCV(SVC(class_weight='balanced'), param_grid)

def keras(dim):
    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    model.add(Dense(64, activation='relu', input_dim=dim))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model
