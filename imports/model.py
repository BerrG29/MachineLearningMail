from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
#import keras
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation
#from keras.optimizers import SGD
from sklearn import svm


def naiveBayes(X,Y):
    models={'gaussian':GaussianNB(),'bernoulli': BernoulliNB(),'multinomial':MultinomialNB()}
    param_grid_bernoulli={'alpha': [0.0, 0.5, 1.0],
                'binarize':[0.2, 0.5, 0.7, 1.0],
                'fit_prior': [True, False]
    }
    param_grid_multinomial={'alpha': [0.0, 0.5, 1.0],
                'fit_prior': [True, False]
    }
    grid_resultat=[]
    score_precedent=0
    for modele in models:
        if modele=='gaussian':
            grid = GridSearchCV(models[modele], param_grid={}, cv=5)
            resu=grid.fit(X.toarray(),Y)
        elif modele=='bernoulli':
            grid = GridSearchCV(models[modele], param_grid=param_grid_bernoulli, cv=5)
            resu=grid.fit(X.toarray(),Y)
        else:
            grid = GridSearchCV(models[modele], param_grid=param_grid_multinomial, cv=5)
            resu=grid.fit(X.toarray(),Y)
            
        if grid.best_score_ > score_precedent:
           score_precedent=grid.best_score_
           grid_resultat= resu
    return grid_resultat

def decision_tree(X,y):
    param_grid = {
        'criterion': ['gini','entropy'], 
        'splitter': ['best','random'],
        'max_depth':[10,20,30,40,50]
    }
    dtc = DecisionTreeClassifier(random_state=0)
    grid = GridSearchCV(dtc, cv=5, param_grid=param_grid)
    return  grid.fit(X,y)

def SVM():
    #param_grid= {'kernel':['rbf','linear','sigmoid','poly'],'degree':[1,2,3,4],'C':[1,10,100,1000],'gamma': [0.1,0.01,1,0.001]}
    #best params directly
    param_grid= {'kernel':['sigmoid'],'degree':[1],'C':[10],'gamma': [1]}
    clf=svm.SVC(class_weight='balanced')
    return GridSearchCV(clf,cv=5,param_grid=param_grid)

def neural_network():
     param_grid= {'hidden_layer_sizes':[(7, 7), (60,), (60, 7)],
                                        'activation':['identity', 'logistic', 'tanh', 'relu'],
                                        'solver':['lbfgs', 'sgd', 'adam'],
                                        'learning_rate' : ['constant', 'invscaling', 'adaptive']}
     clf = MLPClassifier()
     grid=GridSearchCV(clf,cv=5, param_grid=param_grid)
     return grid

# def keras(dim):
#     model = Sequential()
#     # Dense(64) is a fully-connected layer with 64 hidden units.
#     # in the first layer, you must specify the expected input data shape:
#     # here, 20-dimensional vectors.
#     model.add(Dense(64, activation='relu', input_dim=dim))
#     model.add(Dropout(0.5))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(10, activation='softmax'))

#     sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#     model.compile(loss='categorical_crossentropy',
#                   optimizer=sgd,
#                   metrics=['accuracy'])
#     return model
