import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
#import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.grid_search import GridSearchCV
# nltk.download('punkt')
# nltk.download('wordnet')
#stemmer = PorterStemmer()

#def stem_tokens(tokens, stemmer):
 #   stemmed = []
  #  for item in tokens:
   #     stemmed.append(stemmer.stem(item))
    #return stemmed

#def tokenize(text):
 #   tokens = nltk.word_tokenize(text)
  #  stems = stem_tokens(tokens, stemmer)
   # return stems

#######################DATASET##################################################

df = pd.read_csv('../shorted_email_1800_500.csv', na_values=['?'],header=0)
df["features"] = df["X-From"].map(str)
X=df['features']
y=df['class']

########################DESCRIPTORS##############################################
#vectorizer = TfidfVectorizer(tokenizer=tokenize,sublinear_tf=True,stop_words='english')
vectorizer = TfidfVectorizer(sublinear_tf=True,stop_words='english')
X = vectorizer.fit_transform(X)

########################MODEL###################################################
##Naive bayes Gaussian
#clf = GaussianNB()
##Decision tree
parameters = {
                'criterion': ['gini','entropy'], 
                'splitter': ['best','random']
         }
dtc = DecisionTreeClassifier(random_state=0)
grid = GridSearchCV(dtc, parameters)
#
##Neural network
parameters = {'hidden_layer_sizes ': [(100,),(100,100), (100,100,100)], 'alpha':[10.0 ** -np.arange(1, 7)], 'activation':('identity', 'logistic', 'tanh', 'relu'), 'solver' :('lbfgs', 'sgd', 'adam'), 'learning_rate' : ('constant', 'invscaling', 'adaptive'), 'learning_rate_init':[0.05, 0.01, 0.005, 0.001], 'power_t':[0.1, 0.5, 1, 2]}
nn=neural_network.MLPClassifier()
grid = GridSearchCV(nn, parameters)

#######################RESULTS##################################################
print(np.mean(cross_val_score(clf,X, y, cv=10)))

print("The best model : ")
print("- best score  : {}\n".format(grid.best_score_))
print("- best params : {}\n".format(grid.best_params_))
print("- best estimator : {}\n".format(grid.best_estimator_))