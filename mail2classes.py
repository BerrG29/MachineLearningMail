import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer

# nltk.download('punkt')
# nltk.download('wordnet')
stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

df = pd.read_csv('americasOrCalendar.csv', na_values=['?'],header=0)

df["all"] = df["X-To"].map(str) + df["content"].map(str)

X=df['all']
# print(X)
# X=df[['content', 'Subject']]
y=df['class']

vectorizer = TfidfVectorizer(tokenizer=tokenize,sublinear_tf=True,stop_words='english')
# vectorizer = TfidfVectorizer(sublinear_tf=True,stop_words='english')
X = vectorizer.fit_transform(X)
# # print (X)
clf = DecisionTreeClassifier(random_state=0)
print(np.mean(cross_val_score(clf, X, y, cv=10)))