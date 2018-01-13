import pandas as pd
import numpy as np
#import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
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

df = pd.read_csv('../shorted_email_1800_500.csv', na_values=['?'],header=0)

df["all"] = df["X-From"].map(str)

X=df['all']
# print(X)
# X=df[['content', 'Subject']]
y=df['class']


#vectorizer = TfidfVectorizer(tokenizer=tokenize,sublinear_tf=True,stop_words='english')
vectorizer = TfidfVectorizer(sublinear_tf=True,stop_words='english')
X = vectorizer.fit_transform(X)
#clf = GaussianNB()
# # print (X)
clf = DecisionTreeClassifier(random_state=0)
print(np.mean(cross_val_score(clf,X, y, cv=10)))
