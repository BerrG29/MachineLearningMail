from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#import nltk
########################DESCRIPTORS##############################################

#nltk.download('punkt')
#nltk.download('wordnet')
#stemmer = PorterStemmer()

"""
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems
"""

def tf_idf(X):
    #vectorizer = TfidfVectorizer(tokenizer=tokenize,sublinear_tf=True,stop_words='english')
    vectorizer = TfidfVectorizer(sublinear_tf=True,stop_words='english')
    return vectorizer.fit_transform(X)

def frequence(X):
    vectorizer=CountVectorizer(stop_words='english')
    X= vectorizer.fit_transform(X)
