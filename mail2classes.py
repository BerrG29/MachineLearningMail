import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('americasOrCalendar.csv', na_values=['?'],header=0)

X=df['content']
y=df['class']

vectorizer = TfidfVectorizer(sublinear_tf=True,stop_words='english', max_df=0.6)
X = vectorizer.fit_transform(X)

clf = DecisionTreeClassifier(random_state=0)
print(np.mean(cross_val_score(clf, X, y, cv=10)))




