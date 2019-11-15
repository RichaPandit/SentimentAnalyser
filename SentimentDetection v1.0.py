import math
import random
from collections import defaultdict
from pprint import pprint

# Prevent future/deprecation warnings from showing in output
import warnings
warnings.filterwarnings(action='ignore')

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

mHeadline = str(sys.argv[1])
mHeadline = mHeadline.replace('"','')
mHeadline = [mHeadline]
mFilePath = str(sys.argv[2])


df = pd.read_csv('C:/Users/richa.pandit/Desktop/imdb_labelled.tsv', sep='\t')
df.head()


df = df[df.label != 0]
df.label.value_counts()

from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(max_features=1000, binary=True)
X = vect.fit_transform(df.headline)

X.toarray()


from sklearn.model_selection import train_test_split

X = df.headline
y = df.label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(max_features=1000, binary=True)

X_train_vect = vect.fit_transform(X_train)

counts = df.label.value_counts()

print("\nPredicting only -1 = {:.2f}% accuracy",format(counts[-1] / sum(counts) * 100))


from imblearn.over_sampling import SMOTE

sm = SMOTE()

X_train_res, y_train_res = sm.fit_sample(X_train_vect, y_train)


unique, counts = np.unique(y_train_res, return_counts=True)
print(list(zip(unique, counts)))

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

nb.fit(X_train_res, y_train_res)

nb.score(X_train_res, y_train_res)

X_test1= mHeadline
X_test_vect1 = vect.transform(X_test1) #Insert sentence here

X_test_vect = vect.transform(X_test)

y_pred = nb.predict(X_test_vect)

y_pred

y_pred1 = nb.predict(X_test_vect1)

y_pred1


from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\nF1 Score: {:.2f}".format(f1_score(y_test, y_pred) * 100))
print("\nCOnfusion Matrix:\n", confusion_matrix(y_test, y_pred))


Html_file= open(mFilePath,"w")
Html_file.write(str(y_pred1))
Html_file.write('\n')
Html_file.write(str(accuracy_score(y_test, y_pred) * 100))
Html_file.write('\n')
Html_file.write(str(f1_score(y_test, y_pred) * 100))
Html_file.close()
