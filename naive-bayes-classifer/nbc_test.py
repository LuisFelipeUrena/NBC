import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

from nbc import NaiveBayesClassifier

# in this case we will use the Wine dataset provided by the sklearn library
wine = load_wine()

X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=128)

nb = NaiveBayesClassifier()

nb.fit(X_train,y_train)
predictions = nb.predict(X_test)
acc = accuracy_score(y_test,predictions)



print(f'By hand Naive Bayes Classification acurracy is: {acc}')

# here we implement the GaussianNB which is built in Sklearn library, just for comparison
gnb = GaussianNB()
gnb.fit(X_train,y_train)
gnb_pred = gnb.predict(X_test)
acc_2 = accuracy_score(y_test,gnb_pred)

print(f'Scikit Learn Gaussian Naive Bayes Classification acurracy is: {acc_2}')

