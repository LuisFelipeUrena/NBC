import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

from nbc import NaiveBayesClassifier

X, y = datasets.make_classification(n_samples=1000, n_features =10, n_classes=2,random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

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

