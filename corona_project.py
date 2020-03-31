#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 12:54:44 2020

@author: imran
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, roc_curve




def generate_model_details(y_test, y_pred):
    print("Accuracy = ", accuracy_score(y_test, y_pred))
    print("Precision = ", precision_score(y_test, y_pred))
    print("Recall = ", recall_score(y_test, y_pred))
    print("F1 score = ", f1_score(y_test, y_pred))


def generate_roc_auc_curve(classifier, X_test):
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label='AUC ROC curve')
    plt.legend(loc=4)
    plt.show()
    pass




dataset = pd.read_csv('/home/imran/PROGRAMS/Machine Learning/Corona/corona/corona_2.csv')

dataset = dataset.drop(['Timestamp'], axis = 1)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 9:].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 0] = le.fit_transform(X[:, 0])
X[:, 1] = le.fit_transform(X[:, 1])
X[:, 2] = le.fit_transform(X[:, 2])
X[:, 3] = le.fit_transform(X[:, 3])
X[:, 4] = le.fit_transform(X[:, 4])
X[:, 5] = le.fit_transform(X[:, 5])
X[:, 6] = le.fit_transform(X[:, 6])
X[:, 7] = le.fit_transform(X[:, 7])
X[:, 8] = le.fit_transform(X[:, 8])
le_y = LabelEncoder()
y = le_y.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)


from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Reds)
plt.colorbar()
classnames=['Not corona', 'corona ']
tick_marks = np.arange(len(classnames))
plt.xticks(tick_marks, classnames, rotation=70)
plt.yticks(tick_marks, classnames)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


generate_model_details(y_test, y_pred)
generate_roc_auc_curve(regressor, X_test)


unique, counts = np.unique(y_train, return_counts=True)
print (np.asarray((unique, counts)).T)


'''
from sklearn.metrics import accuracy_score
accuracy_rate = accuracy_score(y_test, predictoin)



import scipy.sparse
mat = scipy.sparse.eye(3)
temp = pd.DataFrame.sparse.from_spmatrix(mat)
'''