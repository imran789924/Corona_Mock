#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 12:54:44 2020

@author: imran
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('/home/imran/PROGRAMS/Machine Learning/Practice/corona/data.csv')

dataset = dataset.drop(['Timestamp'], axis = 1)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 7:].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 0] = le.fit_transform(X[:, 0])
X[:, 1] = le.fit_transform(X[:, 1])
X[:, 2] = le.fit_transform(X[:, 2])
X[:, 3] = le.fit_transform(X[:, 3])
X[:, 4] = le.fit_transform(X[:, 4])
X[:, 5] = le.fit_transform(X[:, 5])
X[:, 6] = le.fit_transform(X[:, 6])


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(X_train, y_train)
predictoin = regressor.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictoin)
