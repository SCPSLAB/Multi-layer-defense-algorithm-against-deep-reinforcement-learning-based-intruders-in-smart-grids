#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, confusion_matrix

data =pd.read_csv('FinalDataset.csv')

npdata = np.array(data)


y = npdata[:, -1]
X = npdata[:,:-1]

y_normal = y[y == 0]
X_normal = X[y == 0]


X_train, X_test, y_train, y_test = train_test_split(X_normal, y_normal, test_size = 0.2)
X_attack = npdata[y == 1]
X_attack = X_attack[:, :-1]
X_test = np.concatenate((X_test, X_attack), axis = 0)


y_test[y_test == 0] = 1
y_test.shape


y2 = np.zeros((X_attack.shape[0],))
y_test = np.concatenate((y_test, y2), axis = 0)
y_test[y_test ==0] = -1


isolation = IsolationForest(n_estimators = 60)
isolation = isolation.fit(X_train)
if_pred = isolation.predict(X_test)
print(accuracy_score(if_pred, y_test))
print(precision_score(if_pred, y_test))
print(recall_score(if_pred, y_test))
print(f1_score(if_pred, y_test))

confusion_matrix(if_pred, y_test)

confusion_matrix( y_test, if_pred)