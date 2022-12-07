#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.utils import resample
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from numpy import mean
from numpy import std
from numpy import dstack
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
import numpy as np
from sklearn.metrics import accuracy_score
from keras.utils.np_utils import to_categorical 
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
import keras
from IPython.display import Image 
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import utils
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import SVC
from imblearn.under_sampling import EditedNearestNeighbours
from array import array
from numpy import where
from keras.models import load_model
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.layers import BatchNormalization
from sklearn.metrics import cohen_kappa_score
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
# importing python utility libraries
import os, sys, random, io, urllib
from datetime import datetime
import random as rd

# importing pytorch libraries
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
# importing python plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('darkgrid')
from IPython.display import Image, display
from keras.layers import Input
from keras.layers import Activation
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from collections import Counter


df1 =pd.read_csv('AttackScenario1.csv')
df2 =pd.read_csv('NormalSamples.csv')
dataset = pd.concat([df1, df2])

dataset=dataset.sample(frac = 1)
dataset


print(dataset.info())
dataset.describe().T

print(dataset['Label'].value_counts())

df_majority = dataset[dataset.Label==0]
df_minority = dataset[dataset.Label==1]
 
df_minority_upsampled = resample(df_minority, 
                                 replace=True,        # sample with replacement
                                 n_samples=353280,    # to match majority class
                                 random_state=123)    # reproducible results
 
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
df_upsampled.Label.value_counts()

normal_mask = df_upsampled['Label']==0
attack_mask = df_upsampled['Label']!=0

dataset.drop('Label',axis=1,inplace=True)

df_normal = df_upsampled[normal_mask]
df_attack = df_upsampled[attack_mask]

print(f"Normal count: {len(df_normal)}")
print(f"Attack count: {len(df_attack)}")

x_normal = df_normal.values
x_attack = df_attack.values

x_normal_train, x_normal_test = train_test_split(x_normal, test_size=0.3, random_state=12)

print(f"Normal train count: {len(x_normal_train)}")
print(f"Normal test count: {len(x_normal_test)}")


x_normal_train.shape, x_normal_test.shape

sc = StandardScaler()

x_normal_train = sc.fit_transform(x_normal_train)
x_normal_test = sc.fit_transform(x_normal_test)
x_normal = sc.fit_transform(x_normal)
x_attack = sc.fit_transform(x_attack)

x_normal_train = pd.DataFrame(x_normal_train)
x_normal_test = pd.DataFrame(x_normal_test)
x_attack = pd.DataFrame(x_attack)
x_normal = pd.DataFrame(x_normal)

def create_dataset(X, time_steps):
    Xs = []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
    return np.array(Xs)


TIME_STEPS = 1
Xnormal_train = create_dataset(x_normal_train, TIME_STEPS)
print(Xnormal_train.shape)


Xnormal_test = create_dataset(x_normal_test, TIME_STEPS)
print(Xnormal_test.shape)


Xattack = create_dataset(x_attack, TIME_STEPS)
print(Xattack.shape)

Xnormal = create_dataset(x_normal,TIME_STEPS)
print(Xattack.shape)


model = keras.Sequential()
model.add(keras.layers.LSTM(units=64, input_shape=(Xnormal_train.shape[1], Xnormal_train.shape[2])))
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.RepeatVector(n=Xnormal_train.shape[1]))
model.add(keras.layers.LSTM(units=32, return_sequences=True))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=Xnormal_train.shape[2])))
model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(
    Xnormal_train, Xnormal_train,
    epochs=15,
    batch_size=50,
    validation_split=0.2, 
    shuffle = False
)


plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend();

X_train_pred = model.predict(Xnormal_train)

train_mae_loss = np.mean(np.abs(X_train_pred - Xnormal_train), axis=1)
train_mae_loss.shape

sns.distplot(train_mae_loss, bins=10, kde=True);

X_test_pred = model.predict(Xnormal_test)

test_mae_loss = np.mean(np.abs(X_test_pred - Xnormal_test), axis=1)
test_mae_loss.shape

sns.distplot(test_mae_loss, bins=10, kde=True);


X_normal_pred = model.predict(Xnormal)

normal_mae_loss = np.mean(np.abs(X_normal_pred - Xnormal), axis=1)
normal_mae_loss.shape

sns.distplot(normal_mae_loss, bins=10, kde=True);

X_attack_pred = model.predict(Xattack)

attack_mae_loss = np.mean(np.abs(X_attack_pred - Xattack), axis=1)
attack_mae_loss.shape


sns.distplot(attack_mae_loss, bins=10, kde=True);


score1 = model.evaluate(Xnormal_train, Xnormal_train)
print("\n Sample Loss(MAE) & Accuracy Scores (Train):", score1[0], score1[1], "\n") 

score2 = model.evaluate(Xnormal_test, Xnormal_test)
print("\nOut of Sample Loss(MAE) & Accuracy Scores (Test):", score2[0], score2[1], "\n") 

score3 = model.evaluate(Xattack, Xattack)
print("\nAttack Underway Loss(MAE) & Accuracy Scores (Anomaly):", score3[0], score3[1], "\n")