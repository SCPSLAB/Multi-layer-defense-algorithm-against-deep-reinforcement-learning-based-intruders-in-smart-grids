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
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
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
from scipy import stats
import seaborn as sns
import pickle
from pylab import rcParams
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.callbacks import ModelCheckpoint, EarlyStopping



dataset =pd.read_csv('Dataset.csv')

dataset=dataset.sample(frac = 1)
dataset


print(dataset.info())


dataset.describe().T


print(dataset['Label'].value_counts())



data = dataset.values
X = data[:,0:46]
y = data[:,46]
X_train,X_test,Y_train,Y_test=train_test_split(X,y,stratify=y,test_size=0.4)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=1)



from sklearn.preprocessing import  StandardScaler, MinMaxScaler

scaler = MinMaxScaler()
#scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

from keras import regularizers

# nb_epoch = 30
# batch_size = 50
# input_dim = X_train.shape[1] #num of columns
# encoding_dim = 100
# hidden_dim = int(encoding_dim / 3) #i.e. 7
# learning_rate = 1e-4

# input_layer = Input(shape=(input_dim, ))
# encoder = Dense(encoding_dim, activation="tanh", activity_regularizer=regularizers.l1(learning_rate))(input_layer)
# encoder = Dense(hidden_dim, activation="relu")(encoder)
# decoder = Dense(hidden_dim, activation='relu')(encoder)
# decoder = Dense(input_dim, activation='tanh')(decoder)
# autoencoder = Model(inputs=input_layer, outputs=decoder)


input_dim = X_train.shape[1]
encoding_dim = 150

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="tanh",activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="tanh")(encoder)
encoder = Dense(int(encoding_dim / 4), activation="tanh")(encoder)
decoder = Dense(int(encoding_dim/ 2), activation='tanh')(encoder)
decoder = Dense(int(encoding_dim), activation='tanh')(decoder)
decoder = Dense(input_dim, activation='tanh')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.summary()

n_epochs = 50
batch_size = 512

autoencoder.compile(optimizer='adam', 
                    loss='mean_squared_error', 
                    #loss='mean_squared_logarithmic_error',
                    metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="autoencoder_fraud.h5",monitor='val_accuracy',
                               verbose=1,
                               save_best_only=True, mode='max')
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)





history = autoencoder.fit(X_train, X_train,
                    epochs=n_epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_test, X_test),
                    verbose=1,
                    callbacks=[checkpointer, tensorboard]).history

autoencoder = load_model('autoencoder_fraud.h5')


plt.plot(history['loss'], linewidth=2, label='Train')
plt.plot(history['val_loss'], linewidth=2, label='Test')
plt.legend(loc='upper right')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
#plt.ylim(ymin=0.70,ymax=1)
plt.show()

plt.plot(history['accuracy'], linewidth=2, label='Train')
plt.plot(history['val_accuracy'], linewidth=2, label='Test')
plt.legend(loc='upper right')
plt.title('Model accuracy')
plt.ylabel('Loss')
plt.xlabel('Epoch')
#plt.ylim(ymin=0.70,ymax=1)
plt.show()


test_x_predictions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - test_x_predictions, 2), axis=1)
error_df = pd.DataFrame({'Reconstruction_error': mse,
                        'True_class': Y_test})
error_df.describe()



false_pos_rate, true_pos_rate, thresholds = roc_curve(error_df.True_class, error_df.Reconstruction_error)
roc_auc = auc(false_pos_rate, true_pos_rate,)

plt.plot(false_pos_rate, true_pos_rate, linewidth=5, label='AUC = %0.3f'% roc_auc)
plt.plot([0,1],[0,1], linewidth=5)

plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.legend(loc='lower right')
plt.title('Receiver operating characteristic curve (ROC)')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df.True_class, error_df.Reconstruction_error)
plt.plot(recall_rt, precision_rt, linewidth=5, label='Precision-Recall curve')
plt.title('Recall vs Precision')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

plt.plot(threshold_rt, precision_rt[1:], label="Precision",linewidth=5)
plt.plot(threshold_rt, recall_rt[1:], label="Recall",linewidth=5)
plt.title('Precision and recall for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Precision/Recall')
plt.legend()
plt.show()



threshold_fixed = .2
groups = error_df.groupby('True_class')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "Fraud" if name == 1 else "Normal")
ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.savefig('AutoEncoder.png', dpi = 1200)
plt.show();

rcParams['figure.figsize'] = 14, 8.7 # Golden Mean
LABELS = ["Normal","Fraud"]
col_list = ["cerulean","scarlet"]# https://xkcd.com/color/rgb/
sns.set(style='white', font_scale=1.75, palette=sns.xkcd_palette(col_list))


pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]
conf_matrix = confusion_matrix(error_df.True_class, pred_y)

plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


rf_pred = model.predict(X_test)
print(accuracy_score(y_test, rf_pred))
print(precision_score(y_test, rf_pred, average='weighted'))
print(recall_score(y_test, rf_pred, average='weighted'))
print(f1_score(y_test, rf_pred, average='weighted'))

