#!/usr/bin/env python
# coding: utf-8



from sklearn.datasets import make_blobs
from matplotlib import pyplot
from numpy import where
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential, Model, Input
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from sklearn.datasets import make_blobs
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import Callback
from keras.optimizers import SGD
from keras import backend
from math import pi
from math import cos
from math import floor
from keras.models import load_model
from matplotlib import pyplot
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


X, y = make_blobs(n_samples=2000, centers=3, n_features=2, cluster_std=2, random_state=1)

y = to_categorical(y)
# split into train and test
n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]


model = Sequential()
# model.add(Dense(32, input_dim=2, activation='relu'))
model.add(Dense(8, input_dim=2, activation='relu'))
model.add(Dense(3, activation='softmax'))
opt = SGD(lr=0.05, momentum=0.8)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=50)
# checkpoint = ModelCheckpoint('model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]


history = model.fit(trainX, trainy, validation_data=(testX, testy),callbacks=[early_stopping_monitor], epochs=200, verbose=1)


_, train_acc = model.evaluate(trainX, trainy, verbose=0)
_, test_acc = model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))



plt.xlabel('Epochs')
plt.ylabel('Accuracy')
# plt.title('Train vs. Test Accuracy')

plt.plot(history.history['val_accuracy'], color = 'red', label = 'Test')
# plt.plot(history.history['accuracy'], color = 'black', label = 'Train')

plt.legend(loc='best', shadow = True, fontsize = 'medium')
plt.savefig('dnn8.png', dpi = 300)


plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Test')
plt.legend()
plt.show()
plt.legend(loc='best', shadow = True, fontsize = 'medium')
plt.savefig('dnn.png', dpi = 300)


def cosine_annealing(epoch, n_epochs, n_cycles, lrate_max):
    epochs_per_cycle = floor(n_epochs/n_cycles)
    cos_inner = (pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
    return lrate_max/2 * (cos(cos_inner) + 1)


n_epochs = 100
n_cycles = 5
lrate_max = 0.01
series = [cosine_annealing(i, n_epochs, n_cycles, lrate_max) for i in range(n_epochs)]

# plot series
pyplot.plot(series)
pyplot.show()

class CosineAnnealingLearningRateSchedule(Callback):
    def __init__(self, n_epochs, n_cycles, lrate_max, verbose=0):
        self.epochs = n_epochs
        self.cycles = n_cycles
        self.lr_max = lrate_max
        self.lrates = list()

    def cosine_annealing(self, epoch, n_epochs, n_cycles, lrate_max):
        epochs_per_cycle = floor(n_epochs/n_cycles)
        cos_inner = (pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
        return lrate_max/2 * (cos(cos_inner) + 1)

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.cosine_annealing(epoch, self.epochs, self.cycles, self.lr_max)
        backend.set_value(self.model.optimizer.lr, lr)
        self.lrates.append(lr)

n_epochs = 400
n_cycles = n_epochs / 50
ca = CosineAnnealingLearningRateSchedule(n_epochs, n_cycles, 0.01)
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=n_epochs, verbose=0, callbacks=[ca])

pyplot.plot(ca.lrates)
pyplot.show()


pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()

class SnapshotEnsemble(Callback):
    def __init__(self, n_epochs, n_cycles, lrate_max, verbose=0):
        self.epochs = n_epochs
        self.cycles = n_cycles
        self.lr_max = lrate_max
        self.lrates = list()

    def cosine_annealing(self, epoch, n_epochs, n_cycles, lrate_max):
        epochs_per_cycle = floor(n_epochs/n_cycles)
        cos_inner = (pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
        return lrate_max/2 * (cos(cos_inner) + 1)

    def on_epoch_begin(self, epoch, logs={}):
        lr = self.cosine_annealing(epoch, self.epochs, self.cycles, self.lr_max)
        backend.set_value(self.model.optimizer.lr, lr)
        self.lrates.append(lr)

    def on_epoch_end(self, epoch, logs={}):
        epochs_per_cycle = floor(self.epochs / self.cycles)
        if epoch != 0 and (epoch + 1) % epochs_per_cycle == 0:
            filename = "snapshot_model_%d.h5" % int((epoch + 1) / epochs_per_cycle)
            self.model.save(filename)
            print('>saved snapshot %s, epoch %d' % (filename, epoch))


X, y = make_blobs(n_samples=1100, centers=3, n_features=2, cluster_std=2, random_state=2)
y = to_categorical(y)
n_train = 100
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
model = Sequential()
model.add(Dense(50, input_dim=2, activation='relu'))
model.add(Dense(3, activation='softmax'))
opt = SGD(momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
n_epochs = 500
n_cycles = n_epochs / 50
ca = SnapshotEnsemble(n_epochs, n_cycles, 0.01)
model.fit(trainX, trainy, validation_data=(testX, testy), epochs=n_epochs, verbose=0, callbacks=[ca])

class SnapshotEnsemble(Callback):
    def __init__(self, n_epochs, n_cycles, lrate_max, verbose=0):
        self.epochs = n_epochs
        self.cycles = n_cycles
        self.lr_max = lrate_max
        self.lrates = list()

    def cosine_annealing(self, epoch, n_epochs, n_cycles, lrate_max):
        epochs_per_cycle = floor(n_epochs/n_cycles)
        cos_inner = (pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
        return lrate_max/2 * (cos(cos_inner) + 1)

    def on_epoch_begin(self, epoch, logs={}):
        lr = self.cosine_annealing(epoch, self.epochs, self.cycles, self.lr_max)
        backend.set_value(self.model.optimizer.lr, lr)
        self.lrates.append(lr)

    def on_epoch_end(self, epoch, logs={}):
        epochs_per_cycle = floor(self.epochs / self.cycles)
        if epoch != 0 and (epoch + 1) % epochs_per_cycle == 0:
            filename = "snapshot_model_%d.h5" % int((epoch + 1) / epochs_per_cycle)
            self.model.save(filename)
            print('>saved snapshot %s, epoch %d' % (filename, epoch))

X, y = make_blobs(n_samples=1100, centers=3, n_features=2, cluster_std=2, random_state=2)
y = to_categorical(y)
n_train = 100
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
model = Sequential()
model.add(Dense(50, input_dim=2, activation='relu'))
model.add(Dense(3, activation='softmax'))
opt = SGD(momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
n_epochs = 500
n_cycles = n_epochs / 50
ca = SnapshotEnsemble(n_epochs, n_cycles, 0.01)
model.fit(trainX, trainy, validation_data=(testX, testy), epochs=n_epochs, verbose=0, callbacks=[ca])