#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 14:58:18 2018

@author: mshokry
"""

from keras.models import Sequential
from keras.layers.core import Activation, Dropout,Dense
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix,log_loss
import keras.losses as losses
import pandas as pd
import numpy as np
import matplotlib.pylab as plt 

data = pd.read_csv('DAT_ASCII_EURUSD_M1_201806.csv',sep=';')

dataset = data.iloc[:,1].values
scaler = StandardScaler() 
scaler.fit(dataset)
dataset = scaler.transform(dataset)
plt.plot(dataset)
plt.xlabel('time period')
plt.ylabel('normalized series value')

def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    for i in range(len(series)-window_size):
        X.append(series[i:i+window_size])
    y=series[window_size:]
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X))
    y = np.asarray(y)
    y.shape = (len(y),1)
    return X,y
window_size = 40
X,y = window_transform_series(series = dataset,window_size = window_size)
# split our dataset into training / testing sets
train_test_split = int(np.ceil(2*len(y)/float(3)))   # set the split point
# partition the training set
X_train = X[:train_test_split,:]
y_train = y[:train_test_split]
# keep the last chunk for testing
X_test = X[train_test_split:,:]
y_test = y[train_test_split:]
# NOTE: to use keras's RNN LSTM module our input must be reshaped to [samples, window size, stepsize] 
X_train = np.asarray(np.reshape(X_train, (X_train.shape[0], window_size, 1)))
X_test = np.asarray(np.reshape(X_test, (X_test.shape[0], window_size, 1)))

### TODO: create required RNN model
# import keras network libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras

# given - fix random seed - so we can all reproduce the same results on our default time series
np.random.seed(0)


# TODO: implement build_part1_RNN in my_answers.py
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(20, input_shape=(window_size,1)))
    model.add(Dense(1,activation='linear'))
    return model

model = build_part1_RNN(window_size)
model.compile(loss='mean_squared_error', optimizer='rmsprop')

model.load_weights('bes.h5')
# generate predictions for training
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
# print out training and testing errors
training_error = model.evaluate(X_train, y_train, verbose=0)
print('training error = ' + str(training_error))

testing_error = model.evaluate(X_test, y_test, verbose=0)
print('testing error = ' + str(testing_error))
### Plot everything - the original series as well as predictions on training and testing sets

def predict_next(model,data,seconds):     
    # create output
    predicted_val = np.zeros(( seconds))
    x_test = np.zeros((1, window_size, 1))
    x_test[0] = data
    for i in range(seconds):        
        # make this round's prediction
        test_predict = model.predict(x_test,verbose = 0)[0]
        #print(test_predict,test_predict.shape,predicted_val.shape)
        # update predicted_chars and input
        predicted_val[i] = test_predict
        #predicted_val= np.append(predicted_val,test_predict)
        #print(x_test.shape)
        x_test[0,:-1,0] = x_test[0,1:,0]
        x_test[0,-1,0] = test_predict
        #print(x_test.shape)
        #input_chars = input_chars[1:]
    return predicted_val

vals = 20
pre = 10
p2 = np.zeros(( vals,pre))
for i in range(vals):
    t2 = predict_next(model,X_train[i],pre)
    p2[i] = t2
# plot original series
plt.plot(dataset[:(window_size+1)+vals+pre],color = 'k')
# plot training set prediction
for i in range(vals):
# plot testing set prediction
    location = window_size+i
    plt.plot(np.arange(location,location + len(t2),1),p2[i],color = 'r')

vals = 20
pre = 10
start = 50
p2 = np.zeros(( vals,pre))
for i in range(vals):
    t2 = predict_next(model,X_train[start+i],pre)
    p2[i] = t2
# plot training set prediction
for i in range(vals):
# plot testing set prediction
    location = start+window_size+i
    plt.plot(np.arange(location,location + len(t2),1),p2[i],color = 'g')

# pretty up graph
plt.xlabel('day')
plt.ylabel('(normalized) price of Apple stock')
plt.legend(['original series','training fit','testing fit'],loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

