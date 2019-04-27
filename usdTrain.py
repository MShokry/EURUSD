#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 19:38:07 2018

@author: mshokry
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 13:26:36 2018

@author: mshokry

https://github.com/SiaFahim/lstm-crypto-predictor/blob/master/lstm_crypto_price_prediction.ipynb
https://medium.com/@siavash_37715/how-to-predict-bitcoin-and-ethereum-price-with-rnn-lstm-in-keras-a6d8ee8a5109
https://www.kaggle.com/pablocastilla/predict-stock-prices-with-lstm
https://github.com/jgpavez/LSTM---Stock-prediction
https://github.com/logan4/Forex-Price-Predictor
https://github.com/llSourcell/Stock_Market_Prediction/blob/master/Generating%20Different%20Models.ipynb
https://machinelearningmastery.com/time-series-prediction-with-deep-learning-in-python-with-keras/

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

data = pd.read_csv('EURUSD601.csv',sep=',')

dataset = data.iloc[:,3].values
dataset = dataset.reshape(-1, 1)

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
    model.add(LSTM(100, input_shape=(window_size,1)))
    model.add(Dense(1,activation='linear'))
    return model

model = build_part1_RNN(window_size)
# build model using keras documentation recommended optimizer initialization
optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# compile the model
model.compile(loss='mean_squared_error', optimizer=optimizer)
model.fit(X_train, y_train, epochs=5, batch_size=100, verbose=2)
model.save_weights('bes.h5')


# generate predictions for training
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
# print out training and testing errors
training_error = model.evaluate(X_train, y_train, verbose=0)
print('training error = ' + str(training_error))

testing_error = model.evaluate(X_test, y_test, verbose=0)
print('testing error = ' + str(testing_error))
### Plot everything - the original series as well as predictions on training and testing sets
import matplotlib.pyplot as plt
#%matplotlib inline

# plot original series
plt.plot(dataset,color = 'k')

# plot training set prediction
split_pt = train_test_split + window_size 
plt.plot(np.arange(window_size,split_pt,1),train_predict,color = 'b')

# plot testing set prediction
plt.plot(np.arange(split_pt,split_pt + len(test_predict),1),test_predict,color = 'r')

# pretty up graph
plt.xlabel('day')
plt.ylabel('(normalized) price of Apple stock')
plt.legend(['original series','training fit','testing fit'],loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


