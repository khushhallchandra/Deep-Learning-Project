import tensorflow as tf
tf.python.control_flow_ops = tf
import matplotlib.pyplot as plt
import scipy as sp
from __future__ import print_function
import numpy as np
from numpy import genfromtxt
from sklearn.metrics import mean_squared_error
import math
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, LSTM
from keras.layers.core import Reshape
from keras.utils import np_utils
from keras import backend as K
import keras 
 
np.random.seed(1337) 
 
nb_epoch = 50  
nb_classes = 20
batch_size = 10
 
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (7, 7)
 
input_shape = (55, 20, 1) 
 
#Reading data
N=200
training_inputs = []
for i in range(1,N+1):
    temp = sp.delete(genfromtxt('data/'+str(i)+'.csv', delimiter=','),0,0)
    out = [temp[0][0:20]]
    for i in range(1,55):
        out = np.append(out,[temp[i][0:20]],axis=0)
    training_inputs.append(out)
training_inputs = np.array(training_inputs)
 
labels = sp.delete(sp.delete(genfromtxt('trainLabels.csv', delimiter=','),0,0),0,1)
 
 #spliting data into test and train
Y_train = labels[0:180,0:20]
Y_test = labels[180:200,0:20]
 
img_rows = 55
img_cols = 20
trX = training_inputs[0:180]
teX = training_inputs[180:200]
X_train = trX.reshape(trX.shape[0], img_rows, img_cols, 1)
X_test = teX.reshape(teX.shape[0], img_rows, img_cols, 1)
 
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
 
Y_train = Y_train.astype('float32')
Y_test = Y_test.astype('float32')
 
# Creating Convolution model
model = Sequential()
 
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))
# model.add(Reshape((84,32)))
# model.add(LSTM(128))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
# model.add(Activation('relu'))
 
# model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['mean_squared_error'])
sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='mean_squared_error', optimizer=sgd)
model.compile(loss='mean_squared_error', optimizer='rmsprop')
 
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_split=0.05)
 
# model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
# score = model.evaluate(X_test, Y_test)
# print (score)

trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)

trainScore = math.sqrt(mean_squared_error(Y_train, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(Y_test, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

trainPredictPlot = np.zeros(200)
trainPredictPlot[:] = np.nan
trainPredictPlot[0:180] = trainPredict[:,0]
# shift test predictions for plotting
testPredictPlot = np.zeros(200)
testPredictPlot[:] = np.nan
testPredictPlot[180:] = testPredict[:,0]

# ploting the time series graph of price change for a financial instrument:
plt.xlabel('No. of days')
plt.ylabel('Price change')
plt.title('Change in Stock price vs days')

plt.plot(labels[:,0], label='Actual data')
plt.plot(trainPredictPlot,label='Train predicted ')
plt.plot(testPredictPlot, label='Test predicted')
plt.legend(loc='lower right')
plt.show()