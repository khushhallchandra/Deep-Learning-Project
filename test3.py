import tensorflow as tf
tf.python.control_flow_ops = tf
from __future__ import print_function
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
from numpy import genfromtxt
from sklearn.metrics import mean_squared_error
import math
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import keras 

np.random.seed(1337) 

nb_epoch = 50	
nb_classes = 100
batch_size = 10

nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (7, 7)

nOutFeature = nb_classes
nInFeature = 0#442-198

input_shape = (55, nOutFeature + nInFeature, 1) 



N=200
training_inputs = []
for i in range(1,N+1):
    temp = sp.delete(genfromtxt('../data/battleFin/data/'+str(i)+'.csv', delimiter=','),0,0)
    out = [temp[0][0:442]]
    for i in range(1,55):
        out = np.append(out,[temp[i][0:442]],axis=0)
    training_inputs.append(out)
training_inputs = np.array(training_inputs)

#Normalization 
maxOfFeature = np.zeros(442-198)
for i in range(198,442):
	maxOfFeature[i-198] = training_inputs[:,:,i].max()
	training_inputs[:,:,i] /= maxOfFeature[i-198]

outData = training_inputs[:,:,0:nOutFeature]
inData = training_inputs[:,:,198:198+nInFeature]
training_inputs = np.append(outData,inData,axis=2)

labels = sp.delete(sp.delete(genfromtxt('../data/battleFin/trainLabels.csv', delimiter=','),0,0),0,1)

Y_train = labels[0:180,0:nb_classes]
Y_test = labels[180:200,0:nb_classes]

img_rows = 55
img_cols = nInFeature + nOutFeature
trX = training_inputs[0:180]
teX = training_inputs[180:200]
X_train = trX.reshape(trX.shape[0], img_rows, img_cols, 1)
X_test = teX.reshape(teX.shape[0], img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

Y_train = Y_train.astype('float32')
Y_test = Y_test.astype('float32')

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

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

predictedY_test = model.predict(X_test)
predictedY_train = model.predict(X_train)
print ("Train error(RMSE) is:")
print (math.sqrt(mean_squared_error(predictedY_train,Y_train)))
print ("Test error(RMSE) is:")
print (math.sqrt(mean_squared_error(predictedY_test,Y_test)))


# shift train predictions for plotting
trainPredictPlot = np.empty_like(labels[:,0])
trainPredictPlot[:] = np.nan
trainPredictPlot[0:180] = predictedY_train[:,0]
# shift test predictions for plotting
testPredictPlot = np.empty_like(labels[:,0])
testPredictPlot[:] = np.nan
testPredictPlot[180:] = predictedY_test[:,0]
# plot baseline and predictions

plt.xlabel('No. of days')
plt.ylabel('Price')
plt.title('Change in Stock price vs days')

plt.plot(labels[:,0], label='Actual data')
plt.plot(trainPredictPlot,label='Train predicted ')
plt.plot(testPredictPlot, label='Test predicted')
plt.legend(loc='lower right')
plt.show()