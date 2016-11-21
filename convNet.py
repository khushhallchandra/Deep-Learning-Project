import tensorflow as tf
tf.python.control_flow_ops = tf
import scipy as sp
from __future__ import print_function
import numpy as np
from numpy import genfromtxt

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import keras 

nb_epoch = 20
nb_classes = 198
batch_size = 10

nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

input_shape = (55, 198, 1) 


N=200
training_inputs = []
for i in range(1,N+1):
    temp = sp.delete(genfromtxt('../data/battleFin/data/'+str(i)+'.csv', delimiter=','),0,0)
    out = [temp[0][0:198]]
    for i in range(1,55):
        out = np.append(out,[temp[i][0:198]],axis=0)
    training_inputs.append(out)
training_inputs = np.array(training_inputs)

labels = sp.delete(sp.delete(genfromtxt('../data/battleFin/trainLabels.csv', delimiter=','),0,0),0,1)

Y_train = labels[0:180]
Y_test = labels[180:200]

img_rows = 55
img_cols = 198
trX = training_inputs[0:180]
teX = training_inputs[180:200]
X_train = trX.reshape(trX.shape[0], img_rows, img_cols, 1)
X_test = teX.reshape(teX.shape[0], img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

Y_train = Y_train.astype('float32')
Y_test = Y_test.astype('float32')

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
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
model.add(Activation('relu'))

# model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['mean_squared_error'])
sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='mean_squared_error', optimizer=sgd)
model.compile(loss='mean_squared_error', optimizer='rmsprop')

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_split=0.05,)

# model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test)

print('Test score:', score[0])
print('Test accuracy:', score[1])
