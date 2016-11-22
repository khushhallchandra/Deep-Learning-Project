# LSTM for international airline passengers problem with window regression framing
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
from keras.models import Sequential
from keras.layers import Dense ,Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from numpy import genfromtxt

# reading data from database
def getdata():
	dataX = np.zeros((200,55,442))
	for i in range(200):
		data = genfromtxt( ('data/'+str(i+1)+'.csv'), delimiter=',')
		dataX[i,:,:] = data[1:,:]
	data = genfromtxt( 'trainLabels.csv', delimiter=',')
	dataY = data[1:,1:]
	return dataX, dataY
# fix random seed for reproducibility
np.random.seed(7)
# loading the dataset
dataX, dataY = getdata()

#Normalizing the output
# scaler = MinMaxScaler(feature_range=(0, 1))
# dataY = scaler.fit_transform(dataY1[:,0:20])

# split into train and test sets
train_size = int(dataY.shape[0] * 0.9)
test_size = dataY.shape[0] - train_size

trainX, trainY = dataX[0:train_size,:,0:20],dataY[0:train_size,0:20]  #create_dataset(train,vol_train, look_back)
testX, testY = dataX[train_size:,:,0:20],dataY[train_size:,0:20]   #create_dataset(test,vol_test, look_back)

# Smoothning the labels
# trainY1 =trainY
# for i in range(trainY.shape[0]):
# 	p = max(i-1,0)
# 	q = min(i+2,trainY.shape[0])
# 	trainY1[i,:] = np.sum(trainY[p:q,:],0)/(q-p)

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(25, input_shape=(55,20)))
# model.add(LSTM(25 ,activation='relu'))
model.add(Dense(20))
model.compile(loss='mean_squared_error', optimizer='rmsprop')
model.fit(trainX, trainY, nb_epoch=50, batch_size=10, verbose=1, validation_split=0.05)
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
# trainPredict = scaler.inverse_transform(trainPredict)
# trainY = scaler.inverse_transform(trainY)
# testPredict = scaler.inverse_transform(testPredict)
# testY = scaler.inverse_transform(testY)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataY[:,0])
trainPredictPlot[:] = np.nan
trainPredictPlot[0:train_size] = trainPredict[:,0]

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataY[:,0])
testPredictPlot[:] = np.nan
testPredictPlot[train_size:] = testPredict[:,0]

# ploting the time series graph of price change for a financial instrument:
plt.xlabel('No. of days')
plt.ylabel('Price change')
plt.title('Change in Stock price vs days')

plt.plot(dataY[:,0], label='Actual data')
plt.plot(trainPredictPlot,label='Train predicted ')
plt.plot(testPredictPlot, label='Test predicted')
plt.legend(loc='lower right')
plt.show()
