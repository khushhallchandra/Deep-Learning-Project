
import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


data_folder = 'data/'
currency = 'euro'
length = 4
data = np.genfromtxt(data_folder+currency+'.csv', delimiter=',')[:365,1]
def get_data(currency,length):
	data = np.genfromtxt(data_folder+currency+'.csv', delimiter=',')[:365,1]
	X,Y = [],[]
	for i in range(len(data)-length):
		X.append(data[i:i+length])
		Y.append(data[i+length])
	return np.array(X),np.array(Y)

dataset, dataset_Y = get_data(currency,length)
train_size = int(dataset.shape[0] * 0.7)
test_size = dataset.shape[1] - train_size

trainX, testX = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
trainY, testY = dataset_Y[0:train_size], dataset_Y[train_size:len(dataset)]

trainX = np.reshape(trainX,(trainX.shape[0],1,trainX.shape[1]))
testX = np.reshape(testX,(testX.shape[0],1,testX.shape[1]))

print trainX.shape
print testX.shape
print trainY.shape
print testY.shape

model = Sequential()
model.add(LSTM(1, input_dim=length))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=20, batch_size=1)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

#trainPredict = scaler.inverse_transform(trainPredict)
#trainY = scaler.inverse_transform([trainY])
#testPredict = scaler.inverse_transform(testPredict)
#testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

plt.plot(trainPredict,'r')
plt.plot(list(data[:252]),'b')
plt.show()



