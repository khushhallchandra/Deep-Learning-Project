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
length = 1
data = np.genfromtxt(data_folder+currency+'.csv', delimiter=',')[:365,1]
def get_data(currency,length):
	data = np.genfromtxt(data_folder+currency+'.csv', delimiter=',')[:365,1]
	X,Y = [],[]
	for i in range(len(data)-length-1):
		X.append(-data[i:i+length]+data[i+1:i+1+length])
		Y.append(data[i+length+1]-data[i:i+length])
	return np.array(X),np.array(Y)

def convert_binary(X):
	new_X = np.zeros(X.shape[0]-1)
	for i in range(X.shape[0]-1):
		new_X[i] = (X[i+1]-X[i])>0
	return new_X


def build_model(length):
	model = Sequential()
	model.add(LSTM(20,input_dim=length))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model


datasetX, datasetY = get_data(currency,length)
train_size = int(datasetX.shape[0] * 0.7)
test_size = datasetX.shape[0] - train_size

trainX, testX = datasetX[0:train_size,:], datasetX[train_size:len(datasetX),:]
trainY, testY = datasetY[0:train_size], datasetY[train_size:len(datasetX)]

#print trainX

trainX = np.reshape(trainX,(trainX.shape[0],1,trainX.shape[1]))
testX = np.reshape(testX,(testX.shape[0],1,testX.shape[1]))

model = build_model(length)
model.fit(trainX, trainY, nb_epoch=40, batch_size=1)

# make predictions
trainout = model.predict(trainX)
testout = model.predict(testX)

# calculate Binary error

binTestout = testout>0
binTestTruth = testY>0

binTrainout = trainout>0
binTrainTruth = trainY>0

binTestTruth_new = np.zeros((len(binTestTruth),1))
for i in range(len(binTestTruth)):
        binTestTruth_new[i] = binTestTruth[i]
zz = (binTestTruth_new==binTestout)
testScore = sum(zz)[0]*1.0/binTestout.shape[0]

binTrainTruth_new = np.zeros((len(binTrainTruth),1))
for i in range(len(binTrainTruth)):
	binTrainTruth_new[i] = binTrainTruth[i]
zz = (binTrainTruth_new==binTrainout)
trainScore = sum(zz)[0]*1.0/binTrainout.shape[0]

#trainPredict = scaler.inverse_transform(trainPredict)
#trainY = scaler.inverse_transform([trainY])
#testPredict = scaler.inverse_transform(testPredict)
#testY = scaler.inverse_transform([testY])

trainScorer = math.sqrt(mean_squared_error(binTrainout, binTrainTruth))
testScorer = math.sqrt(mean_squared_error(binTestout, binTestTruth))

print('Train Score: %.2f Binary error' % (trainScore))
print('Test Score: %.2f Binary error' % (testScore))
print('Train Score: %.2f RMSE error' % (trainScorer))
print('Test Score: %.2f RMSE error' % (testScorer))

plt.subplot(1,2,1)
plt.plot(testout,'r')
plt.subplot(1,2,1)
plt.plot(list(testY),'b')

plt.subplot(1,2,2)
plt.plot(trainout,'r')
plt.subplot(1,2,2)
plt.plot(list(trainY),'b')
plt.show()

