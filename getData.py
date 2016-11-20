import matplotlib.pyplot as plt
import numpy as np
from yahoo_finance import Share

def movingAverage(data,step=5):
	#data is one dimensional numpy array
	print "default step size is 5"
	length = data.size
	out = []
	defaultStep = 5
	for i in range(step-defaultStep):
		out.append(0)
	i=0
	while(i+step<=length):
		temp = data[i:i+step]
		i+=1
		out.append(np.mean(temp))
	return out

def series(p = 0):
	yahoo = Share('YHOO')

	data =  yahoo.get_historical('2013-12-19', '2016-09-29')

	#length of time series
	length = len(data)
	nFeatures = 5
	outData = np.zeros([nFeatures,length])

	for i in range(length):
		outData[0][i] = float(data[i]['Close'])
		outData[1][i] = float(data[i]['Open'])
		outData[2][i] = float(data[i]['High'])
		outData[3][i] = float(data[i]['Low'])
		outData[4][i] = float(data[i]['Volume'])
	print "return a numpy array of size 5x700"
	print "if you want to plot data, pass argument series(1)"
	if(p):
		plt.plot(outData[0],'r',label='Close')
		plt.plot(outData[1],'g',label='Open')
		plt.plot(outData[2],'b',label='High')
		plt.plot(outData[3],'k',label='Low')
		plt.legend()
		plt.show()
	return outData