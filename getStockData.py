from numpy import genfromtxt
import numpy as np
def series(sample=600):
	print "returned a 5 X sample numpy array"
	print "series(nSample) default size of nSample is 600"
	data = genfromtxt('stockData.csv', delimiter=',')
	return np.stack((data[0][0:sample],data[1][0:sample],data[2][0:sample],data[3][0:sample],data[4][0:sample]))