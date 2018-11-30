import numpy as np
from scipy.io import loadmat
from __init__ import *

def nn_input(key):
	data = loadmat('./data/Sol_'+str(key)+'.mat')
	Mat_1 = np.array(data['Expression1'])
	Mat_2 = np.array(data['Expression2'])

	Mat_3 = np.array(data['Expression3'])
	mats = [Mat_1 , Mat_2, Mat_3]
	side = np.genfromtxt('./data/Side_'+str(key)+'.csv',delimiter=',')
	window_size = Mat_1.shape

	X_main = np.zeros((window_size[0], window_size[1], FEATURES))


	X_side = np.zeros(SIDE_FEATURES)
	for i in range(SIDE_FEATURES):
		X_side[i] = side[i]
	#print np.shape(X_main)
	for i in range(FEATURES):
		X_main[:,:,i] += mats[i]
	Y  = np.genfromtxt('./data/Sol_'+str(key)+'.csv',delimiter=',') 

	#print Y
	return [X_main, X_side], Y


# if __name__ == '__main__':
# 	nn_input(1)