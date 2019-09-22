import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def shuffle_together(arr1,arr2):
	c=list(zip(arr1,arr2))
	np.random.shuffle(c)
	arr_1,arr_2=zip(*c)
	return arr_1,arr_2

def acc(label,pred):
	# input predictions and labels and calculate accuracy
	correct=0
	for i in range(len(label)):
		if np.argmax(label[i])==np.argmax(pred[i]):
			correct+=1
	res=correct/len(label)
	return res

def accuracy(label,pred):
	# onehot labels for the qa
	# pred shape: [batch,question_num,answer_num]
	# number of questions predicted correctly
	res=np.zeros(shape=np.shape(pred)[0])
	for i,p in enumerate(pred):
		for j in range(np.shape(pred)[1]):
			if np.argmax(pred[i][j])==np.argmax(label[i][j]):
				res[i]+=1
	return res
	

def mask_generator(masklength,hintnum=1):
	res=np.zeros(masklength)
	choices=np.random.choice(list(range(masklength)),hintnum,replace=False)
	res[choices]=1
	return res

