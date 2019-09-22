import cv2
import numpy as np
import os
import re

class loader_qa:
	def __init__(self,filedir,train_val_test=(0.8,0.1,0.1)):
		# question answer onehot loader
		self.onehot=np.load(filedir)
		self.indices=list(range(len(self.onehot)))
		
		trainnum,valnum,testnum=list(map(lambda x: round(x*len(self.onehot)),train_val_test))
		self.trains,self.vals,self.tests=self.onehot[:trainnum],self.onehot[trainnum:trainnum+valnum],self.onehot[trainnum+valnum:trainnum+valnum+testnum]
		
		# create indices for shuffling and sampling
		self.indices=list(range(len(self.trains)))
		self.batch_counter=0
		self.sample_counter=0

	def preprocess(self,inputs):
		# flatten
		return np.reshape(inputs,[np.shape(inputs)[0],-1])
	
	def train_next_batch(self,batch_size=10,batch_per_epoch=0):
		batch_x=[]
		batch_y=[]
		np.random.shuffle(self.indices)
		while True:
			sample=self.trains[self.indices[self.sample_counter]]
			batch_x.append(sample)
			batch_y.append(sample)
			
			self.sample_counter+=1
			if self.sample_counter%batch_size==0 or self.sample_counter==len(self.indices):
				yield [batch_x,batch_y]
				batch_x=[]
				batch_y=[]
				self.batch_counter+=1
			if batch_per_epoch>0 and self.batch_counter==batch_per_epoch or self.sample_counter==len(self.indices):
				np.random.shuffle(self.indices)
				self.sample_counter=0

	def get_single(self,index):
		return self.onehot[index]

	def get_whole(self):
		return self.onehot
