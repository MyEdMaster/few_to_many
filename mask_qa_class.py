# This is the mask question-answer model class definition
# time: 16/08/2019
# *mym*
import tensorflow as tf
import numpy as np
import basic_layers as ml
import data_loader
import os
import utils

class mask_qa:
	def __init__(self,restoring_epoch=0,namescope="mask_qa",model_dir='./models/'):
		self.sess=tf.Session()
		self.namescope=namescope
		self.model_dir=model_dir
		self.sh=(27,4)
		self.restoring_epoch=restoring_epoch
		
		if not os.path.exists(self.model_dir):
			os.mkdir(self.model_dir)
		
		if restoring_epoch<=0:
			print("Choosing not to restore model. Please notice to run 'obj.train()' first to get a trained model before deploy.")
			## [model definition] ##
			self.lr=tf.placeholder(dtype=tf.float32,name="lr")
			self.x=tf.placeholder(dtype=tf.float32,shape=[None,*self.sh],name="x")
			self.y=tf.placeholder(dtype=tf.float32,shape=[None,*self.sh],name="y")
			self.mask=tf.placeholder(dtype=tf.float32,shape=[self.sh[0],1],name="mask")
			self.mask_switch=tf.placeholder(dtype=tf.bool,name="mask_switch")
			with tf.variable_scope(self.namescope):
				inputs=self.mask*self.x
				mask_trainable=tf.get_variable(name="mask_trainable",shape=[self.sh[0],1],trainable=True)
				inputs=tf.cond(self.mask_switch,lambda: mask_trainable*inputs,lambda: inputs)
				inputs=tf.reshape(inputs,shape=[-1,inputs.get_shape()[1].value*inputs.get_shape()[2].value],name="masked_inputs")
				fc1=ml.fc_layer(inputs,units=500,activation=tf.nn.relu,name="fc1")
				fc2=ml.fc_layer(fc1,units=self.sh[0]*self.sh[1],name="fc2")
				outputs=tf.reshape(fc2,shape=[-1,*self.y.get_shape()[1:]],name="outputs")
			
			self.loss=tf.reduce_mean(tf.reduce_sum([tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs[:,i,:],labels=self.y[:,i,:]) for i in range(self.sh[0])]),name="loss")
			self.train_step=tf.train.AdamOptimizer(learning_rate=self.lr,name="Adam").minimize(self.loss,name="train_step")
			## end of [model definition] ##
			self.saver=tf.train.Saver()
			self.graph=tf.get_default_graph()
		else:
			self.saver=tf.train.import_meta_graph(self.model_dir+"_iter_"+str(self.restoring_epoch)+".ckpt.meta")
			self.saver.restore(self.sess,self.model_dir+"_iter_"+str(self.restoring_epoch)+".ckpt")
			# restore handles by name from graph
			self.graph=tf.get_default_graph()
			self.x=self.graph.get_tensor_by_name("x:0")
			self.y=self.graph.get_tensor_by_name("y:0")
			self.lr=self.graph.get_tensor_by_name("lr:0")
			self.mask=self.graph.get_tensor_by_name("mask:0")
			self.mask_switch=self.graph.get_tensor_by_name("mask_switch:0")
			self.loss=self.graph.get_tensor_by_name("loss:0")
			self.train_step=self.graph.get_operation_by_name("train_step")
			
		self.forward=self.graph.get_tensor_by_name(self.namescope+"/outputs:0")
		
	def train(self,dataset_filename="onehot.npy",mask_hintnum=3,batch_size=10,max_epoch=100,init_lr=1e-2,lr_decay=0.1,lr_decay_iter=50,min_lr=1e-3,train_mask=False,val_iter=10,val_mask=[3,17,20],save_iter=50,test_iter=50):
		if not os.path.exists(dataset_filename):
			print("dataset not found")
			return
		
		loader=data_loader.loader_qa(filedir="./onehot.npy",train_val_test=(1.0,0,0))
		iterator=loader.train_next_batch(batch_size=batch_size)
		
		init=tf.global_variables_initializer()
		self.sess.run(init)
		
		buffer_lr=init_lr
		
		for e in range(self.restoring_epoch,max_epoch):
			b=0
			epoch_loss=0
			if not (e+1)%lr_decay_iter:
				buffer_lr=max(buffer_lr*lr_decay,min_lr)
			for b in range(len(loader.trains)//batch_size):
				batch_x,batch_y=next(iterator)
				batch_mask=np.reshape(utils.mask_generator(masklength=self.sh[0],hintnum=mask_hintnum),[self.sh[0],1])
				_,batch_loss=self.sess.run([self.train_step,self.loss],feed_dict={self.x:batch_x,self.y:batch_y,self.mask:batch_mask,self.lr:buffer_lr,self.mask_switch:train_mask})
				epoch_loss+=batch_loss
			if not (e+1)%val_iter:
				train_whole=loader.get_whole()
				mask_input=[20,17,3]
				mask_onehot=np.zeros([self.sh[0],1])
				mask_onehot[mask_input]=1
				qa_acc=utils.accuracy(label=train_whole,pred=self.sess.run(self.forward,feed_dict={self.x:train_whole,self.mask:mask_onehot,self.mask_switch:False}))
				print("epoch "+str(e+1)+": loss "+str(epoch_loss/b))
				print("number of correct pred for each sample")
				print(qa_acc)
		
			if not (e+1)%save_iter:
				self.saver.save(self.sess,self.model_dir+"_iter_"+str(e+1)+".ckpt")
			if not (e+1)%test_iter:
				max_mean_acc=0
				best_choice=(list(range(mask_hintnum)))
				stack=[-1]
				leftindex=[-1]
				while stack:
					if len(stack)-1==mask_hintnum:
						mask_onehot=np.zeros([self.sh[0],1])
						mask_onehot[stack[1:]]=1
						score=np.mean(utils.accuracy(label=train_whole,pred=self.sess.run(self.forward,feed_dict={self.x:train_whole,self.mask:mask_onehot,self.mask_switch:False})))
						if max_mean_acc<score:
							best_choice=stack[1:]
							max_mean_acc=score
						stack.pop()
						leftindex.pop()
						if leftindex:
							leftindex[-1]+=1
					elif leftindex[-1]<self.sh[0]-1:
						stack.append(leftindex[-1]+1)
						leftindex.append(stack[-1])
					else:
						stack.pop()
						leftindex.pop()
						if leftindex:
							leftindex[-1]+=1						
				print("best choice is:")
				print(best_choice)
				print("best avg score:")
				print(max_mean_acc)	
			
	def predict(self,original):
		# original input is an array with (question id,answer id)
		# both id start with 0
		# inputs shape: [1,27,4]
		inputs,mask=self.convert_inputs(original)
		prob=self.sess.run(self.forward,feed_dict={self.x:inputs,self.mask:mask,self.mask_switch:False})
		return np.argmax(prob,axis=-1)
		
	def convert_inputs(self,original):
		# original input is an array: [(question id,answer id)]
		# feedback from one person
		# both id start with 0
		inputs=np.zeros(shape=[1,self.sh[0],self.sh[1]])
		mask=np.zeros(shape=[self.sh[0],1])
		for qa in original:
			inputs[0,qa[0],qa[1]]=1
			mask[qa[1]]=1
		return inputs,mask
		
mqa=mask_qa(restoring_epoch=0)
mqa.train(mask_hintnum=3,max_epoch=1000,test_iter=100,val_iter=10,val_mask=[3,17,20])
print(mqa.predict([[0,1],[1,2],[2,3]]))

