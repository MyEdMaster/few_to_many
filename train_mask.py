import tensorflow as tf
import numpy as np
import basic_layers as ml
import data_loader
import os
import utils
## [model definition]
sh=(27,4)
batch_size=10
init_lr=1e-3
lr_decay=0.1
lr_decay_iter=50
min_lr=1e-8
lr=tf.placeholder(dtype=tf.float32,name="lr")
mask_hintnum=3
train_mask=True
train_mask_hint=3

x=tf.placeholder(dtype=tf.float32,shape=[None,*sh],name="x")
y=tf.placeholder(dtype=tf.float32,shape=[None,*sh],name="y")
mask=tf.placeholder(dtype=tf.float32,shape=[sh[0],1],name="mask")
mask_switch=tf.placeholder(dtype=tf.bool,name="mask_switch")

with tf.variable_scope("denoise_AE"):
	#inputs=tf.multiply(mask,x)
	inputs=mask*x
	#mask_trainable=tf.get_variable(name="mask_trainable",shape=[sh[0],1],initializer=tf.constant_initializer(0.5*np.ones([sh[0],1])),trainable=True)
	mask_trainable=tf.get_variable(name="mask_trainable",shape=[sh[0],1],trainable=True)
	#mt=np.zeros([sh[0],1])
	#mt[np.argsort(mask_trainable,axis=0)[-train_mask_hint:]]=1
	#inputs=tf.cond(mask_switch,lambda: mt*inputs,lambda: inputs)
	inputs=tf.cond(mask_switch,lambda: mask_trainable*inputs, lambda: inputs)
	inputs=tf.reshape(inputs,shape=[-1,inputs.get_shape()[1].value*inputs.get_shape()[2].value],name="masked_inputs")
	# activation can be linear
	fc1=ml.fc_layer(inputs,units=10,activation=tf.nn.leaky_relu,name="fc1")
	fc2=ml.fc_layer(fc1,units=20,activation=tf.nn.leaky_relu,name="fc2")
	fc3=ml.fc_layer(fc2,units=sh[0]*sh[1],name="fc3")
	outputs=tf.reshape(fc3,shape=[-1,*y.get_shape()[1:]])
loss=tf.reduce_mean(tf.reduce_mean(tf.pow(tf.add(outputs,-y),2),axis=[1,2]),name="loss")

train_step=tf.train.AdamOptimizer(learning_rate=lr,name="Adam").minimize(loss)
## end of [model definition]


## [run time operations] 
save_dir="./models/"
saver=tf.train.Saver()
save_iter=10

init=tf.global_variables_initializer()
sess=tf.Session()
## end of [run time operations]

print("restoring")
sess.run(init)
restore_epoch=200
if restore_epoch:
	saver.restore(sess,save_dir+"_iter_"+str(restore_epoch)+".ckpt")

loader=data_loader.loader_qa(filedir="./onehot.npy",train_val_test=(1.0,0,0))
iterator=loader.train_next_batch(batch_size=batch_size)

## [start training]
buffer_lr=init_lr
max_epoch=400
for e in range(restore_epoch,max_epoch):
	b=0
	epoch_loss=0
	if not (e+1)%lr_decay_iter:
		buffer_lr=max(buffer_lr*lr_decay,min_lr)
	for b in range(len(loader.trains)//batch_size):
		batch_x,batch_y=next(iterator)
		batch_mask=np.reshape(utils.mask_generator(masklength=sh[0],hintnum=mask_hintnum),[sh[0],1])
		_,batch_loss=sess.run([train_step,loss],feed_dict={x:batch_x,y:batch_y,mask:batch_mask,lr:buffer_lr,mask_switch:train_mask})
		epoch_loss+=batch_loss
	if not (e+1)%10:
		#print(mt)
		mask_trainable_res=sess.run(mask_trainable)
		print(np.argsort(mask_trainable_res))
		print("epoch "+str(e+1)+": "+str(epoch_loss/b))
		print(mask_trainable_res)
	if not (e+1)%save_iter:
		if not os.path.exists(save_dir):
			os.system("mkdir "+save_dir)
		saver.save(sess,save_dir+"_iter_"+str(e+1)+".ckpt")
		
## end of [training]

