import tensorflow as tf
import numpy as np

def fc_layer(inputs,units=100,W_init=None,stddev=1e-3,use_bias=True,bias_init=None,activation=None,name="fc0"):
	# save the first rank and flatten all the rest
	# be sure that inputs have at least rank of 2
	rest=1
	for m in inputs.get_shape()[1:]:
		rest=rest*m.value
	flat=tf.reshape(inputs,shape=[-1,rest])
	with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
		if W_init:
			W=tf.get_variable(initializer=W_init,dtype=tf.float32,name="W")
		else:
			W=tf.get_variable(initializer=tf.random.truncated_normal(shape=[rest,units],stddev=stddev),dtype=tf.float32,name="W")
		if use_bias:
			if bias_init:
				b=tf.get_variable(initializer=bias_init,dtype=tf.float32,name="b")
			else:
				b=tf.get_variable(initializer=tf.random.truncated_normal(shape=[units],stddev=stddev),dtype=tf.float32,name="b")
		
		res=tf.matmul(flat,W)
		res=tf.nn.bias_add(res,b)
		if activation:
			res=activation(res)
		return res

def conv2d_layer(inputs,kernel_size=3,filters=64,strides=[2,2],kernel_init=None,stddev=1e-3,use_bias=True,bias_init=None,activation=None,name="conv0"):
	# kernel_size and filters are both integers
	# inputs is a tensor of [batch,height,width,channels]
	with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
		# auto_reuse also reuse scope
		if not kernel_init:
			kernels=tf.get_variable(initializer=tf.random.truncated_normal(shape=[kernel_size,kernel_size,inputs.get_shape()[-1].value,filters],stddev=stddev),dtype=tf.float32,name="kernels")
		else:
			kernels=tf.get_variable(initializer=kernel_init,dtype=tf.float32,name="kernels")
		if use_bias:
			if not bias_init:
				bias=tf.get_variable(initializer=tf.random.truncated_normal(shape=[filters],stddev=stddev),dtype=tf.float32,name="bias")
			else:
				bias=tf.get_variable(initializer=bias_init,dtype=tf.float32,name="bias")
		res=tf.nn.convolution(inputs,kernels,padding="SAME",strides=strides,name=name)
		res=tf.nn.bias_add(res,bias,data_format="NHWC")
		if activation:
			res=activation(res)
		return res
