"""Builds the specified model:
1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. cost() - Adds to the inference model the layers required to generate the cost function.
3. training() - Adds to the loss model the Ops required to generate and apply gradients.

This file is used by sr_nn.py and not meant to be run.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def inference(method, x, opt):
	""" Define the model up to where it may be used for inference.
	Args:
		method (str): model type
	Returns:
		y_pred: the predicted output patch (tensor)
	"""
	method = opt['method']
	n_h1 = opt['n_h1']
	n_h2 = opt['n_h2']
	n_h3 = opt['n_h3']
	m = opt['m']
	n = opt['n']

	if method == 'cnn_simple':
		h1_1 = conv3d(x, [3,3,3,6,n_h1], [n_h1], '1_1')
		h1_2 = conv3d(tf.nn.relu(h1_1), [1,1,1,n_h1,n_h1], [n_h1], '1_2')
		y_pred = conv3d(tf.nn.relu(h1_2), [3,3,3,n_h1,6*(m**3)], [6*(m**3)], '2_1')
	elif method == 'cnn_residual':
		h1 = tf.nn.relu(conv3d(x, [3,3,3,6,n_h1], [n_h1], '1'))
		# Residual blocks
		h2 = residual_block(h1, n_h1, n_h1, 'res2')
		h3 = residual_block(h2, n_h1, n_h1, 'res3')
		# Output
		h4 = conv3d(h3, [3,3,3,n_h1,n_h2], [n_h2], '4')
		h5 = residual_block(h4, n_h2, n_h2, 'res5')
		h6 = residual_block(h5, n_h2, n_h2, 'res6')
		y_pred = conv3d(h6, [1,1,1,n_h2,6*(m**3)], [6*(m**3)], '7')
	else:
		raise ValueError('The chosen method not available ...')
	
	return y_pred

def residual_block(x, n_in, n_out, name):
	"""A residual block of constant spatial dimensions and 1x1 convs only"""
	b = tf.get_variable(name+'_2b', dtype=tf.float32, shape=[n_out],
						initializer=tf.constant_initializer(1e-2))
	assert n_out >= n_in
	
	h1 = conv3d(x, [1,1,1,n_in,n_out], [n_out], name+'1')
	h2 = conv3d(tf.nn.relu(h1), [1,1,1,n_out,n_out], None, name+'2')
	h3 = tf.pad(x, [[0,0],[0,0],[0,0],[0,0],[0,n_out-n_in]]) + h2
	return tf.nn.relu(tf.nn.bias_add(h3, b))


def get_weights(filter_shape, W_init=None, name=''):
	if W_init == None:
		# He/Xavier
		prod_length = len(filter_shape) - 1
		stddev = np.sqrt(2.0 / np.prod(filter_shape[:prod_length])) 
		W_init = tf.random_normal(filter_shape, stddev=stddev)
	return tf.Variable(W_init, name=name)

def conv3d(x, w_shape, b_shape=None, name=''):
	"""Return the 3D convolution"""
	w_name = name + '_w'
	b_name = name + '_b'
	w = get_weights(w_shape, name=w_name)
	z = tf.nn.conv3d(x, w, strides=(1,1,1,1,1), padding='VALID')
	if b_shape is not None:
		b = tf.get_variable(b_name, dtype=tf.float32, shape=b_shape,
							initializer=tf.constant_initializer(1e-2))
		z = tf.nn.bias_add(z, b)
	return z

def scaled_prediction(method, x, transform, opt):
	x_mean = tf.constant(transform['input_mean'], name='x_mean')
	x_std = tf.constant(transform['input_std'], name='x_std')
	y_mean = tf.constant(transform['output_mean'], name='y_mean')
	y_std = tf.constant(transform['output_std'], name='y_std')

	x_scaled = tf.div(tf.subtract(x - transform['input_mean']), transform['input_std'])
	y = inference(method, x_scaled, opt)
	y_pred = tf.add(tf.mul(transform['output_std'], y), transform['output_mean'], name='y_pred')
	return y_pred