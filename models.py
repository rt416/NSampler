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

	if method == 'cnn_simple':
		h1_1 = conv3d(x, [3,3,3,6,n_h1], [n_h1], '1_1')
		h1_2 = conv3d(tf.nn.relu(h1_1), [1,1,1,n_h1,n_h2], [n_h2], '1_2')
		y_pred = conv3d(tf.nn.relu(h1_2), [2,2,2,n_h2,6], [6], '2_1')
	else:
		raise ValueError('The chosen method not available ...')
	
	return y_pred

def get_weights(filter_shape, W_init=None, name=''):
	if W_init == None:
		# He/Xavier
		prod_length = len(filter_shape) - 1
		stddev = np.sqrt(2.0 / np.prod(filter_shape[:prod_length])) 
		W_init = tf.random_normal(filter_shape, stddev=stddev)
	return tf.Variable(W_init, name=name)

def conv3d(x, w_shape, b_shape, name):
	"""Return the 3D convolution"""
	w_name = 'w' + name
	b_name = 'b' + name
	w = get_weights(w_shape, name='l1_1')
	b = tf.get_variable(b_name, dtype=tf.float32, shape=b_shape,
						initializer=tf.constant_initializer(1e-2))
	z = tf.nn.conv3d(x, w, strides=(1,1,1,1,1), padding='VALID')
	return tf.nn.bias_add(z, b)

