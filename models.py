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


def inference(method, x, keep_prob, opt):
	""" Define the model up to where it may be used for inference.
	Args:
		method (str): model type
		x: a minibatch of row-vectorised input patches (tensor)
		keep_prob: keep probability for drop-out (tensor)
		n_in: no of input units
		n_out: no of output units
		n_h1: no of units in hidden layer 1
		n_h2: no of units in hidden layer 2
	Returns:
		y_pred: the predicted output patch (tensor)
		L2_sqr: the L2 norm of weights (biases not included)
		L1: the L1 norm of weights
	"""
	method = opt['method']
	n_in = opt['n_in']
	n_out = opt['n_out']
	n_h1 = opt['n_h1']
	n_h2 = opt['n_h2']
	n_h3 = opt['n_h3']

	if method == 'cnn_simple':
		h1_1 = conv3d(x, [3,3,3,6,n_h1], [n_h1], '1_1')
		h1_2 = conv3d(tf.nn.relu(h1_1), [1,1,1,n_h1,n_h2], [n_h2], '1_2')
		y_pred = conv3d(tf.nn.relu(h1_2), [2,2,2,n_h2,6], [6], '2_1')
		L2_sqr = 1.
		L1 = 1.
	else:
		raise ValueError('The chosen method not available ...')
	
	return y_pred, L2_sqr, L1

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

def cost(y, y_pred, L2_sqr, L1, L2_reg, L1_reg):
    """ Define the cost dunction
        Args:
            y(tensor placeholder): a minibatch of row-vectorised ground truth HR patches
            y_pred (tensor function): the corresponding set of predicted patches
            L2_sqr: the L2 norm of weights (biases not included)
            L1: the L1 norm of weights
            L2_reg: the L2-norm regularisation coefficient
            L1_reg: the L1-norm regularisation coefficient
        Returns:
            cost: the loss function to be minimised
    """
    # Predictive metric and regularisers:
    mse = tf.reduce_mean((y - y_pred) ** 2)  # mse in the normalised space
    cost = mse + L2_reg * L2_sqr + L1_reg * L1
    return cost


