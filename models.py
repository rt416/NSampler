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
	# build the selected model: followed http://cs231n.github.io/neural-networks-2/ for initialisation.
	# todo: remove dropout in the first hidden layer. Done!
	if method == 'cnn_simple':
		weights = {'w1_1' : get_weights([3,3,3,6,n_h1], name='l1_1'),
				   'w1_2' : get_weights([1,1,1,n_h1,n_h2], name='l1_2'),
				   'w2_1' : get_weights([2,2,2,n_h2,6], name='l2_1')
		}
		biases = {'b1_1' : tf.get_variable('b1_1', dtype=tf.float32, shape=[n_h1],
									initializer=tf.constant_initializer(1e-2)),
				'b1_2' : tf.get_variable('b1_2', dtype=tf.float32, shape=[n_h2],
								  initializer=tf.constant_initializer(1e-2)),
				'b2_1' : tf.get_variable('b2_1', dtype=tf.float32, shape=[6],
								  initializer=tf.constant_initializer(1e-2))
				}
		h1_1 = conv3d(x, weights['w1_1'], biases['b1_1'])
		h1_2 = conv3d(tf.nn.relu(h1_1), weights['w1_2'], biases['b1_2'])
		y_pred = conv3d(tf.nn.relu(h1_2), weights['w2_1'], biases['b2_1'])
		L2_sqr = 1.
		L1 = 1.
	else:
		raise ValueError('The chosen method not available ...')
	
	return y_pred, L2_sqr, L1

def get_weights(filter_shape, W_init=None, name=''):
	if W_init == None:
		mean = 0.
		stddev = np.sqrt(2.0 / np.prod(filter_shape[:4]))
		if len(filter_shape) == 1:
			mean = 0.01
			stddev = 0.
		W_init = tf.random_normal(filter_shape, mean=mean, stddev=stddev)
	return tf.Variable(W_init, name=name)


def conv3d(x, w, b):
	z = tf.nn.conv3d(x, w, strides=(1,1,1,1,1), padding='VALID')
	return tf.nn.bias_add(z, b)


def kl_log_uniform_prior(varQ):
    """Compute the gaussian-log uniform KL-div from VDLRT"""
    c1 = 1.16145124
    c2 = -1.50204118
    c3 = 0.58629921
    KL = 0.5*tf.log(varQ) + c1*varQ + c2*tf.pow(varQ,2) + c3*tf.pow(varQ,3)
    return tf.reduce_mean(KL)


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


def training(cost, learning_rate, global_step=None, option='standard'):
    """ Define the optimisation method
        Args:
            cost: loss function to be minimised
            global_step (tensor): the number of optimization steps undergone.
            learning_rate: the learning rate
            option: optimisation method
        Returns:
            train_op: training operation
    """
    if option == 'standard':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif option == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
    else:
        raise ValueError('The chosen method not available ...')

    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op
