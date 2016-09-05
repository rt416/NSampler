"""Builds the specified models"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def inference(method, x, n_in, n_out, n_h1, n_h2):
    """ Define the model up to where it may be used for inference.
    Args:
        method (str): model type
        x: a minibatch of row-vectorised input patches
        n_in: no of input units
        n_out: no of output units
        n_h1: no of units in hidden layer 1
        n_h2: no of units in hidden layer 2
    Returns:
        y_pred: the predicted output patch (tensor) 
    """
    if method == 'linear':
        # Standard linear regression:
        W1 = tf.Variable(
            tf.random_normal([n_in, n_out], stddev=np.sqrt(2.0 / n_in)),
            name='W1')
        b1 = tf.Variable(tf.constant(1e-2, shape=[n_out]), name='b1')
        y_pred = tf.matmul(x, W1) + b1  # predicted high-res patch in the normalised space
        L2_sqr = tf.reduce_sum(W1 ** 2)
        L1 = tf.reduce_sum(tf.abs(W1))
        
    elif method == 'mlp_h=1':
        # MLP with one hidden layer:
        W1 = tf.Variable(
            tf.random_normal([n_in, n_h1], stddev=np.sqrt(2.0 / n_in)),
            name='W1')
        b1 = tf.Variable(tf.constant(1e-2, shape=[n_h1]), name='b1')
    
        hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    
        W2 = tf.Variable(
            tf.random_normal([n_h1, n_out], stddev=np.sqrt(2.0 / n_h1)),
            name='W2')
        b2 = tf.Variable(tf.constant(1e-2, shape=[n_out]), name='b2')
    
        y_pred = tf.matmul(hidden1, W2) + b2
        L2_sqr = tf.reduce_sum(W1 ** 2) + tf.reduce_sum(W2 ** 2) 
        L1 = tf.reduce_sum(tf.abs(W1)) + tf.reduce_sum(tf.abs(W2)) 

    elif method == 'mlp_h=2':
        # MLP with two hidden layers:
        W1 = tf.Variable(
            tf.random_normal([n_in, n_h1], stddev=np.sqrt(2.0 / n_in)),
            name='W1')
        b1 = tf.Variable(tf.constant(1e-2, shape=[n_h1]), name='b1')
    
        hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    
        W2 = tf.Variable(
            tf.random_normal([n_h1, n_h2], stddev=np.sqrt(2.0 / n_h1)),
            name='W2')
        b2 = tf.Variable(tf.constant(1e-2, shape=[n_h2]), name='b2')
    
        hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + b2)
    
        W3 = tf.Variable(
            tf.random_normal([n_h2, n_out], stddev=np.sqrt(2.0 / n_h2)),
            name='W3')
        b3 = tf.Variable(tf.constant(1e-2, shape=[n_out]), name='b3')
    
        y_pred = tf.matmul(hidden2, W3) + b3
        L2_sqr = tf.reduce_sum(W1 ** 2) + tf.reduce_sum(W2 ** 2) + tf.reduce_sum(W3 ** 2)
        L1 = tf.reduce_sum(tf.abs(W1)) + tf.reduce_sum(tf.abs(W2)) + tf.reduce_sum(tf.abs(W3))

    else:
        raise ValueError('The chosen method not available ...')

    return y_pred, L2_sqr, L1


def loss(y, y_pred, L2_sqr, L1, L2_reg, L1_reg):
    # Predictive metric and regularisers:
    mse = tf.reduce_mean((y - y_pred) ** 2)  # mse in the normalised space
    cost = mse + L2_reg * L2_sqr + L1_reg * L1
    return cost
