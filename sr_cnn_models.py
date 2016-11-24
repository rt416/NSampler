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



def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1])

# todo: need to define the reshuffling:
def cnn_vanilla(x):
    """
    The vanilla CNN with 2 convolutions + 1 deconvolution
    The input image x needs to be reshaped to the shape [batch, in_height, in_width, in_depth, in_channels]
    The kernel of shape [filter_height, filter_width, filter_depth, in_channels, out_channels]
    """
    us = 2  # upsampling rate
    dt_no = 6  # number of dti components
    f_1, f_2, f_3 = 3, 1, 3
    n_1, n_2, n_3 = 64, 34, dt_no * (us**3)

    W_conv1 = weight_variable([f_1, f_1, f_1, dt_no, n_1], name='W_conv1')
    b_conv1 = bias_variable([n_1], name='b_conv1')
    h_conv1 = tf.nn.relu(conv3d(x, W_conv1) + b_conv1)

    W_conv2 = weight_variable([f_2, f_2, f_2, n_1, n_2], name='W_conv2')
    b_conv2 = bias_variable([n_2], name='b_conv2')
    h_conv2 = tf.nn.relu(conv3d(x, W_conv2) + b_conv2)

    W_conv3 = weight_variable([f_2, f_2, f_3, n_2, n_3], name='W_conv3')
    b_conv3 = bias_variable([n_2], name='b_conv3')
    y_pred = conv3d(x, W_conv3) + b_conv3

    return y_pred








def inference(method, x, keep_prob, n_in, n_out, n_h1=None, n_h2=None, n_h3=None):
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
    # build the selected model: followed http://cs231n.github.io/neural-networks-2/ for initialisation.
    # todo: remove dropout in the first hidden layer. Done!
    if method == 'linear':
        # Standard linear regression:
        W1 = tf.Variable(
            tf.random_normal([n_in, n_out], stddev=np.sqrt(2.0 / n_in)),
            name='W1')
        b1 = tf.Variable(tf.constant(1e-2, shape=[n_out]), name='b1')
        hidden1 = tf.matmul(x, W1) + b1
        y_pred = tf.nn.dropout(hidden1, keep_prob)  # predicted high-res patch in the normalised space

        L2_sqr = tf.reduce_sum(W1 ** 2)
        L1 = tf.reduce_sum(tf.abs(W1))

    elif method == 'mlp_h=1':
        # MLP with one hidden layer:
        W1 = tf.Variable(
            tf.random_normal([n_in, n_h1], stddev=np.sqrt(2.0 / n_in)),
            name='W1')
        b1 = tf.Variable(tf.constant(1e-2, shape=[n_h1]), name='b1')
        hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
        hidden1_drop = tf.nn.dropout(hidden1, keep_prob)

        W2 = tf.Variable(
            tf.random_normal([n_h1, n_out], stddev=np.sqrt(2.0 / n_h1)),
            name='W2')
        b2 = tf.Variable(tf.constant(1e-2, shape=[n_out]), name='b2')
        y_pred = tf.matmul(hidden1_drop, W2) + b2

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
        hidden2_drop = tf.nn.dropout(hidden2, keep_prob)

        W3 = tf.Variable(
            tf.random_normal([n_h2, n_out], stddev=np.sqrt(2.0 / n_h2)),
            name='W3')
        b3 = tf.Variable(tf.constant(1e-2, shape=[n_out]), name='b3')
        y_pred = tf.matmul(hidden2_drop, W3) + b3

        L2_sqr = tf.reduce_sum(W1 ** 2) + tf.reduce_sum(W2 ** 2) + tf.reduce_sum(W3 ** 2)
        L1 = tf.reduce_sum(tf.abs(W1)) + tf.reduce_sum(tf.abs(W2)) + tf.reduce_sum(tf.abs(W3))
    elif method == 'mlp_h=3':
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
        hidden2_drop = tf.nn.dropout(hidden2, keep_prob)

        W3 = tf.Variable(
            tf.random_normal([n_h2, n_h3], stddev=np.sqrt(2.0 / n_h2)),
            name='W3')
        b3 = tf.Variable(tf.constant(1e-2, shape=[n_h3]), name='b3')
        hidden3 = tf.nn.relu(tf.matmul(hidden2_drop, W3) + b3)
        hidden3_drop = tf.nn.dropout(hidden3, keep_prob)

        W4 = tf.Variable(
            tf.random_normal([n_h3, n_out], stddev=np.sqrt(2.0 / n_h3)),
            name='W4')
        b4 = tf.Variable(tf.constant(1e-2, shape=[n_out]), name='b4')
        y_pred = tf.matmul(hidden3_drop, W4) + b4

        L2_sqr = tf.reduce_sum(W1 ** 2) + tf.reduce_sum(W2 ** 2) + \
                 tf.reduce_sum(W3 ** 2) + tf.reduce_sum(W4 ** 2)
        L1 = tf.reduce_sum(tf.abs(W1)) + tf.reduce_sum(tf.abs(W2)) + \
             tf.reduce_sum(tf.abs(W3)) + tf.reduce_sum(tf.abs(W4))

    elif method == 'mlp_h=1_kingma':
        # MLP with one hidden layer:
        rho1 = get_weights([n_in, ], name='rho1')
        sigma1 = tf.nn.softplus(rho1)
        x_drop = x * (1. + sigma1 * tf.random_normal(tf.shape(x)))
        aff1, W1, b1 = affine_layer(x_drop, n_in, n_h1, name='aff1')
        h1 = tf.nn.relu(aff1)

        rho2 = get_weights([n_h1, ], name='rho2')
        sigma2 = tf.nn.softplus(rho2)
        h1_drop = h1 * (1. + sigma2 * tf.random_normal(tf.shape(h1)))
        y_pred, W2, b2 = affine_layer(h1_drop, n_h1, n_out, name='aff2')

        L2_sqr = tf.reduce_sum(W1 ** 2) + tf.reduce_sum(W2 ** 2)
        L1 = tf.reduce_sum(tf.abs(W1)) + tf.reduce_sum(tf.abs(W2))
        KL1 = kl_log_uniform_prior(tf.pow(sigma1, 2.))
        KL2 = kl_log_uniform_prior(tf.pow(sigma2, 2.))
        KL_list = [KL1, KL2]

    else:
        raise ValueError('The chosen method not available ...')

    return y_pred, L2_sqr, L1

def get_weights(filter_shape, W_init=None, name=''):
    if W_init == None:
        mean = 0.
        stddev = np.sqrt(2.0 / filter_shape[0])
        if len(filter_shape) == 1:
            mean = 0.01
            stddev = 0.
        W_init = tf.random_normal(filter_shape, mean=mean, stddev=stddev)
    return tf.Variable(W_init, name=name)


def affine_layer(x, n_in, n_out, name='aff'):
    W = get_weights([n_in, n_out], name=name+'_W')
    b = get_weights([n_out], name=name+'_b')
    return tf.matmul(x, W) + b, W, b


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
