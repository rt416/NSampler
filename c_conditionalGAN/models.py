"""Builds the specified discriminator and generator models:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from common.ops import *
from collections import OrderedDict


class discriminator(object):
    def __init__(self,
                 upsampling_rate,
                 layers=4,
                 filters_num=50,
                 bn=False):
        """ Simple discriminator network. Based on DCGAN and pix2pix."""

        self.upsampling_rate = upsampling_rate
        self.layers = layers
        self.filters_num = filters_num
        self.bn = bn

    def forwardpass(self, x, y, phase, reuse=False, input_on=True):
        # Note: input = high-res patch y and low-res patch x
        # with the high-res y input.
        net = []
        net = record_network(net, y)

        # define the network:
        n_f = self.filters_num
        lyr = 0
        with tf.variable_scope("forwardpass") as scope:
            if reuse:
                scope.reuse_variables()
                # tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            if input_on:
                y = crop_and_or_concat_basic(x, y, name='concat1')
                net = record_network(net, y)

            while lyr < self.layers:
                y = conv3d(y, filter_size=3, out_channels=n_f, name='d_conv_' + str(lyr + 1))
                net = record_network(net, y)

                # non-linearity + batch norm:
                # todo: need to optimise the kernel and stride size.
                # y = batchnorm(y, phase, on=self.bn, name='d_BN%d' % len(net))
                y = batch_norm(name='d_BN%d' % len(net))(y, train=phase)
                y = lrelu(y, name='d_activation%d' % len(net))
                lyr += 1
                n_f = int(2 * n_f)

            output = conv3d(y, filter_size=3, out_channels=1, name='d_conv_last')
            net = record_network(net, output)
            print_network(net)
        return tf.nn.sigmoid(output), output



class espcn(object):
    def __init__(self,
                 upsampling_rate,
                 out_channels,
                 layers=2,
                 filters_num=50,
                 bn=False):
        """
        3D Efficient Subpixel-Shifted Convolutional Network
        Need to set opt['is_shuffle']=True

        Args:
            input (tf tensor float 32): input tensor
            upsampling_rate (int): upsampling rate
            out_channels: number of channels in the output image
            layers: number of hidden layers
            filters_num(int): number of filters in the first layer
            bn (boolean): set True fro BatchNorm
        """
        self.upsampling_rate=upsampling_rate
        self.layers=layers
        self.out_channels=out_channels
        self.filters_num=filters_num
        self.bn = bn

    # ------------- STANDARD NETWORK ----------------
    def forwardpass(self, x, phase):
        net = []
        net = record_network(net, x)

        # define the network:
        n_f = self.filters_num
        lyr = 0
        while lyr < self.layers:
            if lyr==1: # second layer with kernel size 1 other layers three
                x = conv3d(x, filter_size=1, out_channels=n_f, name='g_conv_'+str(lyr+1))
            else:
                x = conv3d(x, filter_size=3, out_channels=n_f, name='g_conv_'+str(lyr+1))

            # double the num of features in the second lyr onward
            if lyr == 0: n_f = int(2 * n_f)
            net = record_network(net, x)

            # non-linearity + batch norm:
            x = batchnorm(x, phase, on=self.bn, name='g_BN%d' % len(net))
            x = tf.nn.relu(x, name='g_activation%d' % len(net))
            lyr += 1

        y_pred = conv3d(x,
                        filter_size=3,
                        out_channels=self.out_channels*(self.upsampling_rate)** 3,
                        name='g_conv_last')

        net = record_network(net, y_pred)
        print_network(net)

        # define the loss:
        y = tf.placeholder(tf.float32,
                           shape=get_tensor_shape(y_pred),
                           name='real_y')

        with tf.name_scope('loss'):
            cost = tf.reduce_mean(tf.square(y - y_pred))

        return y_pred, y, cost

    def scaled_prediction(self, x, phase, transform):
        x_mean = tf.constant(np.float32(transform['input_mean']), name='x_mean')
        x_std = tf.constant(np.float32(transform['input_std']), name='x_std')
        y_mean = tf.constant(np.float32(transform['output_mean']),
                             name='y_mean')
        y_std = tf.constant(np.float32(transform['output_std']), name='y_std')
        x_scaled = tf.div(x - x_mean, x_std)

        y = tf.placeholder(tf.float32, name='input_y')  # dummy: you don't need it.
        y_norm, _, _ = self.forwardpass(x_scaled, y, phase)
        y_pred = tf.add(y_std * y_norm, y_mean, name='y_pred')
        return y_pred

    # ------------ BAYESIAN NETWORK -------------------
    # variational dropout only.
    # todo: implement standard dropout i.e. binary and gaussian

    def forwardpass_vardrop(self, x, phase, keep_prob, params, num_data):
        net = []
        net = record_network(net, x)

        # define the network:
        n_f = self.filters_num
        lyr = 0
        kl = 0
        while lyr < self.layers:
            if lyr == 1:  # second layer with kernel size 1 other layers three
                x = conv3d(x, filter_size=1, out_channels=n_f,
                           name='conv_' + str(lyr + 1))
            else:
                x = conv3d(x, filter_size=3, out_channels=n_f,
                           name='conv_' + str(lyr + 1))

            # double the num of features in the second lyr onward
            if lyr == 0: n_f = int(2 * n_f)
            net = record_network(net, x)

            # non-linearity + batch norm + noise injection:
            x = batchnorm(x, phase, on=self.bn, name='BN%d' % len(net))
            x = tf.nn.relu(x, name='activation%d' % len(net))
            x, kl_tmp = normal_mult_noise(x, keep_prob, params,
                                          name='mulnoise%d'% len(net))
            kl += kl_tmp
            lyr += 1

        y_pred = conv3d(x,
                        filter_size=3,
                        out_channels=self.out_channels*(self.upsampling_rate)**3,
                        name='conv_last')

        net = record_network(net, y_pred)
        print_network(net)

        # define the loss:
        y = tf.placeholder(tf.float32,
                           shape=get_tensor_shape(y_pred),
                           name='real_y')

        with tf.name_scope('kl_div'):
            down_sc = 1.0
            kl_div = down_sc*kl
            tf.summary.scalar('kl_div_average', kl_div)

        with tf.name_scope('expected_negloglikelihood'):
            e_negloglike = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_pred), [1, 2, 3, 4]), 0)
            e_negloglike *= num_data
            tf.summary.scalar('e_negloglike', e_negloglike)

        with tf.name_scope('loss'):  # negative evidence lower bound (ELBO)
            cost = tf.add(e_negloglike, -kl_div, name='neg_ELBO')
            tf.summary.scalar('neg_ELBO', cost)

        return y_pred, y, cost

    # ------------ HETEROSCEDASTIC NETWORK ------------
    def forwardpass_hetero(self, x, phase):
        # define the mean network:
        with tf.name_scope('mean_network'):
            h = x + 0.0
            net = []
            net = record_network(net, h)

            # define the network:
            n_f = self.filters_num
            lyr = 0
            while lyr < self.layers:
                if lyr == 1:  # second layer with kernel size 1 other layers three
                    h = conv3d(h, filter_size=1, out_channels=n_f,
                               name='conv_' + str(lyr + 1))
                else:
                    h = conv3d(h, filter_size=3, out_channels=n_f,
                               name='conv_' + str(lyr + 1))

                # double the num of features in the second lyr onward
                if lyr == 0: n_f = int(2 * n_f)
                net = record_network(net, h)

                # non-linearity + batch norm:
                h = batchnorm(h, phase, on=self.bn, name='BN%d' % len(net))
                h = tf.nn.relu(h, name='activation%d' % len(net))
                lyr += 1

            y_pred = conv3d(h,
                            filter_size=3,
                            out_channels=self.out_channels*(self.upsampling_rate)**3,
                            name='conv_last')

            net = record_network(net, y_pred)
            print("Mean network architecture is ...")
            print_network(net)

        # define the covariance network:
        with tf.name_scope('precision_network'):
            h = x + 0.0
            n_f = self.filters_num
            lyr = 0
            while lyr < self.layers:
                if lyr == 1:  # second layer with kernel size 1 other layers three
                    h = conv3d(h, filter_size=1, out_channels=n_f,
                               name='conv_' + str(lyr + 1))
                else:
                    h = conv3d(h, filter_size=3, out_channels=n_f,
                               name='conv_' + str(lyr + 1))

                # double the num of features in the second lyr onward
                if lyr == 0: n_f = int(2 * n_f)

                # non-linearity + batch norm:
                h = batchnorm(h, phase, on=self.bn, name='BN_' + str(lyr + 1))
                h = tf.nn.relu(h, name='activation_' + str(lyr + 1))
                lyr += 1

            h_last = conv3d(h,
                            filter_size=3,
                            out_channels=self.out_channels*(self.upsampling_rate)**3,
                            name='conv_last_prec')
            y_prec = tf.nn.softplus(h_last) + 1e-6  # precision matrix (diagonal)
            y_std = tf.sqrt(1./y_prec, name='y_std')

        # define the loss:
        y = tf.placeholder(tf.float32,
                           shape=get_tensor_shape(y_pred),
                           name='real_y')

        with tf.name_scope('loss'):
            cost = tf.reduce_mean(tf.square(tf.mul(y_prec, (y - y_pred)))) \
                   - tf.reduce_mean(tf.log(y_prec))

        return y_pred, y_std, y, cost

    # ------------ BAYESIAN HETEROSCEDASTIC NETWORK ------------
    def forwardpass_hetero_vardrop(self, x, phase, keep_prob, params,
                                   trade_off, num_data, cov_on=False):
        """ We only perform variational dropout on the parameters of
        the mean network

        params:
            num_data (int_: number of training data points.
            cov (boolean): set True if you want to perform variational dropout
            on the parameters of the covariance network.
        """

        # define the mean network:
        with tf.name_scope('mean_network'):

            h = x + 0.0  # define the input
            net = []
            net = record_network(net, h)
            n_f = self.filters_num
            lyr = 0
            kl=0

            while lyr < self.layers:
                if lyr == 1:  # second layer with kernel size 1 other layers three
                    h = conv3d(h, filter_size=1, out_channels=n_f,
                               name='conv_' + str(lyr + 1))
                else:
                    h = conv3d(h, filter_size=3, out_channels=n_f,
                               name='conv_' + str(lyr + 1))

                # double the num of features in the second lyr onward
                if lyr == 0: n_f = int(2 * n_f)
                net = record_network(net, h)

                # non-linearity + batch norm:
                h = batchnorm(h, phase, on=self.bn, name='BN%d' % len(net))
                h = tf.nn.relu(h, name='activation%d' % len(net))
                h, kl_tmp = normal_mult_noise(tf.nn.relu(h), keep_prob, params,
                                              name='mulnoise%d' % len(net))
                kl += kl_tmp
                lyr += 1

            y_pred = conv3d(h,
                            filter_size=3,
                            out_channels=self.out_channels*(self.upsampling_rate) ** 3,
                            name='conv_last')

            net = record_network(net, y_pred)
            print("Mean network architecture is ...")
            print_network(net)

        # define the covariance network:
        with tf.name_scope('precision_network'):
            h = x + 0.0
            n_f = self.filters_num
            lyr = 0
            kl_prec=0  # kl-div for params of prec network

            while lyr < self.layers:
                if lyr == 1:  # second layer with kernel size 1 other layers three
                    h = conv3d(h, filter_size=1, out_channels=n_f,
                               name='conv_' + str(lyr + 1))
                else:
                    h = conv3d(h, filter_size=3, out_channels=n_f,
                               name='conv_' + str(lyr + 1))

                # double the num of features in the second lyr onward
                if lyr == 0: n_f = int(2 * n_f)

                # non-linearity + batch norm:
                h = batchnorm(h, phase, on=self.bn, name='BN' + str(lyr + 1))
                h = tf.nn.relu(h, name='activation_' + str(lyr + 1))

                # inject multiplicative noise if specified:
                if cov_on:
                    h, kl_tmp = normal_mult_noise(tf.nn.relu(h), keep_prob, params,
                                                  name='mulnoise_'+str(lyr+1))
                    kl_prec += kl_tmp
                lyr += 1

            h_last = conv3d(h,
                            filter_size=3,
                            out_channels=self.out_channels*(self.upsampling_rate) ** 3,
                            name='conv_last_prec')
            y_prec = tf.nn.softplus(h_last) + 1e-6  # precision matrix (diagonal)
            y_std = tf.sqrt(1. / y_prec, name='y_std')

        # define the loss:
        y = tf.placeholder(tf.float32,
                           shape=get_tensor_shape(y_pred),
                           name='real_y')

        with tf.name_scope('kl_div'):
            down_sc = 1.0
            kl_div = down_sc * (kl+kl_prec)
            tf.summary.scalar('kl_div_average', kl_div)

        with tf.name_scope('expected_negloglikelihood'):
            # expected NLL for mean network
            mse_sum = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_pred), [1, 2, 3, 4]), 0)
            mse_sum *= num_data
            tf.summary.scalar('mse_sum', mse_sum)

            # expected NLL for heteroscedastic network
            e_negloglike = tf.reduce_mean(tf.reduce_sum(tf.square(tf.mul(y_prec, (y - y_pred))), [1, 2, 3, 4]), 0) \
                         - tf.reduce_mean(tf.reduce_sum(tf.log(y_prec), [1, 2, 3, 4]), 0)
            e_negloglike *= num_data
            tf.summary.scalar('e_negloglike', e_negloglike)

        with tf.name_scope('loss'):  # negative evidence lower bound (ELBO)
            cost = trade_off*(e_negloglike-kl_div)+(1.-trade_off)*(mse_sum-kl_div)
            tf.summary.scalar('neg_ELBO', cost)

        return y_pred, y_std, y, cost

    # -------- UTILITY ----------------
    def build_network(self, x, phase, keep_prob, params, trade_off,
                      num_data, cov_on, hetero, vardrop):
        if hetero:
            if vardrop:  # hetero + var. drop
                y_pred, y_std, y, cost = self.forwardpass_hetero_vardrop(x, phase, keep_prob, params, trade_off, num_data, cov_on)
            else:        # hetero
                y_pred, y_std, y, cost = self.forwardpass_hetero(x, phase)
        else:
            if vardrop:  # var. drop
                y_pred, y, cost = self.forwardpass_vardrop(x, phase, keep_prob, params, num_data)
            else:        # standard network
                y_pred, y, cost = self.forwardpass(x, phase)
            y_std = 1  # just constant number
        return y_pred, y_std, y, cost

    def scaled_prediction_mc(self, x, phase, keep_prob, params,
                             trade_off, num_data, cov_on, transform,
                             hetero, vardrop):
        x_mean = tf.constant(np.float32(transform['input_mean']), name='x_mean')
        x_std = tf.constant(np.float32(transform['input_std']), name='x_std')
        y_mean = tf.constant(np.float32(transform['output_mean']), name='y_mean')
        y_std = tf.constant(np.float32(transform['output_std']), name='y_std')
        x_scaled = tf.div(x - x_mean, x_std)
        y = tf.placeholder(tf.float32, name='input_y')

        if hetero:
            if vardrop:
                y_norm, y_norm_std, _, _ = self.forwardpass_hetero_vardrop(x_scaled, y, phase, keep_prob, params, trade_off, num_data, cov_on)
            else:
                y_norm, y_norm_std, _, _ = self.forwardpass_hetero(x_scaled, y, phase)

            y_pred = tf.add(y_std * y_norm, y_mean, name='y_pred')
            y_pred_std = tf.mul(y_std, y_norm_std, name='y_pred_std')
        else:
            if vardrop:
                y_norm, _, _ = self.forwardpass_vardrop(x_scaled, phase, keep_prob, params, num_data)
            else:
                y_norm, _, _ = self.forwardpass(x_scaled, phase)

            y_pred = tf.add(y_std * y_norm, y_mean, name='y_pred')
            y_pred_std = 1  # just constant number

        return y_pred, y_pred_std

    def get_output_shape(self):
        return get_tensor_shape(self.y_pred)
