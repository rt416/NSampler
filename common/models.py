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

from common.ops import *
from collections import OrderedDict


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
    def forwardpass(self, x, y, phase):
        net = []
        net = record_network(net, x)

        # define the network:
        n_f = self.filters_num
        lyr = 0
        while lyr < self.layers:
            if lyr==1: # second layer with kernel size 1 other layers three
                x = conv3d(x, filter_size=1, out_channels=n_f, name='conv_'+str(lyr+1))
            else:
                x = conv3d(x, filter_size=3, out_channels=n_f, name='conv_'+str(lyr+1))

            # double the num of features in the second lyr onward
            if lyr == 0: n_f = int(2 * n_f)
            net = record_network(net, x)

            # non-linearity + batch norm:
            x = batchnorm(x, phase, on=self.bn, name='BN%d' % len(net))
            x = tf.nn.relu(x, name='activation%d' % len(net))
            lyr += 1

        y_pred = conv3d(x,
                        filter_size=3,
                        out_channels=self.out_channels*(self.upsampling_rate)** 3,
                        name='conv_last')

        net = record_network(net, y_pred)
        print_network(net)

        # define the loss:
        with tf.name_scope('loss'):
            cost = tf.reduce_mean(tf.square(y - y_pred))

        return y_pred, cost

    def scaled_prediction(self, x, phase, transform):
        x_mean = tf.constant(np.float32(transform['input_mean']), name='x_mean')
        x_std = tf.constant(np.float32(transform['input_std']), name='x_std')
        y_mean = tf.constant(np.float32(transform['output_mean']),
                             name='y_mean')
        y_std = tf.constant(np.float32(transform['output_std']), name='y_std')
        x_scaled = tf.div(x - x_mean, x_std)

        y = tf.placeholder(tf.float32, name='input_y')  # dummy: you don't need it.
        y_norm, _ = self.forwardpass(x_scaled, y, phase)
        y_pred = tf.add(y_std * y_norm, y_mean, name='y_pred')
        return y_pred

    # ------------ BAYESIAN NETWORK -------------------
    # variational dropout only.
    # todo: implement standard dropout i.e. binary and gaussian

    def forwardpass_vardrop(self, x, y, phase, keep_prob, params, num_data):
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

        return y_pred, cost

    # ------------ HETEROSCEDASTIC NETWORK ------------
    def forwardpass_hetero(self, x, y, phase):
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
        with tf.name_scope('loss'):
            # (previous ver, 12 Aug):
            # cost = tf.reduce_mean(tf.square(tf.mul(y_prec, (y - y_pred)))) \
            #        - tf.reduce_mean(tf.log(y_prec))
            cost = tf.reduce_mean(tf.mul(y_prec, tf.square(y - y_pred))) \
                   - tf.reduce_mean(tf.log(y_prec))

        return y_pred, y_std, cost

    # ------------ BAYESIAN HETEROSCEDASTIC NETWORK ------------
    def forwardpass_hetero_vardrop(self, x, y, phase, keep_prob, params,
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
            # (previous ver, 12 Aug):
            # e_negloglike = tf.reduce_mean(tf.reduce_sum(tf.square(tf.mul(y_prec, (y - y_pred))), [1, 2, 3, 4]), 0) \
            #              - tf.reduce_mean(tf.reduce_sum(tf.log(y_prec), [1, 2, 3, 4]), 0)

            e_negloglike = tf.reduce_mean(tf.reduce_sum(tf.mul(y_prec, tf.square(y - y_pred)), [1, 2, 3, 4]), 0) \
                           - tf.reduce_mean(tf.reduce_sum(tf.log(y_prec), [1, 2, 3, 4]), 0)
            e_negloglike *= num_data
            tf.summary.scalar('e_negloglike', e_negloglike)

        with tf.name_scope('loss'):  # negative evidence lower bound (ELBO)
            cost = trade_off*(e_negloglike-kl_div)+(1.-trade_off)*(mse_sum-kl_div)
            tf.summary.scalar('neg_ELBO', cost)

        return y_pred, y_std, cost

    # -------- UTILITY ----------------
    def build_network(self, x, y, phase, keep_prob, params, trade_off,
                      num_data, cov_on, hetero, vardrop):
        if hetero:
            if vardrop:  # hetero + var. drop
                y_pred, y_std, cost = self.forwardpass_hetero_vardrop(x, y, phase, keep_prob, params, trade_off, num_data, cov_on)
            else:        # hetero
                y_pred, y_std, cost = self.forwardpass_hetero(x, y, phase)
        else:
            if vardrop:  # var. drop
                y_pred, cost = self.forwardpass_vardrop(x, y, phase, keep_prob, params, num_data)
            else:        # standard network
                y_pred, cost = self.forwardpass(x, y, phase)
            y_std = 1  # just constant number
        return y_pred, y_std, cost

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
                y_norm, y_norm_std, _ = self.forwardpass_hetero_vardrop(x_scaled, y, phase, keep_prob, params, trade_off, num_data, cov_on)
            else:
                y_norm, y_norm_std, _ = self.forwardpass_hetero(x_scaled, y, phase)

            y_pred = tf.add(y_std * y_norm, y_mean, name='y_pred')
            y_pred_std = tf.mul(y_std, y_norm_std, name='y_pred_std')
        else:
            if vardrop:
                y_norm, _ = self.forwardpass_vardrop(x_scaled, y, phase, keep_prob, params, num_data)
            else:
                y_norm, _ = self.forwardpass(x_scaled, y, phase)

            y_pred = tf.add(y_std * y_norm, y_mean, name='y_pred')
            y_pred_std = 1  # just constant number

        return y_pred, y_pred_std

    def get_output_shape(self):
        return get_tensor_shape(self.y_pred)


class espcn_LRT(object):
    def __init__(self,
                 upsampling_rate,
                 out_channels,
                 layers=2,
                 filters_num=50,
                 bn=False):
        """
        3D Efficient Subpixel-Shifted Convolutional Network
        Need to set opt['is_shuffle']=True

        Variational dropout is implemented with local reparametrisation trick.

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
    def forwardpass(self, x, y, phase):
        net = []
        net = record_network(net, x)

        # define the network:
        n_f = self.filters_num
        lyr = 0
        while lyr < self.layers:
            if lyr==1: # second layer with kernel size 1 other layers three
                x = conv3d(x, filter_size=1, out_channels=n_f, name='conv_'+str(lyr+1))
            else:
                x = conv3d(x, filter_size=3, out_channels=n_f, name='conv_'+str(lyr+1))

            # double the num of features in the second lyr onward
            if lyr == 0: n_f = int(2 * n_f)
            net = record_network(net, x)

            # non-linearity + batch norm:
            x = batchnorm(x, phase, on=self.bn, name='BN%d' % len(net))
            x = tf.nn.relu(x, name='activation%d' % len(net))
            lyr += 1

        y_pred = conv3d(x,filter_size=3, out_channels=self.out_channels*(self.upsampling_rate)** 3, name='conv_last')
        net = record_network(net, y_pred)
        print_network(net)

        # define the loss:
        with tf.name_scope('loss'):
            cost = tf.reduce_mean(tf.square(y - y_pred))

        return y_pred, cost

    def scaled_prediction(self, x, phase, transform):
        x_mean = tf.constant(np.float32(transform['input_mean']), name='x_mean')
        x_std = tf.constant(np.float32(transform['input_std']), name='x_std')
        y_mean = tf.constant(np.float32(transform['output_mean']),
                             name='y_mean')
        y_std = tf.constant(np.float32(transform['output_std']), name='y_std')
        x_scaled = tf.div(x - x_mean, x_std)

        y = tf.placeholder(tf.float32, name='input_y')  # dummy: you don't need it.
        y_norm, _ = self.forwardpass(x_scaled, y, phase)
        y_pred = tf.add(y_std * y_norm, y_mean, name='y_pred')
        return y_pred

    # ------------ BAYESIAN NETWORK -------------------
    # variational dropout only.

    def forwardpass_vardrop(self, x, y, phase, keep_prob, params, num_data):
        net = []
        net = record_network(net, x)

        # define the network:
        n_f = self.filters_num
        lyr = 0
        kl = 0
        h = x + 0.0

        while lyr < self.layers:
            if lyr == 1:  # second layer with kernel size 1 other layers three
                h, kl_tmp = conv3d_vardrop_LRT(h, n_f, params, keep_prob, filter_size=1, determinisitc=False, name='conv_' + str(lyr + 1))
            else:
                h, kl_tmp = conv3d_vardrop_LRT(h, n_f, params, keep_prob, filter_size=3, determinisitc=False, name='conv_' + str(lyr + 1))

            # double the num of features in the second lyr onward
            if lyr == 0: n_f = int(2 * n_f)
            net = record_network(net, h)

            # non-linearity + batch norm + noise injection:
            h = batchnorm(h, phase, on=self.bn, name='BN%d' % len(net))
            h = tf.nn.relu(h, name='activation%d' % len(net))
            kl += kl_tmp
            lyr += 1

        n_f = self.out_channels*(self.upsampling_rate)**3
        y_pred, kl_tmp = conv3d_vardrop_LRT(h, n_f, params, keep_prob, filter_size=3, determinisitc=False, name='conv_last')
        kl += kl_tmp
        net = record_network(net, y_pred)
        print_network(net)

        # y_pred = conv3d(h,
        #                 filter_size=3,
        #                 out_channels=self.out_channels * (self.upsampling_rate) ** 3,
        #                 name='conv_last')



        # y_pred, kl_tmp = conv3d_vardrop_LRT(x,
        #                                     out_channels=self.out_channels*(self.upsampling_rate)**3,
        #                                     params=params, keep_prob=keep_prob, filter_size=3, name='conv_last')


        # define the loss:
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

        return y_pred, cost

    # ------------ HETEROSCEDASTIC NETWORK ------------
    def forwardpass_hetero(self, x, y, phase):
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
        with tf.name_scope('loss'):
            # (previous ver, 12 Aug):
            # cost = tf.reduce_mean(tf.square(tf.mul(y_prec, (y - y_pred)))) \
            #        - tf.reduce_mean(tf.log(y_prec))
            cost = tf.reduce_mean(tf.mul(y_prec, tf.square(y - y_pred))) \
                   - tf.reduce_mean(tf.log(y_prec))

        return y_pred, y_std, cost

    # ------------ BAYESIAN HETEROSCEDASTIC NETWORK ------------
    def forwardpass_hetero_vardrop(self, x, y, phase, keep_prob, params,
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
            # (previous ver, 12 Aug):
            # e_negloglike = tf.reduce_mean(tf.reduce_sum(tf.square(tf.mul(y_prec, (y - y_pred))), [1, 2, 3, 4]), 0) \
            #              - tf.reduce_mean(tf.reduce_sum(tf.log(y_prec), [1, 2, 3, 4]), 0)

            e_negloglike = tf.reduce_mean(tf.reduce_sum(tf.mul(y_prec, tf.square(y - y_pred)), [1, 2, 3, 4]), 0) \
                           - tf.reduce_mean(tf.reduce_sum(tf.log(y_prec), [1, 2, 3, 4]), 0)
            e_negloglike *= num_data
            tf.summary.scalar('e_negloglike', e_negloglike)

        with tf.name_scope('loss'):  # negative evidence lower bound (ELBO)
            cost = trade_off*(e_negloglike-kl_div)+(1.-trade_off)*(mse_sum-kl_div)
            tf.summary.scalar('neg_ELBO', cost)

        return y_pred, y_std, cost

    # -------- UTILITY ----------------
    def build_network(self, x, y, phase, keep_prob, params, trade_off,
                      num_data, cov_on, hetero, vardrop):
        if hetero:
            if vardrop:  # hetero + var. drop
                y_pred, y_std, cost = self.forwardpass_hetero_vardrop(x, y, phase, keep_prob, params, trade_off, num_data, cov_on)
            else:        # hetero
                y_pred, y_std, cost = self.forwardpass_hetero(x, y, phase)
        else:
            if vardrop:  # var. drop
                y_pred, cost = self.forwardpass_vardrop(x, y, phase, keep_prob, params, num_data)
            else:        # standard network
                y_pred, cost = self.forwardpass(x, y, phase)
            y_std = 1  # just constant number
        return y_pred, y_std, cost

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
                y_norm, y_norm_std, _ = self.forwardpass_hetero_vardrop(x_scaled, y, phase, keep_prob, params, trade_off, num_data, cov_on)
            else:
                y_norm, y_norm_std, _ = self.forwardpass_hetero(x_scaled, y, phase)

            y_pred = tf.add(y_std * y_norm, y_mean, name='y_pred')
            y_pred_std = tf.mul(y_std, y_norm_std, name='y_pred_std')
        else:
            if vardrop:
                y_norm, _ = self.forwardpass_vardrop(x_scaled, y, phase, keep_prob, params, num_data)
            else:
                y_norm, _ = self.forwardpass(x_scaled, y, phase)

            y_pred = tf.add(y_std * y_norm, y_mean, name='y_pred')
            y_pred_std = 1  # just constant number

        return y_pred, y_pred_std

    def get_output_shape(self):
        return get_tensor_shape(self.y_pred)


class dcespcn(object):
    def __init__(self,
                 upsampling_rate,
                 out_channels,
                 layers=2,
                 filters_num=50,
                 bn=False):
        """
        Densely connected ESPCN:
        Input into L th convolution is the concatenation of all the feature maps
        in the preceding L-1 layers.
        We perform all convolutions without padding as we find them detrimental
        to performance. This means that the output feature map size reduces after
        each convolution. We thus crop the feature maps of previous layers accordingly
        before concatenation.
        Args:
            input (tf tensor float 32): input tensor
            upsampling_rate (int): upsampling rate
            out_channels: number of channels in the output image
            layers: number of hidden layers
            filters_num(int): number of filters in the first layer
        """
        self.upsampling_rate=upsampling_rate
        self.layers=layers
        self.out_channels=out_channels
        self.filters_num=filters_num
        self.bn = bn

    # ------------- STANDARD NETWORK ----------------
    def forwardpass(self, x, y, phase):
        net = []
        net = record_network(net, x)

        # define the network:
        n_f = self.filters_num
        x = conv3d(x, filter_size=3, out_channels=n_f,
                   name='conv_' + str(1))
        net = record_network(net, x)
        lyr = 1

        while lyr < self.layers:
            if lyr == 1:  # second layer with kernel size 1 other layers three
                x = conv_dc_3d(x, phase, bn_on=self.bn,
                               out_channels=n_f, filter_size=1,
                               name='conv_dc_' + str(lyr + 1))
            else:
                x = conv_dc_3d(x, phase, bn_on=self.bn,
                               out_channels=n_f, filter_size=3,
                               name='conv_dc_' + str(lyr + 1))

            net = record_network(net, x)
            lyr += 1

        y_pred = conv3d(tf.nn.relu(x),
                        filter_size=3,
                        out_channels=self.out_channels*(self.upsampling_rate)** 3,
                        name='conv_last')

        net = record_network(net, y_pred)
        print_network(net)

        # define the loss:
        with tf.name_scope('loss'):
            cost = tf.reduce_mean(tf.square(y - y_pred))

        return y_pred, cost

    def scaled_prediction(self, x, phase, transform):
        x_mean = tf.constant(np.float32(transform['input_mean']), name='x_mean')
        x_std = tf.constant(np.float32(transform['input_std']), name='x_std')
        y_mean = tf.constant(np.float32(transform['output_mean']), name='y_mean')
        y_std = tf.constant(np.float32(transform['output_std']), name='y_std')
        x_scaled = tf.div(tf.sub(x, x_mean), x_std)

        y = tf.placeholder(tf.float32, name='input_y')
        y_norm, _ = self.forwardpass(x_scaled, y, phase)
        y_pred = tf.add(tf.mul(y_std, y_norm), y_mean, name='y_pred')
        return y_pred

    # ------------ BAYESIAN NETWORK -------------------
    # variational dropout only.
    # todo: implement standard dropout i.e. binary and gaussian

    def forwardpass_vardrop(self, x, y, phase, keep_prob, params, num_data):
        net = []
        net = record_network(net, x)

        # define the network:
        n_f = self.filters_num
        x = conv3d(x, filter_size=3, out_channels=n_f,
                   name='conv_' + str(1))
        net = record_network(net, x)
        lyr = 1
        kl = 0
        while lyr < self.layers:
            if lyr == 1:  # second layer with kernel size 1 other layers three
                x = conv_dc_3d(x, phase, bn_on=self.bn,
                               out_channels=n_f, filter_size=1,
                               name='conv_dc_' + str(lyr + 1))
            else:
                x = conv_dc_3d(x, phase, bn_on=self.bn,
                               out_channels=n_f, filter_size=3,
                               name='conv_dc_' + str(lyr + 1))

            net = record_network(net, x)
            x, kl_tmp = normal_mult_noise(tf.nn.relu(x), keep_prob, params,
                                          name='mulnoise%d' % len(net))
            kl += kl_tmp
            lyr += 1

        y_pred = conv3d(tf.nn.relu(x),
                        filter_size=3,
                        out_channels=self.out_channels*(self.upsampling_rate)**3,
                        name='conv_last')

        net = record_network(net, y_pred)
        print_network(net)

        # define the loss:
        with tf.name_scope('kl_div'):
            down_sc = 1.0
            kl_div = down_sc * kl
            tf.summary.scalar('kl_div_average', kl_div)

        with tf.name_scope('expected_negloglikelihood'):
            e_negloglike = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_pred), [1, 2, 3, 4]), 0)
            e_negloglike *= num_data
            tf.summary.scalar('e_negloglike', e_negloglike)

        with tf.name_scope('loss'):  # negative evidence lower bound (ELBO)
            cost = tf.add(e_negloglike, -kl_div, name='neg_ELBO')
            tf.summary.scalar('neg_ELBO', cost)

        return y_pred, cost

    # ------------ HETEROSCEDASTIC NETWORK ------------
    def forwardpass_hetero(self, x, y, phase):
        # define the mean network:
        with tf.name_scope('mean_network'):
            h = x + 0.0
            net = []
            net = record_network(net, h)

            # define the network:
            n_f = self.filters_num
            h = conv3d(h, filter_size=3, out_channels=n_f,
                       name='conv_' + str(1))
            net = record_network(net, h)
            lyr = 1
            while lyr < self.layers:
                if lyr == 1:  # second layer with kernel size 1 other layers three
                    h = conv_dc_3d(h, phase, bn_on=self.bn,
                                   out_channels=n_f, filter_size=1,
                                   name='conv_dc_' + str(lyr + 1))
                else:
                    h = conv_dc_3d(h, phase, bn_on=self.bn,
                                   out_channels=n_f, filter_size=3,
                                   name='conv_dc_' + str(lyr + 1))

                net = record_network(net, h)
                lyr += 1

            y_pred = conv3d(tf.nn.relu(h),
                            filter_size=3,
                            out_channels=self.out_channels * (
                                                             self.upsampling_rate) ** 3,
                            name='conv_last')

            net = record_network(net, y_pred)
            print_network(net)

        # define the covariance network:
        with tf.name_scope('precision_network'):
            # define the network:
            h = x + 0.0
            n_f = self.filters_num
            h = conv3d(h, filter_size=3, out_channels=n_f,
                       name='conv_' + str(1))
            lyr = 1
            while lyr < self.layers:
                if lyr == 1:  # second layer with kernel size 1 other layers three
                    h = conv_dc_3d(h, phase, bn_on=self.bn,
                                   out_channels=n_f, filter_size=1,
                                   name='conv_dc_' + str(lyr + 1))
                else:
                    h = conv_dc_3d(h, phase, bn_on=self.bn,
                                   out_channels=n_f, filter_size=3,
                                   name='conv_dc_' + str(lyr + 1))

                net = record_network(net, h)
                lyr += 1

            h_last = conv3d(tf.nn.relu(h),
                            filter_size=3,
                            out_channels=self.out_channels * (
                                                             self.upsampling_rate) ** 3,
                            name='conv_last_prec')
            y_prec = tf.nn.softplus(
                h_last) + 1e-6  # precision matrix (diagonal)
            y_std = tf.sqrt(1. / y_prec, name='y_std')

        # define the loss:
        with tf.name_scope('loss'):
            # cost = tf.reduce_mean(tf.square(tf.mul(y_prec, (y - y_pred)))) \
            #        - tf.reduce_mean(tf.log(y_prec))
            cost = tf.reduce_mean(tf.mul(y_prec, tf.square(y - y_pred))) \
                   - tf.reduce_mean(tf.log(y_prec))

        return y_pred, y_std, cost

    # -------- BAYESIAN HETEROSCEDASTIC NETWORK -----------
    def forwardpass_hetero_vardrop(self, x, y, phase, keep_prob, params,
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

            # define the network
            n_f = self.filters_num
            h = conv3d(h, filter_size=3, out_channels=n_f,
                       name='conv_' + str(1))
            net = record_network(net, h)

            lyr = 1
            kl = 0

            while lyr < self.layers:
                if lyr == 1:
                    h = conv_dc_3d(h, phase, bn_on=self.bn,
                                   out_channels=n_f, filter_size=1,
                                   name='conv_dc_' + str(lyr + 1))
                else:
                    h = conv_dc_3d(h, phase, bn_on=self.bn,
                                   out_channels=n_f, filter_size=3,
                                   name='conv_dc_' + str(lyr + 1))

                net = record_network(net, h)
                h, kl_tmp = normal_mult_noise(tf.nn.relu(h), keep_prob, params,
                                              name='mulnoise%d' % len(net))
                kl += kl_tmp
                lyr += 1

            y_pred = conv3d(tf.nn.relu(h),
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
            h = conv3d(h, filter_size=3, out_channels=n_f,
                       name='conv_' + str(1))
            lyr = 1
            kl_prec = 0  # kl-div for params of prec network

            while lyr < self.layers:
                if lyr == 1:
                    h = conv_dc_3d(h, phase, bn_on=self.bn,
                                   out_channels=n_f, filter_size=1,
                                   name='conv_dc_' + str(lyr + 1))
                else:
                    h = conv_dc_3d(h, phase, bn_on=self.bn,
                                   out_channels=n_f, filter_size=3,
                                   name='conv_dc_' + str(lyr + 1))

                # inject multiplicative noise if specified:
                if cov_on:
                    h, kl_tmp = normal_mult_noise(tf.nn.relu(h), keep_prob, params,
                                                  name='mulnoise_' + str(lyr + 1))
                    kl_prec += kl_tmp
                lyr += 1

            h_last = conv3d(tf.nn.relu(h),
                            filter_size=3,
                            out_channels=self.out_channels*(self.upsampling_rate)**3,
                            name='conv_last_prec')
            y_prec = tf.nn.softplus(h_last) + 1e-6  # precision matrix (diagonal)
            y_std = tf.sqrt(1. / y_prec, name='y_std')

        # define the loss:
        with tf.name_scope('kl_div'):
            down_sc = 1.0
            kl_div = down_sc * (kl + kl_prec)
            tf.summary.scalar('kl_div_average', kl_div)

        with tf.name_scope('expected_negloglikelihood'):
            # expected NLL for mean network
            mse_sum = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_pred), [1, 2, 3, 4]), 0)
            mse_sum *= num_data
            tf.summary.scalar('mse_sum', mse_sum)

            # expected NLL for heteroscedastic network
            # (previous ver, 12 Aug):
            # e_negloglike = tf.reduce_mean(tf.reduce_sum(tf.square(tf.mul(y_prec, (y - y_pred))), [1, 2, 3, 4]), 0) \
            #              - tf.reduce_mean(tf.reduce_sum(tf.log(y_prec), [1, 2, 3, 4]), 0)

            e_negloglike = tf.reduce_mean(tf.reduce_sum(tf.mul(y_prec, tf.square(y - y_pred)), [1, 2, 3, 4]), 0) \
                         - tf.reduce_mean(tf.reduce_sum(tf.log(y_prec), [1, 2, 3, 4]), 0)
            e_negloglike *= num_data
            tf.summary.scalar('e_negloglike', e_negloglike)

        with tf.name_scope('loss'):  # negative evidence lower bound (ELBO)
            cost = trade_off * (e_negloglike - kl_div) + (1. - trade_off) * (
            mse_sum - kl_div)
            tf.summary.scalar('neg_ELBO', cost)

        return y_pred, y_std, cost

    # -------- UTILITY ----------------
    def build_network(self, x, y, phase, keep_prob, params, trade_off,
                      num_data, cov_on, hetero, vardrop):
        if hetero:
            if vardrop:  # hetero + var. drop
                y_pred, y_std, cost = self.forwardpass_hetero_vardrop(x, y, phase, keep_prob, params, trade_off, num_data, cov_on)
            else:  # hetero
                y_pred, y_std, cost = self.forwardpass_hetero(x, y, phase)
        else:
            if vardrop:  # var. drop
                y_pred, cost = self.forwardpass_vardrop(x, y, phase, keep_prob, params, num_data)
            else:  # standard network
                y_pred, cost = self.forwardpass(x, y, phase)
            y_std = 1  # just constant number
        return y_pred, y_std, cost

    def scaled_prediction_mc(self, x, phase, keep_prob, params,
                             trade_off, num_data, cov_on, transform,
                             hetero, vardrop):
        x_mean = tf.constant(np.float32(transform['input_mean']), name='x_mean')
        x_std = tf.constant(np.float32(transform['input_std']), name='x_std')
        y_mean = tf.constant(np.float32(transform['output_mean']),
                             name='y_mean')
        y_std = tf.constant(np.float32(transform['output_std']), name='y_std')
        x_scaled = tf.div(x - x_mean, x_std)
        y = tf.placeholder(tf.float32, name='input_y')

        if hetero:
            if vardrop:
                y_norm, y_norm_std, _ = self.forwardpass_hetero_vardrop(x_scaled, y, phase, keep_prob, params, trade_off, num_data, cov_on)
            else:
                y_norm, y_norm_std, _ = self.forwardpass_hetero(x_scaled, y, phase)

            y_pred = tf.add(y_std * y_norm, y_mean, name='y_pred')
            y_pred_std = tf.mul(y_std, y_norm_std, name='y_pred_std')
        else:
            if vardrop:
                y_norm, _ = self.forwardpass_vardrop(x_scaled, y, phase, keep_prob, params, num_data)
            else:
                y_norm, _ = self.forwardpass(x_scaled, y, phase)

            y_pred = tf.add(y_std * y_norm, y_mean, name='y_pred')
            y_pred_std = 1  # just constant number
        return y_pred, y_pred_std

    def get_output_shape(self):
        return get_tensor_shape(self.y_pred)

class espcn_deconv(object):
    def __init__(self,
                 upsampling_rate,
                 out_channels,
                 layers=2,
                 filters_num=50,
                 bn=False):
        """
        3D ESPCN with deconvolution (in place of subpixel-shifted convolution)
        No shuffling needed for patch-loarder.
        Need to set opt['is_shuffle']=False

        Note:
            Backpropagation does not work for tf.nn.conv3d_transpose() for
            padding = 'VALID' for TF version before 0.12. So, the current
            version uses deconv with padding = 'SAME'. This needs to be fixed
            for a faithful reimplementation of ESPCN with a deconvolution layer.

        Args:
            input (tf tensor float 32): input tensor
            upsampling_rate (int): upsampling rate
            out_channels (int): number of channels in the output image
            layers (int): number of hidden layers
            filters_num(int): number of filters in the first layer
            bn (boolean): set True fro BatchNorm
        """
        self.upsampling_rate=upsampling_rate
        self.layers=layers
        self.out_channels=out_channels
        self.filters_num=filters_num
        self.bn = bn

    # ------------- STANDARD NETWORK ----------------
    def forwardpass(self, x, y, phase):
        net = []
        net = record_network(net, x)

        # define the network:
        n_f = self.filters_num
        lyr = 0
        while lyr < self.layers:
            if lyr==1: # second layer with kernel size 1 other layers three
                x = conv3d(x, filter_size=1, out_channels=n_f, name='conv_' + str(lyr + 1))
            else:
                x = conv3d(x, filter_size=3, out_channels=n_f, name='conv_' + str(lyr + 1))

            # double the num of features in the second lyr onward
            if lyr == 0: n_f = int(2 * n_f)
            net = record_network(net, x)

            # non-linearity + batch norm:
            x = batchnorm(x, phase, on=self.bn, name='BN%d' % len(net))
            x = tf.nn.relu(x, name='activation' % len(net))
            lyr += 1

        y_pred = deconv3d(tf.nn.relu(x),
                          filter_size=3 * self.upsampling_rate,
                          stride=self.upsampling_rate,
                          out_channels=self.out_channels,
                          name='deconv',
                          padding="SAME"
                          )
        net = record_network(net, y_pred)
        print_network(net)

        # define the loss:
        with tf.name_scope('loss'):
            cost = tf.reduce_mean(tf.square(y - y_pred))
        return y_pred, cost

    def scaled_prediction(self, x, phase, transform):
        x_mean = tf.constant(np.float32(transform['input_mean']), name='x_mean')
        x_std = tf.constant(np.float32(transform['input_std']), name='x_std')
        y_mean = tf.constant(np.float32(transform['output_mean']), name='y_mean')
        y_std = tf.constant(np.float32(transform['output_std']), name='y_std')
        x_scaled = tf.div(tf.sub(x, x_mean), x_std)

        y = tf.placeholder(tf.float32, name='input_y')
        y_norm, _ = self.forwardpass(x_scaled, y, phase)
        y_pred = tf.add(tf.mul(y_std, y_norm), y_mean, name='y_pred')
        return y_pred

    def get_output_shape(self):
        return get_tensor_shape(self.y_pred)


class unet(object):
    def __init__(self,
                upsampling_rate=2,
                out_channels=6,
                layers=3,
                filters_num=50,
                filter_size=3,
                conv_num=2,
                bn=True,
                is_concat=True):

        """
        3D U-net or SegNet type networks
        Currently, no extra convolutions are performed at the trough of the
        U-shaped architecture.

        Note:
            when input size is odd-numbered, cropping is applied in decoder
            network before concatenation. Another option is to pad, but not
            implemented yet.

        Args:
            upsampling_rate (int): upsampling rate
            out_channels (int): number of output channels
            layers (int): number of hidden layers
            filters_num (int): number of features in the first convolution
            filter_size (int): size of convolution filter. default = 3x3x3
            conv_num (int): number of convolutions in each level of encode/decoder
            bn (boolean): set True fro BatchNorm
            is_concat (boolean): set True for Unet, o/w defines a SegNet.
        """
        self.upsampling_rate = upsampling_rate
        self.layers = layers
        self.out_channels = out_channels
        self.filters_num = filters_num
        self.filter_size = filter_size
        self.conv_num=conv_num
        self.bn = bn
        self.is_concat = is_concat

    def forwardpass(self, input, phase):
        net=[]
        net=record_network(net, input)
        filters_num = self.filters_num
        down_h_convs = OrderedDict()

        # ---------- Encoder network ----------------
        for layer in range(self.layers):
            # Double the number of features:
            if layer!=0: filters_num = int(2 * filters_num)

            # convolutions:
            j=0
            while j<self.conv_num:
                input=conv3d(input, out_channels=filters_num, filter_size=self.filter_size, name='conv%d' % len(net), padding='SAME')
                net = record_network(net, input)
                input = batchnorm(input, phase, on=self.bn, name='BN%d' % len(net))
                input = tf.nn.relu(input, name='activation' % len(net))
                if j==self.conv_num-1:
                    down_h_convs[layer]=input
                j+=1

            # down-sample: conv with stride 2
            input = conv3d(input, out_channels=filters_num, filter_size=self.filter_size, stride=self.upsampling_rate, name='conv%d' % len(net), padding='SAME')
            net = record_network(net, input)
            input = batchnorm(input, phase, on=self.bn, name='BN%d' % len(net))
            input = tf.nn.relu(input, name='activation' % len(net))

        # ---------- Decoder network ----------------
        for layer in range(self.layers-1, -1, -1):
            # upsample:
            input = deconv3d(input, out_channels=filters_num, filter_size=self.upsampling_rate * self.filter_size, stride=self.upsampling_rate, name='deconv%d' % len(net), padding='SAME')
            net = record_network(net, input)
            input = batchnorm(input, phase, on=self.bn, name='BN%d' % len(net))
            input = tf.nn.relu(input, name='activation' % len(net))

            # concatenate or just crop (Unet or Segnet)
            input = crop_and_or_concat_basic(input, down_h_convs[layer], is_concat=self.is_concat, name='concat_or_crop%d'%len(net))
            net = record_network(net, input)

            # convolutions:
            j = 0
            while j < self.conv_num:
                # convolution
                input = conv3d(input, out_channels=filters_num, filter_size=self.filter_size, name='conv%d' % len(net), padding='SAME')
                net = record_network(net, input)
                input = batchnorm(input, phase, on=self.bn, name='BN%d' % len(net))
                input = tf.nn.relu(input, name='activation' % len(net))
                j += 1

            # halve the number of features:
            filters_num = int(filters_num / 2)

        # Last deconv:
        y_pred = deconv3d(input, out_channels=self.out_channels, filter_size=self.upsampling_rate * self.filter_size, stride=self.upsampling_rate, name='deconv%d' % len(net), padding='SAME')
        net= record_network(net, y_pred)

        # Print the architecture
        print_network(net)
        return y_pred

    def scaled_prediction(self, input, phase, transform):
        x_mean = tf.constant(np.float32(transform['input_mean']), name='x_mean')
        x_std = tf.constant(np.float32(transform['input_std']), name='x_std')
        y_mean = tf.constant(np.float32(transform['output_mean']),
                             name='y_mean')
        y_std = tf.constant(np.float32(transform['output_std']), name='y_std')
        x_scaled = tf.div(tf.sub(input, x_mean), x_std)

        y = self.forwardpass(x_scaled, phase)
        y_pred = tf.add(tf.mul(y_std, y), y_mean, name='y_pred')
        return y_pred

    def get_output_shape(self):
        return get_tensor_shape(self.y_pred)

    def cost(self, y, y_pred):
        return tf.reduce_mean(tf.square(y - y_pred))

# -------------------------- old stuff -----------------------------------------
# contains ESPCN with heteroscedastic likelihood and variational dropout

def inference(method, x, y, keep_prob, opt, trade_off=None):
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
    upsampling_rate = opt['upsampling_rate']
    no_channels = opt['no_channels']
    y_std = None

    if method == 'cnn_simple':
        h1_1 = conv3d(x, [3,3,3,no_channels,n_h1], [n_h1], 'conv_1')

        if opt['receptive_field_radius'] == 2:
            h1_2 = conv3d(tf.nn.relu(h1_1), [1,1,1,n_h1,n_h2], [n_h2], 'conv_2')
        elif opt['receptive_field_radius'] == 3:
            h1_2 = conv3d(tf.nn.relu(h1_1), [3,3,3,n_h1,n_h2], [n_h2], 'conv_2')
        elif opt['receptive_field_radius'] == 4:
            h1_2 = conv3d(tf.nn.relu(h1_1), [3,3,3,n_h1,n_h2], [n_h2], 'conv_2')
            h1_2 = conv3d(tf.nn.relu(h1_2), [3,3,3,n_h2,n_h2], [n_h2], 'conv_3')
        elif opt['receptive_field_radius'] == 5:
            h1_2 = conv3d(tf.nn.relu(h1_1), [3,3,3,n_h1,n_h2], [n_h2], 'conv_2')
            h1_2 = conv3d(tf.nn.relu(h1_2), [3,3,3,n_h2,n_h2], [n_h2], 'conv_3')
            h1_2 = conv3d(tf.nn.relu(h1_2), [3,3,3,n_h2,n_h2], [n_h2], 'conv_4')
        elif opt['receptive_field_radius'] > 5:
            h1_2 = conv3d(tf.nn.relu(h1_1), [3, 3, 3, n_h1, n_h2], [n_h2],'conv_2')
            lyr=3
            while lyr<(opt['receptive_field_radius']):
                h1_2 = conv3d(tf.nn.relu(h1_2), [3,3,3,n_h2,n_h2], [n_h2], 'conv_'+str(lyr+1))
                lyr+=1

        y_pred = conv3d(tf.nn.relu(h1_2),
                        [3,3,3,n_h2,no_channels*(upsampling_rate**3)],
                        [no_channels*(upsampling_rate**3)],
                        'conv_last')

        with tf.name_scope('loss'):
            cost = tf.reduce_mean(tf.square(y - y_pred))

    elif method == 'espcn_with_deconv':
        h1_1 = conv3d(x, [3,3,3,no_channels,n_h1], [n_h1], 'conv_1')

        if opt['receptive_field_radius'] == 2:
            h1_2 = conv3d(tf.nn.relu(h1_1), [1,1,1,n_h1,n_h2], [n_h2], 'conv_2')
        elif opt['receptive_field_radius'] == 3:
            h1_2 = conv3d(tf.nn.relu(h1_1), [3,3,3,n_h1,n_h2], [n_h2], 'conv_2')
        elif opt['receptive_field_radius'] == 4:
            h1_2 = conv3d(tf.nn.relu(h1_1), [3,3,3,n_h1,n_h2], [n_h2], 'conv_2')
            h1_2 = conv3d(tf.nn.relu(h1_2), [3,3,3,n_h2,n_h2], [n_h2], 'conv_3')
        elif opt['receptive_field_radius'] == 5:
            h1_2 = conv3d(tf.nn.relu(h1_1), [3,3,3,n_h1,n_h2], [n_h2], 'conv_2')
            h1_2 = conv3d(tf.nn.relu(h1_2), [3,3,3,n_h2,n_h2], [n_h2], 'conv_3')
            h1_2 = conv3d(tf.nn.relu(h1_2), [3,3,3,n_h2,n_h2], [n_h2], 'conv_4')
        elif opt['receptive_field_radius'] > 5:
            h1_2 = conv3d(tf.nn.relu(h1_1), [3, 3, 3, n_h1, n_h2], [n_h2],'conv_2')
            lyr=3
            while lyr<(opt['receptive_field_radius']):
                h1_2 = conv3d(tf.nn.relu(h1_2), [3,3,3,n_h2,n_h2], [n_h2], 'conv_'+str(lyr+1))
                lyr+=1

        y_pred = conv3d(tf.nn.relu(h1_2),
                        [3,3,3,n_h2,no_channels*(upsampling_rate**3)],
                        [no_channels*(upsampling_rate**3)],
                        'conv_last')

        with tf.name_scope('loss'):
            cost = tf.reduce_mean(tf.square(y - y_pred))

    elif method=='cnn_simple_L1':
        h1_1 = conv3d(x, [3, 3, 3, no_channels, n_h1], [n_h1], 'conv_1')

        if opt['receptive_field_radius'] == 2:
            h1_2 = conv3d(tf.nn.relu(h1_1), [1, 1, 1, n_h1, n_h2], [n_h2], 'conv_2')
        elif opt['receptive_field_radius'] == 3:
            h1_2 = conv3d(tf.nn.relu(h1_1), [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
        elif opt['receptive_field_radius'] == 4:
            h1_2 = conv3d(tf.nn.relu(h1_1), [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
            h1_2 = conv3d(tf.nn.relu(h1_2), [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_3')
        elif opt['receptive_field_radius'] == 5:
            h1_2 = conv3d(tf.nn.relu(h1_1), [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
            h1_2 = conv3d(tf.nn.relu(h1_2), [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_3')
            h1_2 = conv3d(tf.nn.relu(h1_2), [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_4')

        y_pred = conv3d(tf.nn.relu(h1_2),
                        [3, 3, 3, n_h2, no_channels * (upsampling_rate ** 3)],
                        [no_channels * (upsampling_rate ** 3)],
                        'conv_last')

        with tf.name_scope('loss'):
            cost = tf.reduce_mean(tf.abs(y - y_pred))

    elif method == 'cnn_heteroscedastic':
        with tf.name_scope('mean_network'):
            h1_1 = conv3d(x, [3, 3, 3, no_channels, n_h1], [n_h1], 'conv_1')
            if opt['receptive_field_radius'] == 2:
                h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_1), keep_prob),
                              [1, 1, 1, n_h1, n_h2], [n_h2], 'conv_2')
            elif opt['receptive_field_radius'] == 3:
                h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_1), keep_prob),
                              [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
            elif opt['receptive_field_radius'] == 4:
                h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_1), keep_prob),
                              [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
                h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_2), keep_prob),
                              [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_3')
            elif opt['receptive_field_radius'] == 5:
                h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_1), keep_prob),
                              [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
                h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_2), keep_prob),
                              [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_3')
                h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_2), keep_prob),
                              [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_4')

            y_pred = conv3d(tf.nn.dropout(tf.nn.relu(h1_2), keep_prob),
                            [3, 3, 3, n_h2, no_channels * (upsampling_rate ** 3)],
                            [no_channels * (upsampling_rate ** 3)], 'conv_last')

        with tf.name_scope('precision_network'):  # diagonality assumed
            h1_1 = conv3d(x, [3, 3, 3, no_channels, n_h1], [n_h1], 'conv_1')
            if opt['receptive_field_radius'] == 2:
                h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_1), keep_prob),
                              [1, 1, 1, n_h1, n_h2], [n_h2], 'conv_2')
            elif opt['receptive_field_radius'] == 3:
                h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_1), keep_prob),
                              [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
            elif opt['receptive_field_radius'] == 4:
                h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_1), keep_prob),
                              [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
                h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_2), keep_prob),
                              [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_3')
            elif opt['receptive_field_radius'] == 5:
                h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_1), keep_prob),
                              [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
                h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_2), keep_prob),
                              [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_3')
                h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_2), keep_prob),
                              [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_4')

            h_last = conv3d(tf.nn.dropout(tf.nn.relu(h1_2), keep_prob),
                            [3, 3, 3, n_h2, no_channels * (upsampling_rate ** 3)],
                            [no_channels * (upsampling_rate ** 3)], 'conv_last')
            y_prec = tf.nn.softplus(h_last) + 1e-6  # precision matrix (diagonal)
            y_std = tf.sqrt(1. / y_prec, name='y_std')

        with tf.name_scope('loss'):
            cost = tf.reduce_mean(tf.square(tf.mul(y_prec, (y - y_pred)))) \
                   - tf.reduce_mean(tf.log(y_prec))

        # with tf.name_scope('covariance_network'):  # diagonality assumed
        #     h1_1 = conv3d(x, [3, 3, 3, no_channels, n_h1], [n_h1], 'conv_1')
        #     if opt['receptive_field_radius'] == 2:
        #         h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_1), keep_prob),
        #                       [1, 1, 1, n_h1, n_h2], [n_h2], 'conv_2')
        #     elif opt['receptive_field_radius'] == 3:
        #         h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_1), keep_prob),
        #                       [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
        #     elif opt['receptive_field_radius'] == 4:
        #         h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_1), keep_prob),
        #                       [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
        #         h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_2), keep_prob),
        #                       [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_3')
        #     elif opt['receptive_field_radius'] == 5:
        #         h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_1), keep_prob),
        #                       [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
        #         h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_2), keep_prob),
        #                       [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_3')
        #         h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_2), keep_prob),
        #                       [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_4')
        #
        #     h_last = conv3d(tf.nn.dropout(tf.nn.relu(h1_2), keep_prob),
        #                     [3, 3, 3, n_h2, no_channels * (upsampling_rate ** 3)],
        #                     [no_channels * (upsampling_rate ** 3)], 'conv_last')
        #     y_cov = tf.nn.softplus(h_last) + 1e-6  # precision matrix (diagonal)
        #     y_std = tf.sqrt(y_cov, name='y_std')
        #
        # with tf.name_scope('loss'):
        #     cost = tf.reduce_mean(tf.square(tf.mul(1./y_cov, (y - y_pred)))) \
        #            + tf.reduce_mean(tf.log(y_cov))

    elif method == 'cnn_dropout':
        h1_1 = conv3d(x, [3, 3, 3, no_channels, n_h1], [n_h1], 'conv_1')

        if opt['receptive_field_radius'] == 2:
            h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_1), keep_prob),
                          [1, 1, 1, n_h1, n_h2], [n_h2], 'conv_2')
        elif opt['receptive_field_radius'] == 3:
            h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_1), keep_prob),
                          [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
        elif opt['receptive_field_radius'] == 4:
            h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_1), keep_prob),
                          [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
            h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_2), keep_prob),
                          [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_3')
        elif opt['receptive_field_radius'] == 5:
            h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_1), keep_prob),
                          [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
            h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_2), keep_prob),
                          [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_3')
            h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_2), keep_prob),
                          [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_4')

        y_pred = conv3d(tf.nn.dropout(tf.nn.relu(h1_2), keep_prob),
                        [3, 3, 3, n_h2, no_channels * (upsampling_rate ** 3)],
                        [no_channels * (upsampling_rate ** 3)], 'conv_last')

        with tf.name_scope('loss'):
            cost = tf.reduce_mean(tf.square(y - y_pred))

    elif method == 'cnn_gaussian_dropout':
        h1_1 = conv3d(x, [3, 3, 3, no_channels, n_h1], [n_h1], 'conv_1')
        params=None
        if opt['receptive_field_radius'] == 2:
            a1_2_drop, _ = normal_mult_noise(tf.nn.relu(h1_1), keep_prob, params, opt, 'mulnoise_1')
            h1_2 = conv3d(a1_2_drop, [1, 1, 1, n_h1, n_h2], [n_h2], 'conv_2')
        elif opt['receptive_field_radius'] == 3:
            a1_2_drop, _ = normal_mult_noise(tf.nn.relu(h1_1), keep_prob, params, opt, 'mulnoise_1')
            h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
        elif opt['receptive_field_radius'] == 4:
            a1_2_drop, _ = normal_mult_noise(tf.nn.relu(h1_1), keep_prob, params, opt, 'mulnoise_1')
            h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
            a1_2_drop, _ = normal_mult_noise(tf.nn.relu(h1_2), keep_prob, params, opt, 'mulnoise_2')
            h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_3')
        elif opt['receptive_field_radius'] == 5:
            a1_2_drop, _ = normal_mult_noise(tf.nn.relu(h1_1), keep_prob, params, opt, 'mulnoise_1')
            h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
            a1_2_drop, _ = normal_mult_noise(tf.nn.relu(h1_2), keep_prob, params, opt, 'mulnoise_2')
            h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_3')
            a1_2_drop, _ = normal_mult_noise(tf.nn.relu(h1_2), keep_prob, params, opt, 'mulnoise_3')
            h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_4')

        a1_2_drop, _ = normal_mult_noise(tf.nn.relu(h1_2), keep_prob, params, opt, 'mulnoise_last')
        y_pred = conv3d(a1_2_drop,
                        [3, 3, 3, n_h2, no_channels * (upsampling_rate ** 3)],
                        [no_channels * (upsampling_rate ** 3)],
                        'conv_last')

        with tf.name_scope('loss'):
            cost = tf.reduce_mean(tf.square(y - y_pred))

    elif method == 'cnn_variational_dropout' or \
         method == 'cnn_variational_dropout_layerwise' or \
         method == 'cnn_variational_dropout_channelwise' or \
         method == 'cnn_variational_dropout_average':

        if method == 'cnn_variational_dropout':
            params='weight'
        elif method == 'cnn_variational_dropout_layerwise':
            params='layer'
        elif method == 'cnn_variational_dropout_channelwise':
            params='channel'
        elif method == 'cnn_variational_dropout_average':
            params = 'weight_average'
        else:
            raise ValueError('no variational parameters specified!')

        h1_1 = conv3d(x, [3, 3, 3, no_channels, n_h1], [n_h1], 'conv_1')

        if opt['receptive_field_radius'] == 2:
            a1_2_drop, kl = normal_mult_noise(tf.nn.relu(h1_1), keep_prob, params, opt, 'mulnoise_1')
            h1_2 = conv3d(a1_2_drop, [1, 1, 1, n_h1, n_h2], [n_h2], 'conv_2')
        elif opt['receptive_field_radius'] == 3:
            a1_2_drop, kl = normal_mult_noise(tf.nn.relu(h1_1), keep_prob, params, opt, 'mulnoise_1')
            h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
        elif opt['receptive_field_radius'] == 4:
            a1_2_drop, kl_1 = normal_mult_noise(tf.nn.relu(h1_1), keep_prob, params, opt, 'mulnoise_1')
            h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
            a1_2_drop, kl_2 = normal_mult_noise(tf.nn.relu(h1_2), keep_prob, params, opt, 'mulnoise_2')
            h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_3')
            kl = kl_1 + kl_2
        elif opt['receptive_field_radius'] == 5:
            a1_2_drop, kl_1 = normal_mult_noise(tf.nn.relu(h1_1), keep_prob, params, opt, 'mulnoise_1')
            h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
            a1_2_drop, kl_2 = normal_mult_noise(tf.nn.relu(h1_2), keep_prob, params, opt, 'mulnoise_2')
            h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_3')
            a1_2_drop, kl_3 = normal_mult_noise(tf.nn.relu(h1_2), keep_prob, params, opt, 'mulnoise_3')
            h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_4')
            kl = kl_1 + kl_2 + kl_3
        a1_2_drop, kl_last = normal_mult_noise(tf.nn.relu(h1_2), keep_prob, params, opt, 'mulnoise_last')
        y_pred = conv3d(a1_2_drop,
                        [3, 3, 3, n_h2, no_channels * (upsampling_rate ** 3)],
                        [no_channels * (upsampling_rate ** 3)],
                        'conv_last')

        with tf.name_scope('kl_div'):
            down_sc = 1.0
            kl_div = down_sc * (kl + kl_last)
            tf.summary.scalar('kl_div_average', kl_div)

        with tf.name_scope('expected_negloglikelihood'):
            e_negloglike = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_pred),[1,2,3,4]),0)
            if not(method == 'cnn_variational_dropout_average'):
                e_negloglike = opt['train_noexamples'] * e_negloglike
            tf.summary.scalar('e_negloglike', e_negloglike)

        with tf.name_scope('loss'):  # negative evidence lower bound (ELBO)
            cost = tf.add(e_negloglike, -kl_div, name='neg_ELBO')
            tf.summary.scalar('neg_ELBO', cost)

    elif method=='cnn_heteroscedastic_variational' or \
         method=='cnn_heteroscedastic_variational_layerwise' or \
         method=='cnn_heteroscedastic_variational_channelwise' or \
         method=='cnn_heteroscedastic_variational_average' :

        if method == 'cnn_heteroscedastic_variational':
            params = 'weight'
        elif method == 'cnn_heteroscedastic_variational_average':
            params = 'weight_average'
        elif method == 'cnn_heteroscedastic_variational_layerwise':
            params = 'layer'
        elif method == 'cnn_heteroscedastic_variational_channelwise':
            params = 'channel'
        else:
            raise ValueError('no variational parameters specified!')

        with tf.name_scope('mean_network'):
            h1_1 = conv3d(x, [3, 3, 3, no_channels, n_h1], [n_h1], 'conv_1')

            if opt['receptive_field_radius'] == 2:
                a1_2_drop, kl = normal_mult_noise(tf.nn.relu(h1_1), keep_prob, params, opt, 'mulnoise_1')
                h1_2 = conv3d(a1_2_drop, [1, 1, 1, n_h1, n_h2], [n_h2], 'conv_2')
            elif opt['receptive_field_radius'] == 3:
                a1_2_drop, kl = normal_mult_noise(tf.nn.relu(h1_1), keep_prob, params, opt,'mulnoise_1')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
            elif opt['receptive_field_radius'] == 4:
                a1_2_drop, kl_1 = normal_mult_noise(tf.nn.relu(h1_1), keep_prob, params, opt, 'mulnoise_1')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
                a1_2_drop, kl_2 = normal_mult_noise(tf.nn.relu(h1_2), keep_prob, params, opt, 'mulnoise_2')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_3')
                kl = kl_1 + kl_2
            elif opt['receptive_field_radius'] == 5:
                a1_2_drop, kl_1 = normal_mult_noise(tf.nn.relu(h1_1), keep_prob, params, opt, 'mulnoise_1')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
                a1_2_drop, kl_2 = normal_mult_noise(tf.nn.relu(h1_2), keep_prob, params, opt, 'mulnoise_2')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_3')
                a1_2_drop, kl_3 = normal_mult_noise(tf.nn.relu(h1_2), keep_prob, params, opt, 'mulnoise_3')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_4')
                kl = kl_1 + kl_2 + kl_3
            a1_2_drop, kl_last = normal_mult_noise(tf.nn.relu(h1_2), keep_prob, params, opt, 'mulnoise_last')
            y_pred = conv3d(a1_2_drop,
                            [3, 3, 3, n_h2, no_channels * (upsampling_rate ** 3)],
                            [no_channels * (upsampling_rate ** 3)],
                            'conv_last')

        with tf.name_scope('kl_div'):
            down_sc = 1.0
            kl_div = down_sc*(kl + kl_last)
            tf.summary.scalar('kl_div', kl_div)

        with tf.name_scope('precision_network'):  # diagonality assumed
            h1_1 = conv3d(x, [3, 3, 3, no_channels, n_h1], [n_h1], 'conv_1')
            if opt['receptive_field_radius'] == 2:
                h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_1), keep_prob),
                              [1, 1, 1, n_h1, n_h2], [n_h2], 'conv_2')
            elif opt['receptive_field_radius'] == 3:
                h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_1), keep_prob),
                              [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
            elif opt['receptive_field_radius'] == 4:
                h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_1), keep_prob),
                              [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
                h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_2), keep_prob),
                              [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_3')
            elif opt['receptive_field_radius'] == 5:
                h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_1), keep_prob),
                              [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
                h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_2), keep_prob),
                              [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_3')
                h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_2), keep_prob),
                              [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_4')

            h_last = conv3d(tf.nn.dropout(tf.nn.relu(h1_2), keep_prob),
                            [3, 3, 3, n_h2, no_channels * (upsampling_rate ** 3)],
                            [no_channels * (upsampling_rate ** 3)], 'conv_last')
            y_prec = tf.nn.softplus(h_last) + 1e-6  # precision matrix (diagonal)
            y_std = tf.sqrt(1. / y_prec, name='y_std')

        with tf.name_scope('expected_negloglikelihood'):
            e_negloglike = tf.reduce_mean(tf.reduce_sum(tf.square(tf.mul(y_prec, (y - y_pred))), [1,2,3,4]), 0) \
                         - tf.reduce_mean(tf.reduce_sum(tf.log(y_prec), [1,2,3,4]), 0)
            if not (method == 'cnn_heteroscedastic_variational_average'):
                e_negloglike = opt['train_noexamples'] * e_negloglike
            tf.summary.scalar('e_negloglike', e_negloglike)

        with tf.name_scope('loss'):  # negative evidence lower bound (ELBO)
            cost = tf.add(e_negloglike, -kl_div, name='neg_ELBO')
            tf.summary.scalar('cost', cost)

    elif method == 'cnn_heteroscedastic_variational_downsc' or \
         method == 'cnn_heteroscedastic_variational_upsc' or \
         method == 'cnn_heteroscedastic_variational_layerwise_downsc' or \
         method == 'cnn_heteroscedastic_variational_channelwise_downsc':

        if method == 'cnn_heteroscedastic_variational_downsc':
            params = 'weight'
            sc = 0.3
        elif method == 'cnn_heteroscedastic_variational_upsc':
            params = 'weight'
            sc = 3.0
        elif method == 'cnn_heteroscedastic_variational_layerwise_downsc':
            params = 'layer'
            sc = 0.3
        elif method == 'cnn_heteroscedastic_variational_channelwise_downsc':
            params = 'channel'
            sc = 0.3
        else:
            raise ValueError('no variational parameters specified!')

        with tf.name_scope('mean_network'):
            h1_1 = conv3d(x, [3, 3, 3, no_channels, n_h1], [n_h1], 'conv_1')

            if opt['receptive_field_radius'] == 2:
                a1_2_drop, kl = normal_mult_noise(tf.nn.relu(h1_1), keep_prob, params, opt, 'mulnoise_1')
                h1_2 = conv3d(a1_2_drop, [1, 1, 1, n_h1, n_h2], [n_h2], 'conv_2')
            elif opt['receptive_field_radius'] == 3:
                a1_2_drop, kl = normal_mult_noise(tf.nn.relu(h1_1), keep_prob, params, opt, 'mulnoise_1')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
            elif opt['receptive_field_radius'] == 4:
                a1_2_drop, kl_1 = normal_mult_noise(tf.nn.relu(h1_1), keep_prob, params, opt, 'mulnoise_1')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
                a1_2_drop, kl_2 = normal_mult_noise(tf.nn.relu(h1_2), keep_prob, params, opt, 'mulnoise_2')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_3')
                kl = kl_1 + kl_2
            elif opt['receptive_field_radius'] == 5:
                a1_2_drop, kl_1 = normal_mult_noise(tf.nn.relu(h1_1), keep_prob, params, opt, 'mulnoise_1')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
                a1_2_drop, kl_2 = normal_mult_noise(tf.nn.relu(h1_2), keep_prob, params, opt, 'mulnoise_2')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_3')
                a1_2_drop, kl_3 = normal_mult_noise(tf.nn.relu(h1_2), keep_prob, params, opt, 'mulnoise_3')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_4')
                kl = kl_1 + kl_2 + kl_3
            a1_2_drop, kl_last = normal_mult_noise(tf.nn.relu(h1_2), keep_prob, params, opt, 'mulnoise_last')
            y_pred = conv3d(a1_2_drop,
                            [3, 3, 3, n_h2, no_channels * (upsampling_rate ** 3)],
                            [no_channels * (upsampling_rate ** 3)],
                            'conv_last')

        with tf.name_scope('kl_div'):
            kl_div = sc * (kl + kl_last)
            tf.summary.scalar('kl_div', kl_div)

        with tf.name_scope('precision_network'):  # diagonality assumed
            h1_1 = conv3d(x, [3, 3, 3, no_channels, n_h1], [n_h1], 'conv_1')
            if opt['receptive_field_radius'] == 2:
                h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_1), keep_prob),
                              [1, 1, 1, n_h1, n_h2], [n_h2], 'conv_2')
            elif opt['receptive_field_radius'] == 3:
                h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_1), keep_prob),
                              [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
            elif opt['receptive_field_radius'] == 4:
                h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_1), keep_prob),
                              [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
                h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_2), keep_prob),
                              [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_3')
            elif opt['receptive_field_radius'] == 5:
                h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_1), keep_prob),
                              [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
                h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_2), keep_prob),
                              [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_3')
                h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_2), keep_prob),
                              [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_4')

            h_last = conv3d(tf.nn.dropout(tf.nn.relu(h1_2), keep_prob),
                            [3, 3, 3, n_h2, no_channels * (upsampling_rate ** 3)],
                            [no_channels * (upsampling_rate ** 3)], 'conv_last')
            y_prec = tf.nn.softplus(h_last) + 1e-6  # precision matrix (diagonal)
            y_std = tf.sqrt(1. / y_prec, name='y_std')

        with tf.name_scope('expected_negloglikelihood'):
            e_negloglike = tf.reduce_mean(tf.reduce_sum(tf.square(tf.mul(y_prec, (y - y_pred))), [1, 2, 3, 4]), 0) \
                           - tf.reduce_mean(tf.reduce_sum(tf.log(y_prec), [1, 2, 3, 4]), 0)
            if not (method == 'cnn_heteroscedastic_variational_average'):
                e_negloglike = opt['train_noexamples'] * e_negloglike
            tf.summary.scalar('e_negloglike', e_negloglike)

        with tf.name_scope('loss'):  # negative evidence lower bound (ELBO)
            cost = tf.add(e_negloglike, -kl_div, name='neg_ELBO')
            tf.summary.scalar('cost', cost)

    elif method == 'cnn_heteroscedastic_variational_hybrid_control' or \
         method == 'cnn_heteroscedastic_variational_channelwise_hybrid_control' or \
         method == 'cnn_heteroscedastic_variational_downsc_control' or \
         method == 'cnn_heteroscedastic_variational_upsc_control':

        if method == 'cnn_heteroscedastic_variational_hybrid_control':
            params = 'weight'
        elif method == 'cnn_heteroscedastic_variational_channelwise_hybrid_control':
            params = 'channel'
        elif method == 'cnn_heteroscedastic_variational_downsc_control' or \
             method == 'cnn_heteroscedastic_variational_upsc_control':
            params = 'weight'
        else:
            raise ValueError('no variational parameters specified!')

        with tf.name_scope('mean_network'):
            h1_1 = conv3d(x, [3, 3, 3, no_channels, n_h1], [n_h1], 'conv_1')

            if opt['receptive_field_radius'] == 2:
                a1_2_drop, kl = normal_mult_noise(tf.nn.relu(h1_1), keep_prob, params, opt, 'mulnoise_1')
                h1_2 = conv3d(a1_2_drop, [1, 1, 1, n_h1, n_h2], [n_h2], 'conv_2')
            elif opt['receptive_field_radius'] == 3:
                a1_2_drop, kl = normal_mult_noise(tf.nn.relu(h1_1), keep_prob, params, opt, 'mulnoise_1')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
            elif opt['receptive_field_radius'] == 4:
                a1_2_drop, kl_1 = normal_mult_noise(tf.nn.relu(h1_1), keep_prob, params, opt, 'mulnoise_1')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
                a1_2_drop, kl_2 = normal_mult_noise(tf.nn.relu(h1_2), keep_prob, params, opt, 'mulnoise_2')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_3')
                kl = kl_1 + kl_2
            elif opt['receptive_field_radius'] == 5:
                a1_2_drop, kl_1 = normal_mult_noise(tf.nn.relu(h1_1), keep_prob, params, opt, 'mulnoise_1')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
                a1_2_drop, kl_2 = normal_mult_noise(tf.nn.relu(h1_2), keep_prob, params, opt, 'mulnoise_2')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_3')
                a1_2_drop, kl_3 = normal_mult_noise(tf.nn.relu(h1_2), keep_prob, params, opt, 'mulnoise_3')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_4')
                kl = kl_1 + kl_2 + kl_3
            a1_2_drop, kl_last = normal_mult_noise(tf.nn.relu(h1_2), keep_prob, params, opt, 'mulnoise_last')
            y_pred = conv3d(a1_2_drop,
                            [3, 3, 3, n_h2, no_channels * (upsampling_rate ** 3)],
                            [no_channels * (upsampling_rate ** 3)],
                            'conv_last')

        with tf.name_scope('kl_div'):
            down_sc = 1.0
            kl_div = down_sc * (kl + kl_last)
            tf.summary.scalar('kl_div', kl_div)

        with tf.name_scope('precision_network'):  # diagonality assumed
            h1_1 = conv3d(x, [3, 3, 3, no_channels, n_h1], [n_h1], 'conv_1')
            if opt['receptive_field_radius'] == 2:
                h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_1), keep_prob),
                              [1, 1, 1, n_h1, n_h2], [n_h2], 'conv_2')
            elif opt['receptive_field_radius'] == 3:
                h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_1), keep_prob),
                              [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
            elif opt['receptive_field_radius'] == 4:
                h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_1), keep_prob),
                              [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
                h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_2), keep_prob),
                              [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_3')
            elif opt['receptive_field_radius'] == 5:
                h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_1), keep_prob),
                              [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
                h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_2), keep_prob),
                              [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_3')
                h1_2 = conv3d(tf.nn.dropout(tf.nn.relu(h1_2), keep_prob),
                              [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_4')

            h_last = conv3d(tf.nn.dropout(tf.nn.relu(h1_2), keep_prob),
                            [3, 3, 3, n_h2, no_channels * (upsampling_rate ** 3)],
                            [no_channels * (upsampling_rate ** 3)], 'conv_last')
            y_prec = tf.nn.softplus(h_last) + 1e-6  # precision matrix (diagonal)
            y_std = tf.sqrt(1. / y_prec, name='y_std')

        with tf.name_scope('expected_negloglikelihood'):
            if method == 'cnn_heteroscedastic_variational_hybrid_control' or \
            method == 'cnn_heteroscedastic_variational_channelwise_hybrid_control':
                mse_sum = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_pred),[1,2,3,4]),0)
                mse_sum = opt['train_noexamples'] * mse_sum
                tf.summary.scalar('mse_sum', mse_sum)

            e_negloglike = tf.reduce_mean(tf.reduce_sum(tf.square(tf.mul(y_prec, (y - y_pred))), [1, 2, 3, 4]), 0) \
                           - tf.reduce_mean(tf.reduce_sum(tf.log(y_prec), [1, 2, 3, 4]), 0)
            e_negloglike = opt['train_noexamples'] * e_negloglike
            tf.summary.scalar('e_negloglike', e_negloglike)

        with tf.name_scope('loss'):  # negative evidence lower bound (ELBO)
            if method == 'cnn_heteroscedastic_variational_hybrid_control' or \
               method == 'cnn_heteroscedastic_variational_channelwise_hybrid_control':
                cost = trade_off*(e_negloglike - kl_div) + (1.- trade_off)*(mse_sum- kl_div)
            elif method == 'cnn_heteroscedastic_variational_downsc_control':
                cost = tf.add(e_negloglike, -trade_off * kl_div, name='neg_ELBO')
            elif method == 'cnn_heteroscedastic_variational_upsc_control':
                cost = tf.add(e_negloglike, -trade_off * kl_div, name='neg_ELBO')
            tf.summary.scalar('cost', cost)

    elif method == 'cnn_heteroscedastic_variational_cov' or \
         method == 'cnn_heteroscedastic_variational_layerwise_cov' or \
         method == 'cnn_heteroscedastic_variational_channelwise_cov':

        if method == 'cnn_heteroscedastic_variational_cov':
            params = 'weight'
        elif method == 'cnn_heteroscedastic_variational_layerwise_cov':
            params = 'layer'
        elif method == 'cnn_heteroscedastic_variational_channelwise_cov':
            params = 'channel'
        else:
            raise ValueError('no variational parameters specified!')

        with tf.name_scope('mean_network'):
            h1_1 = conv3d(x, [3, 3, 3, no_channels, n_h1], [n_h1], 'conv_1')

            if opt['receptive_field_radius'] == 2:
                a1_2_drop, kl_mean = normal_mult_noise(tf.nn.relu(h1_1), keep_prob, params, opt, 'mulnoise_1')
                h1_2 = conv3d(a1_2_drop, [1, 1, 1, n_h1, n_h2], [n_h2], 'conv_2')
            elif opt['receptive_field_radius'] == 3:
                a1_2_drop, kl_mean = normal_mult_noise(tf.nn.relu(h1_1), keep_prob, params, opt, 'mulnoise_1')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
            elif opt['receptive_field_radius'] == 4:
                a1_2_drop, kl_1 = normal_mult_noise(tf.nn.relu(h1_1), keep_prob, params, opt, 'mulnoise_1')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
                a1_2_drop, kl_2 = normal_mult_noise(tf.nn.relu(h1_2), keep_prob, params, opt, 'mulnoise_2')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_3')
                kl_mean = kl_1 + kl_2
            elif opt['receptive_field_radius'] == 5:
                a1_2_drop, kl_1 = normal_mult_noise(tf.nn.relu(h1_1), keep_prob, params, opt, 'mulnoise_1')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
                a1_2_drop, kl_2 = normal_mult_noise(tf.nn.relu(h1_2), keep_prob, params, opt, 'mulnoise_2')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_3')
                a1_2_drop, kl_3 = normal_mult_noise(tf.nn.relu(h1_2), keep_prob, params, opt, 'mulnoise_3')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_4')
                kl_mean = kl_1 + kl_2 + kl_3
            a1_2_drop, kl_mean_last = normal_mult_noise(tf.nn.relu(h1_2), keep_prob, params, opt, 'mulnoise_last')
            y_pred = conv3d(a1_2_drop,
                            [3, 3, 3, n_h2, no_channels * (upsampling_rate ** 3)],
                            [no_channels * (upsampling_rate ** 3)],
                            'conv_last')

        with tf.name_scope('kl_div_mean'):
            down_sc = 1.0
            kl_div_mean = down_sc * (kl_mean + kl_mean_last)
            tf.summary.scalar('kl_div_mean', kl_div_mean)

        with tf.name_scope('precision_network'):  # diagonality assumed
            h1_1 = conv3d(x, [3, 3, 3, no_channels, n_h1], [n_h1], 'conv_1')

            if opt['receptive_field_radius'] == 2:
                a1_2_drop, kl_prec = normal_mult_noise(tf.nn.relu(h1_1), keep_prob, params, opt, 'mulnoise_1')
                h1_2 = conv3d(a1_2_drop, [1, 1, 1, n_h1, n_h2], [n_h2], 'conv_2')
            elif opt['receptive_field_radius'] == 3:
                a1_2_drop, kl_prec = normal_mult_noise(tf.nn.relu(h1_1), keep_prob, params, opt, 'mulnoise_1')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
            elif opt['receptive_field_radius'] == 4:
                a1_2_drop, kl_1 = normal_mult_noise(tf.nn.relu(h1_1), keep_prob, params, opt, 'mulnoise_1')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
                a1_2_drop, kl_2 = normal_mult_noise(tf.nn.relu(h1_2), keep_prob, params, opt, 'mulnoise_2')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_3')
                kl_prec = kl_1 + kl_2
            elif opt['receptive_field_radius'] == 5:
                a1_2_drop, kl_1 = normal_mult_noise(tf.nn.relu(h1_1), keep_prob, params, opt, 'mulnoise_1')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
                a1_2_drop, kl_2 = normal_mult_noise(tf.nn.relu(h1_2), keep_prob, params, opt, 'mulnoise_2')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_3')
                a1_2_drop, kl_3 = normal_mult_noise(tf.nn.relu(h1_2), keep_prob, params, opt, 'mulnoise_3')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_4')
                kl_prec = kl_1 + kl_2 + kl_3

            a1_2_drop, kl_prec_last = normal_mult_noise(tf.nn.relu(h1_2), keep_prob, params, opt, 'mulnoise_last')
            h_last = conv3d(tf.nn.dropout(tf.nn.relu(h1_2), keep_prob),
                            [3, 3, 3, n_h2, no_channels * (upsampling_rate ** 3)],
                            [no_channels * (upsampling_rate ** 3)], 'conv_last')
            y_prec = tf.nn.softplus(h_last) + 1e-6  # precision matrix (diagonal)
            y_std = tf.sqrt(1. / y_prec, name='y_std')

        with tf.name_scope('kl_div_prec'):
            down_sc = 1.0
            kl_div_prec = down_sc * (kl_prec + kl_prec_last)
            tf.summary.scalar('kl_div_prec', kl_div_prec)

        with tf.name_scope('expected_negloglikelihood'):
            e_negloglike = tf.reduce_mean(tf.reduce_sum(tf.square(tf.mul(y_prec, (y - y_pred))), [1, 2, 3, 4]), 0) \
                           - tf.reduce_mean(tf.reduce_sum(tf.log(y_prec), [1, 2, 3, 4]), 0)
            if not (method == 'cnn_heteroscedastic_variational_average'):
                e_negloglike = opt['train_noexamples'] * e_negloglike
            tf.summary.scalar('e_negloglike', e_negloglike)

        with tf.name_scope('loss'):  # negative evidence lower bound (ELBO)
            cost = tf.add(e_negloglike, -kl_div_mean-kl_div_prec, name='neg_ELBO')
            tf.summary.scalar('cost', cost)

    elif method == 'cnn_heteroscedastic_variational_cov_hybrid' or \
         method == 'cnn_heteroscedastic_variational_layerwise_cov_hybrid' or \
         method == 'cnn_heteroscedastic_variational_channelwise_cov_hybrid':

        if method == 'cnn_heteroscedastic_variational_cov_hybrid':
            params = 'weight'
        elif method == 'cnn_heteroscedastic_variational_layerwise_cov_hybrid':
            params = 'layer'
        elif method == 'cnn_heteroscedastic_variational_channelwise_cov_hybrid':
            params = 'channel'
        else:
            raise ValueError('no variational parameters specified!')

        with tf.name_scope('mean_network'):
            h1_1 = conv3d(x, [3, 3, 3, no_channels, n_h1], [n_h1], 'conv_1')

            if opt['receptive_field_radius'] == 2:
                a1_2_drop, kl_mean = normal_mult_noise(tf.nn.relu(h1_1), keep_prob, params, opt, 'mulnoise_1')
                h1_2 = conv3d(a1_2_drop, [1, 1, 1, n_h1, n_h2], [n_h2], 'conv_2')
            elif opt['receptive_field_radius'] == 3:
                a1_2_drop, kl_mean = normal_mult_noise(tf.nn.relu(h1_1), keep_prob, params, opt, 'mulnoise_1')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
            elif opt['receptive_field_radius'] == 4:
                a1_2_drop, kl_1 = normal_mult_noise(tf.nn.relu(h1_1), keep_prob, params, opt, 'mulnoise_1')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
                a1_2_drop, kl_2 = normal_mult_noise(tf.nn.relu(h1_2), keep_prob, params, opt, 'mulnoise_2')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_3')
                kl_mean = kl_1 + kl_2
            elif opt['receptive_field_radius'] == 5:
                a1_2_drop, kl_1 = normal_mult_noise(tf.nn.relu(h1_1), keep_prob, params, opt, 'mulnoise_1')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
                a1_2_drop, kl_2 = normal_mult_noise(tf.nn.relu(h1_2), keep_prob, params, opt, 'mulnoise_2')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_3')
                a1_2_drop, kl_3 = normal_mult_noise(tf.nn.relu(h1_2), keep_prob, params, opt, 'mulnoise_3')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_4')
                kl_mean = kl_1 + kl_2 + kl_3
            a1_2_drop, kl_mean_last = normal_mult_noise(tf.nn.relu(h1_2), keep_prob, params, opt, 'mulnoise_last')
            y_pred = conv3d(a1_2_drop,
                            [3, 3, 3, n_h2, no_channels * (upsampling_rate ** 3)],
                            [no_channels * (upsampling_rate ** 3)],
                            'conv_last')

        with tf.name_scope('kl_div_mean'):
            down_sc = 1.0
            kl_div_mean = down_sc * (kl_mean + kl_mean_last)
            tf.summary.scalar('kl_div_mean', kl_div_mean)

        with tf.name_scope('precision_network'):  # diagonality assumed
            h1_1 = conv3d(x, [3, 3, 3, no_channels, n_h1], [n_h1], 'conv_1')

            if opt['receptive_field_radius'] == 2:
                a1_2_drop, kl_prec = normal_mult_noise(tf.nn.relu(h1_1), keep_prob, params, opt, 'mulnoise_1')
                h1_2 = conv3d(a1_2_drop, [1, 1, 1, n_h1, n_h2], [n_h2], 'conv_2')
            elif opt['receptive_field_radius'] == 3:
                a1_2_drop, kl_prec = normal_mult_noise(tf.nn.relu(h1_1), keep_prob, params, opt, 'mulnoise_1')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
            elif opt['receptive_field_radius'] == 4:
                a1_2_drop, kl_1 = normal_mult_noise(tf.nn.relu(h1_1), keep_prob, params, opt, 'mulnoise_1')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
                a1_2_drop, kl_2 = normal_mult_noise(tf.nn.relu(h1_2), keep_prob, params, opt, 'mulnoise_2')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_3')
                kl_prec = kl_1 + kl_2
            elif opt['receptive_field_radius'] == 5:
                a1_2_drop, kl_1 = normal_mult_noise(tf.nn.relu(h1_1), keep_prob, params, opt, 'mulnoise_1')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h1, n_h2], [n_h2], 'conv_2')
                a1_2_drop, kl_2 = normal_mult_noise(tf.nn.relu(h1_2), keep_prob, params, opt, 'mulnoise_2')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_3')
                a1_2_drop, kl_3 = normal_mult_noise(tf.nn.relu(h1_2), keep_prob, params, opt, 'mulnoise_3')
                h1_2 = conv3d(a1_2_drop, [3, 3, 3, n_h2, n_h2], [n_h2], 'conv_4')
                kl_prec = kl_1 + kl_2 + kl_3

            a1_2_drop, kl_prec_last = normal_mult_noise(tf.nn.relu(h1_2), keep_prob, params, opt, 'mulnoise_last')
            h_last = conv3d(tf.nn.dropout(tf.nn.relu(h1_2), keep_prob),
                            [3, 3, 3, n_h2, no_channels * (upsampling_rate ** 3)],
                            [no_channels * (upsampling_rate ** 3)], 'conv_last')
            y_prec = tf.nn.softplus(h_last) + 1e-6  # precision matrix (diagonal)
            y_std = tf.sqrt(1. / y_prec, name='y_std')

        with tf.name_scope('kl_div_prec'):
            down_sc = 1.0
            kl_div_prec = down_sc * (kl_prec + kl_prec_last)
            tf.summary.scalar('kl_div_prec', kl_div_prec)

        with tf.name_scope('expected_negloglikelihood'):
            mse_sum = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_pred), [1, 2, 3, 4]), 0)
            mse_sum = opt['train_noexamples'] * mse_sum
            tf.summary.scalar('mse_sum', mse_sum)

            e_negloglike = tf.reduce_mean(tf.reduce_sum(tf.square(tf.mul(y_prec, (y - y_pred))), [1, 2, 3, 4]), 0) \
                           - tf.reduce_mean(tf.reduce_sum(tf.log(y_prec), [1, 2, 3, 4]), 0)
            e_negloglike = opt['train_noexamples'] * e_negloglike
            tf.summary.scalar('e_negloglike', e_negloglike)

        with tf.name_scope('loss'):  # negative evidence lower bound (ELBO)
            cost = trade_off * (e_negloglike - kl_div_mean - kl_div_prec) \
                   + (1. - trade_off) * (mse_sum - kl_div_mean)
            tf.summary.scalar('cost', cost)

    elif method == 'cnn_residual':
        h1 = tf.nn.relu(conv3d(x, [3,3,3,no_channels,n_h1], [n_h1], '1'))
        # Residual blocks:
        # todo: include BN
        h2 = residual_block(h1, n_h1, n_h1, 'res2')
        h3 = residual_block(h2, n_h1, n_h1, 'res3')
        # Output
        h4 = conv3d(h3, [3,3,3,n_h1,n_h2], [n_h2], '4')
        h5 = residual_block(h4, n_h2, n_h2, 'res5')
        h6 = residual_block(h5, n_h2, n_h2, 'res6')
        y_pred = conv3d(h6, [1,1,1,n_h2,no_channels*(upsampling_rate**3)],
                            [no_channels*(upsampling_rate**3)], '7')
    else:
        raise ValueError('The chosen method not available ...')
    return y_pred, y_std, cost


def scaled_prediction(method, x, y, keep_prob, transform, opt, trade_off):
    x_mean = tf.constant(np.float32(transform['input_mean']), name='x_mean')
    x_std = tf.constant(np.float32(transform['input_std']), name='x_std')
    y_mean = tf.constant(np.float32(transform['output_mean']), name='y_mean')
    y_std = tf.constant(np.float32(transform['output_std']), name='y_std')
    # x_scaled = tf.div(tf.sub(x - transform['input_mean'), transform['input_std'])
    x_scaled = tf.div(tf.sub(x, x_mean), x_std)
    y, y_uncertainty, cost = inference(method, x_scaled, y, keep_prob, opt, trade_off)
    y_pred = tf.add(tf.mul(y_std, y), y_mean, name='y_pred')

    if opt['method']=='cnn_simple' or \
       opt['method']=='cnn_simple_L1' or \
       opt['method']=='cnn_dropout' or \
       opt['method']=='cnn_gaussian_dropout' or\
       opt['method']=='cnn_variational_dropout' or \
       opt['method']=='cnn_variational_dropout_channelwise':
        y_pred_std = 1
    else:
        y_pred_std = tf.mul(y_std, y_uncertainty, name='y_pred_std')
    return y_pred, y_pred_std




def get_tradeoff_values(opt):
    n_epochs = opt['n_epochs']
    tradeoff_list = np.zeros(n_epochs)
    if opt['method'] == 'cnn_heteroscedastic_variational_hybrid_control' or \
       opt['method'] == 'cnn_heteroscedastic_variational_channelwise_hybrid_control' or \
       opt['method'] == 'cnn_heteroscedastic_variational_cov_hybrid' or \
       opt['method'] == 'cnn_heteroscedastic_variational_layerwise_cov_hybrid' or \
       opt['method'] == 'cnn_heteroscedastic_variational_channelwise_cov_hybrid':

        print('apply trade-off!')
        init_idx = n_epochs//4  # intial stable training with std variational dropout loss
        freq = 1
        counter = 0
        rate  = 1./(len(range(init_idx,3*init_idx))//freq)
        for idx in range(init_idx,3*init_idx):
            if (counter+1)%freq==0:
                tradeoff_list[idx] = tradeoff_list[idx-1] + rate
                counter=0
            else:
                tradeoff_list[idx] = tradeoff_list[idx-1]
                counter+=1
        tradeoff_list[3*init_idx:]=1.  # fine-tune with the true cost function.
    else:
        print('no trade off needed!')
    return tradeoff_list

def get_tradeoff_values_v2(method, n_epochs):
    tradeoff_list = np.zeros(n_epochs)
    if method == 'cnn_heteroscedastic_variational_hybrid_control' or \
       method== 'cnn_heteroscedastic_variational_channelwise_hybrid_control' or \
       method == 'cnn_heteroscedastic_variational_cov_hybrid' or \
       method == 'cnn_heteroscedastic_variational_layerwise_cov_hybrid' or \
       method == 'cnn_heteroscedastic_variational_channelwise_cov_hybrid':

        print('apply trade-off!')
        init_idx = n_epochs//4  # intial stable training with std variational dropout loss
        freq = 1
        counter = 0
        rate  = 1./(len(range(init_idx,3*init_idx))//freq)
        for idx in range(init_idx,3*init_idx):
            if (counter+1)%freq==0:
                tradeoff_list[idx] = tradeoff_list[idx-1] + rate
                counter=0
            else:
                tradeoff_list[idx] = tradeoff_list[idx-1]
                counter+=1
        tradeoff_list[3*init_idx:]=1.  # fine-tune with the true cost function.
    else:
        print('no trade off needed!')
    return tradeoff_list
