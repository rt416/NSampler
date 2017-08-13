import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from collections import OrderedDict

############################################################################
# ------------------------ moved from prvious model.py --------------------
############################################################################


def print_network(network):
    """network (list): list of """
    print("\n------")
    print("Model:")
    print("------")
    for item in network:
        print(item)

    print("")
    total_parameters=count_num_params()
    print("parameters: " + str(total_parameters))
    print("")

def record_network(network, var, name=None):
    if name==None:
        network.append((str(var.name),
                        str(var.get_shape())))
    else:
        network.append((name,
                        str(var.get_shape())))
    return network

def count_num_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        # print(variable.name,variable.get_shape(), variable_parametes)
        total_parameters += variable_parametes
    return total_parameters


def get_weights(filter_shape, W_init=None, name=None):
    if W_init == None:
        # He/Xavier
        prod_length = len(filter_shape) - 1
        stddev = np.sqrt(2.0 / np.prod(filter_shape[:prod_length]))
        W_init = tf.random_normal(filter_shape, stddev=stddev)
    return tf.Variable(W_init, name=name)


def conv3d(input_batch, out_channels, filter_size=3, stride=1, name='',
           summary=True, padding='VALID'):
    """Return the 3D convolution"""
    with tf.name_scope(name):
        in_channels=int(input_batch.get_shape()[-1])
        w = get_weights([filter_size,filter_size,filter_size,in_channels,out_channels])
        variable_summaries(w, summary)

        b = tf.Variable(tf.constant(1e-2,dtype=tf.float32,shape=[out_channels]))
        variable_summaries(b, summary)

        z = tf.nn.conv3d(input_batch, w, strides=(1, stride, stride, stride, 1), padding=padding)
        z = tf.nn.bias_add(z, b)
        variable_summaries(z, summary)
    return z


def conv3d_vardrop_LRT(input_batch, out_channels, params, keep_prob,
                       filter_size=3, stride=1, determinisitc=False,
                       name='', summary=True, padding='VALID'):
    """
    Return the activation function after 3D convolution with variational dropout
    and the corresponding KL term.
    """

    with tf.name_scope(name):

        in_channels = int(input_batch.get_shape()[-1])

        # mean of the filter
        w = get_weights([filter_size, filter_size, filter_size, in_channels, out_channels])
        variable_summaries(w, summary)
        b = tf.Variable(tf.constant(1e-2, dtype=tf.float32, shape=[out_channels]))
        variable_summaries(b, summary)
        mean = tf.nn.conv3d(input_batch, w, strides=(1, stride, stride, stride, 1), padding=padding)
        mean = tf.nn.bias_add(mean, b)

        if determinisitc:  # turn off the multiplicative noise
            kl = 0
            return mean, kl

        # std of the filter
        if params=='weight':  # separate variational parameter for each weight
            w_init = tf.constant(np.float32(1e-4 * np.ones((filter_size, filter_size, filter_size, in_channels, out_channels))))
            rho = get_weights([filter_size, filter_size, filter_size, in_channels, out_channels], W_init=w_init, name='rho')
            alpha = tf.minimum(tf.nn.softplus(rho), 1., name='std')
            kl = kl_log_uniform_prior(alpha, name='kl')
            variable_summaries(alpha, summary)
            std = tf.sqrt(tf.nn.conv3d(tf.square(input_batch), alpha * tf.square(w), strides=(1, stride, stride, stride, 1), padding=padding))

        elif params=='channel': # separate variational parameter for each kernel
            w_init = tf.constant(np.float32(1e-4 * np.ones((1, 1, 1, 1, out_channels))))
            rho = get_weights([1, 1, 1, 1, out_channels], W_init=w_init, name='rho')
            alpha = tf.minimum(tf.nn.softplus(rho), 1., name='std')
            kl = np.prod(get_tensor_shape(w)[:4]) * kl_log_uniform_prior(alpha, name='kl')
            variable_summaries(alpha, summary)
            std = tf.sqrt(tf.nn.conv3d(tf.square(input_batch), alpha * tf.square(w), strides=(1, stride, stride, stride, 1), padding=padding))
        elif params == 'layer':  # separate variational parameter for each layer
            w_init = tf.constant(np.float32(1e-4))
            rho = get_weights(filter_shape=None, W_init=w_init, name='rho')
            alpha = tf.minimum(tf.nn.softplus(rho), 1., name='std')
            kl = np.prod(get_tensor_shape(w)) * kl_log_uniform_prior(alpha, name='kl')
            variable_summaries(alpha, summary)
            std = tf.sqrt(tf.nn.conv3d(tf.square(input_batch), alpha * tf.square(w), strides=(1, stride, stride, stride, 1), padding=padding))

        elif params=='fixed': # standard Gaussian dropout with fixed parameters.
            alpha = (1.-keep_prob) / keep_prob
            std = tf.sqrt(tf.nn.conv3d(tf.square(input_batch), alpha * tf.square(w), strides=(1, stride, stride, stride, 1), padding=padding))
            kl = 0

        # compute:
        a_sample = mean + std * tf.random_normal(tf.shape(mean))
        variable_summaries(a_sample, summary)

    return a_sample, kl

def deconv3d(input_batch, out_channels, filter_size=6, stride=2,
             name="deconv", padding='VALID',with_w=False):

    with tf.variable_scope(name):
        in_channels = int(input_batch.get_shape()[-1])
        w = get_weights([filter_size, filter_size, filter_size, out_channels, in_channels])

        filter_shape=[filter_size, filter_size, filter_size]
        strides=(1,stride,stride,stride,1)
        output_shape=get_output_shape_3d(input_batch, filter_shape, strides[1:-1], out_channels, padding)
        print(output_shape)
        deconv = tf.nn.conv3d_transpose(input_batch, w, output_shape=output_shape, strides=strides, padding=padding)
        biases = tf.get_variable('biases', [out_channels], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def conv_dc_3d(input_old, phase, bn_on, out_channels, filter_size=3,
               stride=1, name='',
               summary=True, padding='VALID'):
    """
    Densely connected layer:
    BN => ReLu => Convolution => Concatenate the output feature map with the
    input feature map.
    Note: input feature maps are cropped before concatenation.
    """
    with tf.variable_scope(name):
        input_new = batchnorm(input_old, phase, on=bn_on, name='BN1')
        input_new = tf.nn.relu(input_new)
        input_new = conv3d(input_new, out_channels, filter_size, stride,
                           name='conv1', padding=padding)
        input_old = crop_and_or_concat_basic(input_old, input_new, name='concat1')
        variable_summaries(input_old, summary)
    return input_old

def get_output_shape_3d(input_batch, filter_shape, strides, out_channels, padding='VALID'):
    """ Get the output shape of 3D de-convolution.
    Here it is assumed that the kernel size is divisible by stride in
    all dimensions.

    e.g. x is tf-tensor of shape (None, 9,9,9,50)
         filter_shape=[2*3,2*3,2*3]
         strides = [2,2,2]
         upsampling_rate=2

    if padding='VALID", then output_shape = [
    """

    shape = get_tensor_shape(input_batch)[1:-1]
    filter_shape_lr = [i/strides[idx] for idx, i in enumerate(filter_shape)
                       if i%strides[idx]==0]
    # print(shape, strides, filter_shape_lr)
    assert len(shape)==len(strides)
    assert len(filter_shape_lr)==len(strides)

    output_shape=[]
    for i in range(3):
        if padding=='VALID':
            output_shape.append(strides[i]*(shape[i]-filter_shape_lr[i]+1))
        elif padding=='SAME':
            output_shape.append(strides[i]*shape[i])
        else:
            raise("the specified padding is available")
    output_shape= [get_tensor_shape(input_batch)[0]] + output_shape + [out_channels]
    return output_shape


def variable_summaries(var, default=False):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    if default:
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

def get_tensor_shape(tensor):
    """Return the shape of a tensor as a tuple"""
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])


def normal_mult_noise(a, keep_prob, params, name, summary=True):
    """Gaussian dropout, Srivastava 2014 JMLR"""
    with tf.name_scope(name):
        if params==None:
            sigma = (1.-keep_prob) / keep_prob
            a_drop = a * (1. + sigma * tf.random_normal(tf.shape(a)))
            kl = None
        elif params=='weight':
            # W_init = tf.constant(1e-4, shape=tf.shape(a)[1:])
            W_init = tf.constant(np.float32(1e-4*np.ones(get_tensor_shape(a)[1:])))
            rho = get_weights(filter_shape=None, W_init=W_init, name='rho')
            sigma = tf.minimum(tf.nn.softplus(rho), 1., name='std')
            a_drop = tf.mul(a, 1. + sigma * tf.random_normal(tf.shape(a)), name='a_drop')
            kl = kl_log_uniform_prior(sigma, name='kl')
            variable_summaries(sigma, summary)
            variable_summaries(a_drop, summary)
            # variable_summaries(kl, summary)
        elif params=='channel':
            # W_init = tf.constant(1e-4, shape=tf.shape(a)[1:])
            W_init = tf.constant(np.float32(1e-4 * np.ones(get_tensor_shape(a)[4])))
            rho = get_weights(filter_shape=None, W_init=W_init, name='rho')
            sigma = tf.minimum(tf.nn.softplus(rho), 1., name='std')
            a_drop = tf.mul(a, 1. + sigma * tf.random_normal(tf.shape(a)), name='a_drop')
            # kl = kl_log_uniform_prior(sigma, name='kl')
            kl = np.prod(get_tensor_shape(a)[1:4]) * kl_log_uniform_prior(sigma, name='kl')
            variable_summaries(a_drop, summary)
            variable_summaries(sigma, summary)
            variable_summaries(kl, summary)
        elif params=='layer':
            rho = get_weights(filter_shape=None, W_init=tf.constant(1e-4), name='rho')
            sigma = tf.minimum(tf.nn.softplus(rho), 1., name='std')
            a_drop = tf.mul(a, 1. + sigma * tf.random_normal(tf.shape(a)), name='a_drop')
            # kl = kl_log_uniform_prior(sigma, name='kl')
            kl = np.prod(get_tensor_shape(a)[1:]) * kl_log_uniform_prior(sigma, name='kl')
            variable_summaries(a_drop, summary)
            variable_summaries(kl, summary)
        elif params=='weight_average':  # use average KL across the weights instead.
            # W_init = tf.constant(1e-4, shape=tf.shape(a)[1:])
            W_init = tf.constant(np.float32(1e-4 * np.ones(get_tensor_shape(a)[1:])))
            rho = get_weights(filter_shape=None, W_init=W_init, name='rho')
            sigma = tf.minimum(tf.nn.softplus(rho), 1., name='std')
            a_drop = tf.mul(a, 1. + sigma * tf.random_normal(tf.shape(a)), name='a_drop')
            kl = kl_log_uniform_prior(sigma, name='kl_mean', average=True)
            variable_summaries(a_drop, summary)
            # variable_summaries(kl, summary)
        elif params=='no_noise': # do nothing
            a_drop = a
            kl = None
    return a_drop, kl


def kl_log_uniform_prior(varQ, name=None, average=False):
    """Compute the gaussian-log uniform KL-div from VDLRT"""
    c1 = 1.16145124
    c2 = -1.50204118
    c3 = 0.58629921
    kl_mtx = 0.5*tf.log(varQ) + c1*varQ + c2*tf.pow(varQ,2) + c3*tf.pow(varQ,3)
    if average:
        kl_div = tf.reduce_mean(kl_mtx, name=name)
    else:
        kl_div = tf.reduce_sum(kl_mtx, name=name)
    return kl_div


def residual_block(x, n_in, n_out, name):
    """A residual block of constant spatial dimensions and 1x1 convs only"""
    b = tf.get_variable(name+'_2b', dtype=tf.float32, shape=[n_out],
                        initializer=tf.constant_initializer(1e-2))
    assert n_out >= n_in

    h1 = conv3d(x, [1,1,1,n_in,n_out], [n_out], name+'1')
    h2 = conv3d(tf.nn.relu(h1), [1,1,1,n_out,n_out], None, name+'2')
    h3 = tf.pad(x, [[0,0],[0,0],[0,0],[0,0],[0,n_out-n_in]]) + h2
    return tf.nn.relu(tf.nn.bias_add(h3, b))



###############################################################
# ------------------------ New stuff --------------------------
###############################################################
def batchnorm(x, phase_train, on=True, name=None):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    if not(on): return x

    n_out=int(x.get_shape()[-1])

    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2,3], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3, name=name)
    return normed


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
         return tf.contrib.layers.batch_norm(x,
                                             decay=self.momentum,
                                             updates_collections=None,
                                             epsilon=self.epsilon,
                                             scale=True,
                                             is_training=train,
                                             scope=self.name)


def binary_cross_entropy(preds, targets, name=None):
    """Computes binary cross entropy given `preds`.

    For brevity, let `x = `, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        preds: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `preds`.
    """
    eps = 1e-12
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds + eps) +
                              (1. - targets) * tf.log(1. - preds + eps)))

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def max_pool(x,n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')



def pad_and_concat(x1, x2):
    """ Unfinished
    get the union of two tensors, pad and then concatenate
    """
    # get the shapes as tuples:
    x1_shape = get_tensor_shape(x1)
    x2_shape = get_tensor_shape(x2)

    # if the two tensors have equal width, height, depth
    if x1_shape[1:-1] == x2_shape[1:-1]:
        return tf.concat([x1, x2], 4)

    # ------ Otherwise pad the union ---------
    # get the size of the common tensor:
    size = [max(x1_shape[i], x2_shape[i]) for i in range(1,4)]
    padding_1 = [[0,0],
                 [1,2],
                 [3,4],
                 [1,3],
                 [0,0]]



def crop_and_concat(x1,x2):
    """ get the intersection between x_1, x_2 and then crop and concatenate
    """
    # if the two tensors have equal width, height, depth
    # if x1.shape[1:-1]==x2.shape[1:-1]:
    # return tf.concat([x1, x2], 4)

    # Otherwise get the intersection:
    x1_shape = get_tensor_shape(x1)
    x2_shape = get_tensor_shape(x2)

    # get the size of the common tensor:
    size = [-1]+[min(x1_shape[i], x2_shape[i]) for i in range(1,4)]+[-1]
    print(size)

    # get where x1 is larger than x2:
    offsets_1 = [0,
                max((x1_shape[1] - x2_shape[1]) // 2, 0),
                max((x1_shape[2] - x2_shape[2]) // 2, 0),
                max((x1_shape[3] - x2_shape[3]) // 2, 0),
                0]
    x1_crop = tf.slice(x1, offsets_1, size)

    # get where x2 is larger than x1:
    offsets_2 = [0,
                 max((x2_shape[1] - x1_shape[1]) // 2, 0),
                 max((x2_shape[2] - x1_shape[2]) // 2, 0),
                 max((x2_shape[3] - x1_shape[3]) // 2, 0),
                 0]
    x2_crop = tf.slice(x2, offsets_2, size)
    return tf.concat([x1_crop, x2_crop], 4)


def crop_and_or_concat_basic(x1,x2,is_concat=True,name=''):
    x1_shape = get_tensor_shape(x1)
    x2_shape = get_tensor_shape(x2)

    # offsets for the top left corner of the crop
    offsets = [0,
               (x1_shape[1] - x2_shape[1]) // 2,
               (x1_shape[2] - x2_shape[2]) // 2,
               (x1_shape[3] - x2_shape[3]) // 2,
               0]
    size = [-1, x2_shape[1], x2_shape[2], x2_shape[3], -1]
    x1_crop = tf.slice(x1, offsets, size, name=name)

    # For segnet, just return cropped feature maps:
    if not(is_concat):
        return x1_crop

    # For unet, skip-connections:
    try:
        y = tf.concat(4, [x1_crop, x2], name=name)
    except TypeError: # Support for verisons of TensorFlow after 1.1
        y = tf.concat([x1_crop, x2], 4, name=name)

    return y