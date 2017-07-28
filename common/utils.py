# Utilities script for common functionality

import os
import sys
import time
sys.path.append("../2_ESPCN")

import cPickle as pkl
import numpy as np
import tensorflow as tf

# FIXME: this is horrid
import models

### NAMING
def define_checkpoint(opt):
    """Create checkpoint directory.
    
    Generate checkpoint name and if directory does not exist, then instantiate
    one.

    Args:
       opt: dict of options, must include save_dir field

    Returns:
        name of checkpoint directory
    """
    nn_file = name_network(opt)
    checkpoint_dir = os.path.join(opt['save_dir'], nn_file)
    if not os.path.exists(checkpoint_dir):
        print(checkpoint_dir)
        os.makedirs(checkpoint_dir)
    return checkpoint_dir


def name_network(opt):
    """Return the model name.

    Return a model name containing...

    Args:
        opt: dict of options

    Returns:
        model name string
    """
    # HEADER:
    nn_header = opt['method'] if opt['dropout_rate'] == 0 else \
                opt['method'] + str(opt['dropout_rate'])

    if opt['is_map']:
        nn_header = 'MAP_' + nn_header

    if opt['hetero']:
        nn_header += '_hetero'

    if opt['vardrop']:
        nn_header += '_vardrop_' + opt['params']+'wise'

    if opt['hybrid_on']:
        nn_header += '_hybrid'

    if opt['cov_on']:
        nn_header += '_cov'

    if opt['valid']:  # Validate on the cost:
        nn_header += '_validcost'

    # BODY
    opt['receptive_field_radius']=(2*opt['input_radius']-2*opt['output_radius'] + 1)//2

    nn_var = (opt['upsampling_rate'],
              opt['no_layers'],
              opt['no_filters'],
              2*opt['input_radius']+1,
              2*opt['receptive_field_radius']+1,
             (2*opt['output_radius']+1)*opt['upsampling_rate'],
              opt['is_BN'])
    nn_str = 'us=%i_lyr=%i_nf=%i_in=%i_rec=%i_out=%i_bn=%i_'

    nn_var += (opt['no_subjects'],
               opt['no_patches'],
               opt['pad_size'],
               opt['is_clip'],
               opt['transform_opt'],
               opt['patch_sampling_opt'],
               opt['patchlib_idx'])
    nn_str += 'ts=%d_pl=%d_pad=%d_clip=%i_nrm=%s_smpl=%s_%03i'
    nn_body = nn_str % nn_var

    return nn_header + '_' + nn_body


def define_logdir(opt):
    """Create summary directory.
    
    Generate sumary name and if directory does not exist, then instantiate
    one.

    Args:
       opt: dict of options, must include save_dir field

    Returns:
        name of checkpoint directory
    """
    nn_file = name_network(opt)
    log_dir = os.path.join(opt['log_dir'], nn_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def name_patchlib(opt):
    """Return the patchlib name.

    Return a patchlib name containing...

    Args:
        opt: dict of options

    Returns:
        patchlib name string
    """
    header = 'patchlib'
    if opt['is_map']: header = 'MAP_' + header
    opt['receptive_field_radius'] = (2 * opt['input_radius'] - 2 * opt['output_radius'] + 1) // 2

    # problem definition:
    nn_var = (opt['upsampling_rate'],
              2 * opt['input_radius'] + 1,
              2 * opt['receptive_field_radius'] + 1,
              (2 * opt['output_radius'] + 1) * opt['upsampling_rate'],
              opt['pad_size'],
              opt['is_shuffle'])
    nn_str = 'us=%i_in=%i_rec=%i_out=%i_pad=%i_shuffle=%i_'

    nn_var += (opt['no_subjects'],
               opt['no_patches'],
               opt['transform_opt'],
               opt['patch_sampling_opt'],
               opt['patchlib_idx'])
    nn_str += 'ts=%d_pl=%d_nrm=%s_smpl=%s_%03i'
    nn_body = nn_str % nn_var
    return header+'_'+nn_body


### IO
def save_model(opt, sess, saver, global_step, model_details):
    """Save model

    Create a model checkpoint and save to opt["checkpoint_dir"]

    Args:
        opt: dict of configuration options, defined in run_me.py
        sess: current TF session handle
        saver: tf.Saver instance 
        global_step: iteration counter (can be tf variable)
        model_details: extra information to dump
    """
    checkpoint_dir = opt['checkpoint_dir']
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    save_path = saver.save(sess, checkpoint_prefix, global_step=global_step)
    print("Model saved in file: {:s}".format(save_path))
    with open(os.path.join(checkpoint_dir, 'settings.pkl'), 'wb') as fp:
        pkl.dump(model_details, fp, protocol=pkl.HIGHEST_PROTOCOL)
    print('Model details saved')


def set_network_config(opt):
    """ Define the model type"""
    if opt["method"] == "espcn":
        assert opt["is_shuffle"]
        net = models.espcn(upsampling_rate=opt['upsampling_rate'],
                           out_channels=opt['no_channels'],
                           filters_num=opt['no_filters'],
                           layers=opt['no_layers'],
                           bn=opt['is_BN'])

    elif opt["method"] == "dcespcn" :
        assert opt["is_shuffle"]
        net = models.dcespcn(upsampling_rate=opt['upsampling_rate'],
                             out_channels=opt['no_channels'],
                             filters_num=opt['no_filters'],
                             layers=opt['no_layers'],
                             bn=opt['is_BN'])

    elif opt["method"] == "espcn_deconv" :
        assert not(opt["is_shuffle"])
        net = models.espcn_deconv(upsampling_rate=opt['upsampling_rate'],
                                  out_channels=opt['no_channels'],
                                  filters_num=opt['no_filters'],
                                  layers=opt['no_layers'],
                                  bn=opt['is_BN'])
    elif opt["method"] == "segnet":
        assert not (opt["is_shuffle"])
        net = models.unet(upsampling_rate=opt['upsampling_rate'],
                          out_channels=opt['no_channels'],
                          filters_num=opt['no_filters'],
                          layers=opt['no_layers'],
                          conv_num=2,
                          bn=opt['is_BN'],
                          is_concat=False)

    elif opt["method"] == "unet":
        assert not (opt["is_shuffle"])
        net = models.unet(upsampling_rate=opt['upsampling_rate'],
                          out_channels=opt['no_channels'],
                          filters_num=opt['no_filters'],
                          layers=opt['no_layers'],
                          conv_num=2,
                          bn=opt['is_BN'],
                          is_concat=True)
    else:
        raise ValueError("The specified network type %s not available" %
                        (opt["method"],))
    return net


def get_tradeoff_values(hybrid_on, n_epochs=200):
    if hybrid_on:
        print('apply ascending trade-off!')
        tradeoff_list = np.zeros(n_epochs)
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
        tradeoff_list = np.ones(n_epochs)

    return tradeoff_list


# Monte-Carlo inference:
def mc_inference(fn, fn_std, fd, opt, sess):
    """ Compute the mean and std of samples drawn from stochastic function"""
    no_samples = opt['mc_no_samples']
    if opt['hetero']:
        if opt['cov_on']:
            sum_out = 0.0
            sum_out2 = 0.0
            sum_var = 0.0
            for i in xrange(no_samples):
                current, current_std = sess.run([fn, fn_std], feed_dict=fd)
                sum_out += current
                sum_out2 += current ** 2
                sum_var += current_std ** 2
            mean = sum_out / (1. * no_samples)
            std = np.sqrt((np.abs(sum_out2 - 2 * mean * sum_out + no_samples * mean ** 2) + sum_var) / no_samples)
        else:
            sum_out = 0.0
            sum_out2 = 0.0
            for i in xrange(no_samples):
                current = 1. * fn.eval(feed_dict=fd)
                sum_out += current
                sum_out2 += current ** 2
            mean = sum_out / (1. * no_samples)
            std = np.sqrt(np.abs(sum_out2 - 2 * mean * sum_out + no_samples * mean ** 2) / no_samples)
            std += 1. * fn_std.eval(feed_dict=fd)
    else:
        if opt['vardrop']:
            sum_out = 0.0
            sum_out2 = 0.0
            for i in xrange(no_samples):
                current = 1. * fn.eval(feed_dict=fd)
                sum_out += current
                sum_out2 += current ** 2

            mean = sum_out / (1.*no_samples)
            std = np.sqrt(np.abs(sum_out2 - 2*mean*sum_out + no_samples*mean**2)/no_samples)
        else:
            # raise Exception('The specified method does not support MC inference.')
            mean = fn.eval(feed_dict=fd)
            std = 0.0*mean  # zero in every entry
    return mean, std