# Utilities script for common functionality

import os
import sys
import time

import numpy as np
import tensorflow as tf


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
    optim = opt['optimizer']

    nn_header = opt['method'] if opt['dropout_rate']==0 \
    else opt['method'] + str(opt['dropout_rate'])

    opt['receptive_field_radius']=(2*opt['input_radius']-2*opt['output_radius'] + 1)//2

    nn_var = (opt['upsampling_rate'],
              opt['no_layers'],
              2*opt['input_radius']+1,
              2*opt['receptive_field_radius']+1,
             (2*opt['output_radius']+1)*opt['upsampling_rate'],
              opt['is_BN'])
    nn_str = 'us=%i_lyr=%i_in=%i_rec=%i_out=%i_bn=%i_'

    nn_var += (opt['no_subjects'],
               opt['no_patches'],
               opt['pad_size'],
               opt['is_clip'],
               opt['transform_opt'],
               opt['patch_sampling_opt'],
               opt['patchlib_idx'])
    nn_str += 'ts=%d_pl=%d_pad=%d_clip=%i_nrm=%s_smpl=%s_%03i'
    nn_body = nn_str % nn_var

    if opt['is_map']:
        nn_header = 'MAP_' + nn_header

    if opt['valid']:
        # Validate on the cost:
        nn_header += '_validcost'

    return nn_header + '_' + nn_body
