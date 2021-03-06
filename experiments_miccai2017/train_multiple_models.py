"""Ryu: main experiments script"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Options
opt = {}

# Network:
opt['method'] = 'cnn_heteroscedastic'
opt['valid'] = True
opt['n_h1'] = 50
opt['n_h2'] = 2*opt['n_h1']
opt['n_h3'] = 10

# Training
opt['optimizer'] = tf.train.AdamOptimizer
opt['dropout_rate'] = 0.0
opt['learning_rate'] = 1e-3
opt['L1_reg'] = 0.00
opt['L2_reg'] = 1e-5
opt['n_epochs'] = 200
opt['batch_size'] = 12
opt['validation_fraction'] = 0.5
opt['shuffle'] = True

# Data/task:
opt['cohort'] ='Diverse'
opt['no_subjects'] = 8
opt['b_value'] = 1000
opt['patchlib_idx'] = 1
opt['no_randomisation'] = 1
opt['shuffle_data'] = True
opt['chunks'] = True  # set True if you want to chunk the HDF5 file.

opt['subsampling_rate'] = input("Enter subsampling rate: ")  # 343
opt['upsampling_rate'] = input("Enter upsampling rate: ")  # 2
opt['input_radius'] = input("Enter input radius: ")  # 5
opt['receptive_field_radius'] = input("Enter receptive field radius: ")  # 2
output_radius = ((2*opt['input_radius']-2*opt['receptive_field_radius']+1)//2)
opt['output_radius'] = output_radius
opt['no_channels'] = 6
opt['transform_opt'] = 'standard'  # preprocessing of input/output variables

# Dir:
opt['data_dir'] = '/SAN/vision/hcp/Ryu/IPMI2016/TrainingSet/' # '../data/'
opt['save_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/comparison_v2/models'
opt['log_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/comparison_v2/log'
opt['recon_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/comparison_v2/recon'
opt['mask_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/recon'

opt['save_train_dir_tmp'] = '/SAN/vision/hcp/Ryu/IPMI2016/HCP'
opt['save_train_dir'] = '/SAN/vision/hcp/Ryu/IPMI2016/TrainingSet/'

opt['gt_dir'] = '/SAN/vision/hcp/DCA_HCP.2013.3_Proc/'  # ground truth dir
opt['subpath'] = 'T1w/Diffusion'

opt['input_file_name'] = 'dt_b1000_lowres_' + str(opt['upsampling_rate']) + '_'


def models_update(idx, opt):
    if idx == 1:
        opt['method'] = 'cnn_simple'
        opt['valid'] = False
        opt['dropout_rate'] = 0.0
        name = opt['method']
    elif idx == 2:
        opt['method'] = 'cnn_variational_dropout'
        opt['valid'] = False
        opt['dropout_rate'] = 0.0
        name = opt['method']
        opt['mc_no_samples'] = 100
    elif idx == 3:
        opt['method'] = 'cnn_heteroscedastic_variational_hybrid_control'
        opt['valid'] = False
        opt['dropout_rate'] = 0.0
        name = opt['method']
        opt['mc_no_samples'] = 100
    elif idx == 4:
        opt['method'] = 'cnn_heteroscedastic_variational_channelwise_hybrid_control'
        opt['valid'] = False
        opt['dropout_rate'] = 0.0
        name = opt['method']
        opt['mc_no_samples'] = 100
    else:
        raise ValueError('no network for the given idx.')

    return name, opt


# Choose the experiment option:
from train import train_cnn

# Train:
for model_idx in range(1,5):
    tf.reset_default_graph()
    opt['patchlib_idx'] = 1
    opt['save_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/08feb2017/models'
    opt['log_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/08feb2017/log'
    name, opt = models_update(model_idx, opt)
    train_cnn(opt)


