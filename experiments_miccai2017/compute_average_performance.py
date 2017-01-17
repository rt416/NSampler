"""Ryu: main experiments script"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Options
opt = {}

# Network:
opt['method'] = 'cnn_simple'
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
opt['validation_fraction'] = 0.5

# Data/task:
opt['cohort'] ='Diverse'
opt['no_subjects'] = 8
opt['b_value'] = 1000
opt['no_randomisation'] = 1
opt['shuffle_data'] = True
opt['chunks'] = True  # set True if you want to chunk the HDF5 file.

opt['subsampling_rate'] = 85
opt['upsampling_rate'] = 2
opt['input_radius'] = 5
opt['receptive_field_radius'] = 2
output_radius = ((2*opt['input_radius']-2*opt['receptive_field_radius']+1)//2)
opt['output_radius'] = output_radius
opt['no_channels'] = 6
opt['transform_opt'] = 'standard'  # preprocessing of input/output variables

# Dir:
opt['data_dir'] = '/SAN/vision/hcp/Ryu/IPMI2016/TrainingSet/' # '../data/'
opt['save_dir'] = '../models'
opt['log_dir'] = '../log'
opt['recon_dir'] = '../recon'

opt['save_train_dir_tmp'] = '/SAN/vision/hcp/Ryu/IPMI2016/HCP'
opt['save_train_dir'] = '/SAN/vision/hcp/Ryu/IPMI2016/TrainingSet/'

opt['gt_dir'] = '/SAN/vision/hcp/DCA_HCP.2013.3_Proc/'  # ground truth dir
opt['subpath'] = 'T1w/Diffusion'

opt['input_file_name'] = 'dt_b1000_lowres_' + str(opt['upsampling_rate']) + '_'

# Train standard networks on training sets of varying size:
from train import train_cnn
import reconstruct

# opt['method'] = 'cnn_simple'
#
# for subsample in [5500, 1372, 343, 85]:
#     # Train:
#     opt['subsampling_rate'] = subsample
#     train_cnn(opt)
#
#     # Reconstruct (optional):
#     subjects_list = ['904044', '165840', '889579', '713239',
#                      '899885', '117324', '214423', '857263']
#     rmse_average = 0
#     for subject in subjects_list:
#         opt['subject'] = subject
#         rmse, _ = reconstruct.sr_reconstruct(opt)
#         rmse_average += rmse
#
#     print('\n Average RMSE on Diverse dataset is %.15f.'
#               % (rmse_average / len(subjects_list),))


# Train dropout network
print('Train dropout networks!')
opt['method'] = 'cnn_dropout'
opt['subsampling_rate'] = 1372

for dropout_rate in [0.0, 0.1, 0.3, 0.5, 0.6]:
    # Train:
    opt['dropout_rate'] = dropout_rate
    train_cnn(opt)

    # Reconstruct (optional):
    subjects_list = ['904044', '165840', '889579', '713239',
                     '899885', '117324', '214423', '857263']
    rmse_average = 0
    for subject in subjects_list:
        opt['subject'] = subject
        rmse, _ = reconstruct.sr_reconstruct(opt)
        rmse_average += rmse

    print('\n Average RMSE on Diverse dataset is %.15f.'
          % (rmse_average / len(subjects_list),))


# MC dropout reconstruct:
# import reconstruct_mcdropout
# print('Train dropout networks!')
# opt['method'] = 'cnn_dropout'
# opt['subsampling_rate'] = 1372
# opt['mc_no_samples'] = 100
#
#
# for dropout_rate in [0.0, 0.1, 0.3, 0.5, 0.6]:
#     # Train:
#     opt['dropout_rate'] = dropout_rate
#     # Reconstruct (optional):
#     subjects_list = ['904044', '165840', '889579', '713239',
#                      '899885', '117324', '214423', '857263']
#     rmse_average = 0
#     for subject in subjects_list:
#         opt['subject'] = subject
#         rmse, _ = reconstruct_mcdropout.sr_reconstruct_mcdropout(opt)
#         rmse_average += rmse
#
#     print('\n Average RMSE on Diverse dataset is %.15f.'
#           % (rmse_average / len(subjects_list),))