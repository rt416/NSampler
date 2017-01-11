"""Ryu: main experiments script"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

choose = input("Press 1 for training or 2 for reconstruction: ")

import reconstruct

# Options
opt = {}

# Network:
opt['method'] = 'cnn_simple'
opt['n_h1'] = 50
opt['n_h2'] = 2 * opt['n_h1']
opt['n_h3'] = 10

# Training
opt['optimizer'] = tf.train.AdamOptimizer
opt['dropout_rate'] = 0.0
opt['learning_rate'] = 1e-3
opt['L1_reg'] = 0.00
opt['L2_reg'] = 1e-5
opt['n_epochs'] = 200
opt['batch_size'] = 25
opt['validation_fraction'] = 0.5
opt['shuffle'] = True
opt['validation_fraction'] = 0.5
opt['shuffle'] = True

# Training data/task:
opt['cohort'] = 'Diverse'
opt['no_subjects'] = 8
opt['subsampling_rate'] = 32
opt['upsampling_rate'] = 2
opt['input_radius'] = 5
opt['receptive_field_radius'] = 2
output_radius = ((2 * opt['input_radius'] -
                  2 * opt['receptive_field_radius'] + 1) // 2)
opt['output_radius'] = output_radius
opt['no_channels'] = 6
opt['transform_opt'] = 'standard'  # preprocessing of input/output variables

# Dir:
opt['data_dir'] = '/SAN/vision/hcp/Ryu/IPMI2016/TrainingSet/'  # '../data/'
opt['save_dir'] = '../models'
opt['recon_dir']= '../recon'

opt['gt_dir'] = '/SAN/vision/hcp/DCA_HCP.2013.3_Proc/'  # ground truth dir
opt['subpath'] = 'T1w/Diffusion'

opt['input_file_name'] = 'dt_b1000_lowres_' + str(opt['upsampling_rate']) + '_'

subjects_list = ['904044', '165840', '889579', '713239',
                 '899885', '117324', '214423', '857263']

rmse_average = 0

for subject in subjects_list:
    opt['subject'] = subject
    rmse, _ = reconstruct.sr_reconstruct(opt)
    rmse_average += rmse

print('\n Average RMSE on Diverse dataset is %f.'
      % (rmse_average/len(subjects_list), ))
