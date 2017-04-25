"""Ryu: main experiments script"""

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

# Options
opt = {}

# Network:
opt['method'] = 'cnn_simple'
opt['valid'] = False  # validation metric
opt['n_h1'] = 50
opt['n_h2'] = 2*opt['n_h1']
opt['n_h3'] = 10

# Training
opt['overwrite'] = True  # restart the training completely.
opt['continue'] = False  # set True if you want to continue training from the previous experiment
if opt['continue']: opt['overwrite'] = False

opt['optimizer'] = tf.train.AdamOptimizer
opt['dropout_rate'] = 0.0
opt['learning_rate'] = 1e-3
opt['L1_reg'] = 0.00
opt['L2_reg'] = 1e-5

opt['train_size'] = 17431 # 9000  # 100  # total number of patch pairs (train + valid set)
opt['n_epochs'] = 200
opt['batch_size'] = 12
opt['validation_fraction'] = 0.5
opt['patch_sampling_opt']='separate'  # by default, train and valid sets are separated.
opt['shuffle'] = True


# Data (new):
opt['background_value'] = 0  # background value in the images
#opt['train_subjects']=['117324', '904044']
opt['train_subjects'] = ['992774', '125525', '205119', '133928',
                         '570243', '448347', '654754', '153025']
opt['test_subjects'] = ['904044', '165840', '889579', '713239',
                        '899885', '117324', '214423', '857263']


# Data/task:
opt['cohort'] ='Diverse'
opt['no_subjects'] = 8
opt['b_value'] = 1000
opt['patchlib_idx'] = 1
opt['no_randomisation'] = 1
opt['shuffle_data'] = True
opt['chunks'] = True  # set True if you want to chunk the HDF5 file.

opt['subsampling_rate'] = 343
opt['upsampling_rate'] = 2
opt['input_radius'] = 5
opt['receptive_field_radius'] = 2
output_radius = ((2*opt['input_radius']-2*opt['receptive_field_radius']+1)//2)
opt['output_radius'] = output_radius
opt['no_channels'] = 6
opt['transform_opt'] = '0'  #'standard'  # preprocessing of input/output variables

# # Local dir:
# opt['data_dir'] = '/Users/ryutarotanno/tmp/iqt_DL/auro/TrainingData/'
# opt['save_dir'] = '/Users/ryutarotanno/tmp/iqt_DL/auro/TrainingData/'
# opt['log_dir'] = '/Users/ryutarotanno/tmp/iqt_DL/auro/log/'
# opt['save_train_dir'] = '/Users/ryutarotanno/tmp/iqt_DL/auro/TrainingData/'
#
# opt['gt_dir'] = '/Users/ryutarotanno/DeepLearning/nsampler/data/HCP/'  # ground truth dir
# opt['subpath'] = ''
#
# opt['input_file_name'] = 'dt_b1000_lowres_' + str(opt['upsampling_rate']) + '_'


# cluster directories:
base_dir = '/SAN/vision/hcp/Ryu/miccai2017/25Apr2017/'
opt['data_dir'] = base_dir + 'data/'
opt['save_dir'] = base_dir + 'models/'
opt['log_dir'] = base_dir + 'log/'
opt['recon_dir'] = base_dir + 'recon/'

opt['mask_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/recon/'
opt['gt_dir'] = '/SAN/vision/hcp/DCA_HCP.2013.3_Proc/'  # ground truth dir
opt['subpath'] = '/T1w/Diffusion/'

opt['input_file_name'] = 'dt_b1000_lowres_' + str(opt['upsampling_rate']) + '_'

# Choose the experiment option:
# choose = input("Press 1 for training or 2 or 3 for normal/MC-based reconstruction ")

# Train:
from largesc.train_v2 import train_cnn
tf.reset_default_graph()
train_cnn(opt)

# Reconstruct:
subjects_list = ['904044', '165840', '889579', '713239',
                 '899885', '117324', '214423', '857263']
rmse_average = 0
import largesc.reconstruct_v2 as reconstruct
for subject in subjects_list:
    opt['subject'] = subject
    rmse, _ = reconstruct.sr_reconstruct(opt)
    rmse_average += rmse
print('\n Average RMSE on Diverse dataset is %.15f.'
      % (rmse_average / len(subjects_list),))

    # if choose_rec==1:
    #     import reconstruct
    #     for subject in subjects_list:
    #         opt['subject'] = subject
    #         rmse, _ = reconstruct.sr_reconstruct(opt)
    #         rmse_average += rmse
    #
    #     print('\n Average RMSE on Diverse dataset is %.15f.'
    #           % (rmse_average / len(subjects_list),))
    #
    # elif choose_rec==2:
    #     if opt['method'] == 'cnn_heteroscedastic':
    #         opt['mc_no_samples'] = 1
    #     else:
    #         opt['mc_no_samples'] = 100  # input("number of MC samples: ")
    #
    #     import reconstruct_mcdropout
    #     rmse_noedge = 0
    #     rmse_whole = 0
    #
    #     for subject in subjects_list:
    #         opt['subject'] = subject
    #         rmse, rmse2 = reconstruct_mcdropout.sr_reconstruct_mcdropout(opt)
    #         rmse_noedge += rmse
    #         rmse_whole +=rmse2
    #
    #     print('\n Average RMSE (no edge): %.15f.'
    #           % (rmse_noedge / len(subjects_list),))
    #     print('\n Average RMSE (whole): %.15f.'
#     #           % (rmse_whole / len(subjects_list),))
# elif choose==2:
#     import reconstruct
#     # tf.reset_default_graph()
#
#     subjects_list = ['904044', '165840', '889579', '713239',
#                      '899885', '117324', '214423', '857263']
#     rmse_average = 0
#
#     for subject in subjects_list:
#         opt['subject'] = subject
#         rmse, _ = reconstruct.sr_reconstruct(opt)
#         rmse_average += rmse
#
#     print('\n Average RMSE on Diverse dataset is %.15f'
#           % (rmse_average / len(subjects_list),))
# elif choose==3:
#     opt['mc_no_samples'] = input("number of MC samples: ")
#     import reconstruct_mcdropout
#
#     # subjects_list = ['904044']
#
#     subjects_list = ['904044', '165840', '889579', '713239',
#                      '899885', '117324', '214423', '857263']
#
#     rmse_average = 0
#     for subject in subjects_list:
#         opt['subject'] = subject
#         rmse, _ = reconstruct_mcdropout.sr_reconstruct_mcdropout(opt)
#         rmse_average += rmse
#
#     print('\n Average RMSE on Diverse dataset is %.15f.'
#           % (rmse_average / len(subjects_list),))
