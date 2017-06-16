"""Ryu: main experiments script"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Options
opt = {}

# Network:
opt['method'] = 'cnn_simple'
opt['valid'] = False
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
opt['transform_opt'] = 'scaling' #'standard'  # preprocessing of input/output variables

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


# Choose the experiment option:
choose = input("Press 1 for training or 2 or 3 for normal/MC-based reconstruction ")

if choose == 1:

    choose_2 = input("Press 1 for standard or 2 for scalable training: ")
    choose_rec = input("1 for standard reconstruction, 2 for MC reconstruction ")

    if choose_2==1:
        from train import train_cnn
    elif choose_2==2:
        from train_largesc import train_cnn

    # Train:	
    for idx in range(1,2):
        tf.reset_default_graph()
        opt['patchlib_idx'] = idx
        train_cnn(opt)

        # Reconstruct (optional):
        subjects_list = ['904044', '165840', '889579', '713239',
                         '899885', '117324', '214423', '857263']
        rmse_average = 0
    
        if choose_rec==1:
            import reconstruct
            for subject in subjects_list:
                opt['subject'] = subject
                rmse, _ = reconstruct.sr_reconstruct(opt)
                rmse_average += rmse

            print('\n Average RMSE on Diverse dataset is %.15f.'
                  % (rmse_average / len(subjects_list),))

        elif choose_rec==2:
            if opt['method'] == 'cnn_heteroscedastic':
                opt['mc_no_samples'] = 1
            else:
                opt['mc_no_samples'] = 100  # input("number of MC samples: ")

            import reconstruct_mcdropout
            rmse_noedge = 0
            rmse_whole = 0

            for subject in subjects_list:
                opt['subject'] = subject
                rmse, rmse2 = reconstruct_mcdropout.sr_reconstruct_mcdropout(opt)
                rmse_noedge += rmse
                rmse_whole +=rmse2

            print('\n Average RMSE (no edge): %.15f.'
                  % (rmse_noedge / len(subjects_list),))
            print('\n Average RMSE (whole): %.15f.'
                  % (rmse_whole / len(subjects_list),))
elif choose==2:
    import reconstruct
    # tf.reset_default_graph()
    
    subjects_list = ['904044', '165840', '889579', '713239',
                     '899885', '117324', '214423', '857263']
    rmse_average = 0

    for subject in subjects_list:
        opt['subject'] = subject
        rmse, _ = reconstruct.sr_reconstruct(opt)
        rmse_average += rmse

    print('\n Average RMSE on Diverse dataset is %.15f'
          % (rmse_average / len(subjects_list),))
elif choose==3:
    opt['mc_no_samples'] = input("number of MC samples: ")
    import reconstruct_mcdropout

    # subjects_list = ['904044']

    subjects_list = ['904044', '165840', '889579', '713239',
                     '899885', '117324', '214423', '857263']

    rmse_average = 0
    for subject in subjects_list:
        opt['subject'] = subject
        rmse, _ = reconstruct_mcdropout.sr_reconstruct_mcdropout(opt)
        rmse_average += rmse

    print('\n Average RMSE on Diverse dataset is %.15f.'
          % (rmse_average / len(subjects_list),))
