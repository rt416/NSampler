""" Main script for running statistical analysis for MICCAI 2017 """
import tensorflow as tf
import numpy as np
import cPickle as pkl
import os

# Default options
opt = {}

# Network:
opt['method'] = 'cnn_heteroscedastic'
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
opt['patchlib_idx'] = 1
opt['b_value'] = 1000
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
opt['transform_opt'] = 'standard'  # preprocessing of input/output variables

# Dir:
opt['data_dir'] = '/SAN/vision/hcp/Ryu/IPMI2016/TrainingSet/' # '../data/'
opt['save_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/comparison/models'
opt['log_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/comparison/log'
opt['recon_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/comparison/recon'
opt['mask_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/recon'

opt['save_train_dir_tmp'] = '/SAN/vision/hcp/Ryu/IPMI2016/HCP'
opt['save_train_dir'] = '/SAN/vision/hcp/Ryu/IPMI2016/TrainingSet/'

opt['gt_dir'] = '/SAN/vision/hcp/DCA_HCP.2013.3_Proc/'  # ground truth dir
opt['subpath'] = 'T1w/Diffusion'

opt['input_file_name'] = 'dt_b1000_lowres_' + str(opt['upsampling_rate']) + '_'


# Define model updates:
def models_update(idx, opt):

    if idx == 1:
        opt['method'] = 'cnn_heteroscedastic'
        opt['dropout_rate'] = 0.0
        name = opt['method']
    elif idx == 2:
        opt['method'] = 'cnn_heteroscedastic_variational'
        opt['dropout_rate'] = 0.0
        name = opt['method']
    elif idx == 3:
        opt['method'] = 'cnn_heteroscedastic_variational_downsc'
        opt['dropout_rate'] = 0.0
        name = opt['method']
    elif idx == 4:
        opt['method'] = 'cnn_heteroscedastic_variational_channelwise'
        opt['dropout_rate'] = 0.0
        name = opt['method']
    elif idx == 5:
        opt['method'] = 'cnn_simple'
        opt['dropout_rate'] = 0.0
        name = opt['method']
    elif idx == 6:
        opt['method'] = 'cnn_dropout'
        opt['dropout_rate'] = 0.1
        name = opt['method'] + '_0.1'
    elif idx == 7:
        opt['method'] = 'cnn_gaussian_dropout'
        opt['dropout_rate'] = 0.1
        name = opt['method'] + '_0.1'
    elif idx == 8:
        opt['method'] = 'cnn_variational_dropout'
        opt['dropout_rate'] = 0.0
        name = opt['method']
    elif idx == 9:
        opt['method'] = 'cnn_variational_dropout_channelwise'
        opt['dropout_rate'] = 0.0
        name = opt['method']
    else:
        raise ValueError('no network for the given idx.')

    return name, opt


# Start the experiment:
import analysis_miccai2017

analysis_dir = '/SAN/vision/hcp/Ryu/miccai2017/comparison/analysis'
experiment_name = '30jan17_compare_models'
subjects_list = ['904044', '165840', '889579', '713239',
                 '899885', '117324', '214423', '857263']
err_compare = dict()

for model_idx in range(1,10):
    err_mtx = np.zeros((len(subjects_list),8,6))
    name, opt=models_update(model_idx,opt)
    print("Compute average errors ...")
    for i,patch_idx in enumerate(range(1,9)):
        opt['patchlib_idx'] = patch_idx
        for j, subject in enumerate(subjects_list):
            opt['subject'] = subject
            err = analysis_miccai2017.compute_err(opt)
            err_mtx[j,i,0] = err['rmse_noedge']
            err_mtx[j,i,1] = err['psnr_noedge']
            err_mtx[j,i,2] = err['mssim_noedge']
            err_mtx[j,i,3] = err['rmse_whole']
            err_mtx[j,i,4] = err['psnr_whole']
            err_mtx[j,i,5]= err['mssim_whole']

    err_compare[name]\
    = {'mean':
           {'rmse_noedge': err_mtx.mean(axis=(0,1))[0],
            'psnr_noedge': err_mtx.mean(axis=(0,1))[1],
            'mssim_noedge': err_mtx.mean(axis=(0,1))[2],
            'rmse_whole': err_mtx.mean(axis=(0,1))[3],
            'psnr_whole': err_mtx.mean(axis=(0,1))[4],
            'mssim_whole': err_mtx.mean(axis=(0,1))[5]
            },
       'std':
           {'rmse_noedge': np.std(err_mtx.mean(axis=0),axis=0)[0],
            'psnr_noedge': np.std(err_mtx.mean(axis=0),axis=0)[1],
            'mssim_noedge': np.std(err_mtx.mean(axis=0),axis=0)[2],
            'rmse_whole': np.std(err_mtx.mean(axis=0),axis=0)[3],
            'psnr_whole': np.std(err_mtx.mean(axis=0),axis=0)[4],
            'mssim_whole': np.std(err_mtx.mean(axis=0),axis=0)[5]
            }
       }

    # Save:
    experiment_file = os.path.join(analysis_dir, experiment_name + '.pkl')
    with open(experiment_file, 'wb') as fp:
        if not os.path.exists(analysis_dir):
            os.makedirs((analysis_dir))
        pkl.dump(err_compare, fp, protocol=pkl.HIGHEST_PROTOCOL)
    print('Errors details saved as %s' % (experiment_file,))




