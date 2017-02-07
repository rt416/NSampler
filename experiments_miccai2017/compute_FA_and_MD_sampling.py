""" Perform reconstrution on non- HCP dataset (Prisma, MS, Tumour).
It assumes that DTI is available as nifti files."""

import tensorflow as tf
import configuration
import os
import analysis_miccai2017
import reconstruct_mc_FA_and_MD


# Options
opt = configuration.set_default()

# Training
opt['dropout_rate'] = 0.0

# Data/task:
opt['patchlib_idx'] = 1
opt['subsampling_rate'] = 343
opt['upsampling_rate'] = 2
opt['input_radius'] = 5
opt['receptive_field_radius'] = 2
output_radius = ((2*opt['input_radius']-2*opt['receptive_field_radius']+1)//2)
opt['output_radius'] = output_radius
opt['no_channels'] = 6
if opt['method'] == 'cnn_heteroscedastic':
    opt['mc_no_samples'] = 1
else:
    opt['mc_no_samples'] = 100


# Define model updates:
def models_update(idx, opt):

    if idx == 1:
        opt['method'] = 'cnn_variational_dropout'
        opt['valid'] = False
        opt['dropout_rate'] = 0.0
        name = opt['method']
        opt['save_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/comparison_v1/models'
        opt['mc_no_samples'] = 200
    elif idx == 2:
        opt['method'] = 'cnn_heteroscedastic_variational_hybrid_control'
        opt['valid'] = False
        opt['dropout_rate'] = 0.0
        name = opt['method']
        opt['save_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/comparison_v2/models'
        opt['mc_no_samples'] = 200

    elif idx == 3:
        opt['method'] = 'cnn_heteroscedastic_variational_channelwise_hybrid_control'
        opt['valid'] = False
        opt['dropout_rate'] = 0.0
        name = opt['method']
        opt['save_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/comparison_v2/models'
        opt['mc_no_samples'] = 200

    elif idx == 4:
        opt['method'] = 'cnn_dropout'
        opt['valid'] = False
        opt['dropout_rate'] = 0.1
        name = opt['method'] + '_0.1'
        opt['save_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/comparison_v1/models'
        opt['mc_no_samples'] = 200
    elif idx == 5:
        opt['method'] = 'cnn_gaussian_dropout'
        opt['valid'] = False
        opt['dropout_rate'] = 0.1
        name = opt['method'] + '_0.1'
        opt['save_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/comparison_v1/models'
        opt['mc_no_samples'] = 200
    else:
        raise ValueError('no network for the given idx.')

    return name, opt

models_list = range(1,6)
subjects_list = ['904044', '117324']
base_recon_dir = '/SAN/vision/hcp/Ryu/non-HCP/HCP'

for subject in subjects_list:
    for model_idx in models_list:
        opt['subject'] = subject
        opt['recon_dir'] = opt['recon_dir'] = os.path.join(base_recon_dir, os['subject'])
        name, opt = models_update(model_idx, opt)
        reconstruct_mc_FA_and_MD.sr_reconstruct_FA_and_MD(opt)