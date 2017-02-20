""" Perform reconstrution on non- HCP dataset (Prisma, MS, Tumour).
It assumes that DTI is available as nifti files."""

import tensorflow as tf
import configuration
import os
import analysis_miccai2017

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
        opt['method'] = 'cnn_heteroscedastic'
        opt['valid'] = False
        opt['dropout_rate'] = 0.0
        name = opt['method']
        opt['save_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/comparison_v1/models'
        opt['mc_no_samples'] = 1
    elif idx == 2:
        opt['method'] = 'cnn_variational_dropout'
        opt['valid'] = False
        opt['dropout_rate'] = 0.0
        name = opt['method']
        opt['save_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/comparison_v1/models'
        opt['mc_no_samples'] = 200
    elif idx == 3:
        opt['method'] = 'cnn_heteroscedastic_variational_hybrid_control'
        opt['valid'] = False
        opt['dropout_rate'] = 0.0
        name = opt['method']
        opt['save_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/comparison_v2/models'
        opt['mc_no_samples'] = 200

    elif idx == 4:
        opt['method'] = 'cnn_heteroscedastic_variational_channelwise_hybrid_control'
        opt['valid'] = False
        opt['dropout_rate'] = 0.0
        name = opt['method']
        opt['save_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/comparison_v2/models'
        opt['mc_no_samples'] = 200

    elif idx == 5:
        opt['method'] = 'cnn_simple'
        opt['valid'] = False
        opt['dropout_rate'] = 0.0
        name = opt['method']
        opt['save_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/comparison_v1/models'
    elif idx == 6:
        opt['method'] = 'cnn_dropout'
        opt['valid'] = False
        opt['dropout_rate'] = 0.1
        name = opt['method'] + '_0.1'
        opt['save_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/comparison_v1/models'
        opt['mc_no_samples'] = 200
    elif idx == 7:
        opt['method'] = 'cnn_gaussian_dropout'
        opt['dropout_rate'] = 0.1
        opt['valid'] = False
        name = opt['method'] + '_0.1'
        opt['save_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/comparison_v1/models'
        opt['mc_no_samples'] = 200
    elif idx == 8:
        opt['method'] = 'cnn_variational_dropout_channelwise'
        opt['valid'] = False
        opt['dropout_rate'] = 0.0
        name = opt['method']
        opt['save_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/comparison_v1/models'
        opt['mc_no_samples'] = 200
    else:
        raise ValueError('no network for the given idx.')

    return name, opt

# base directories:
base_input_dir = '/SAN/vision/hcp/DCA_HCP.2013.3_Proc/'
base_recon_dir = '/SAN/vision/hcp/Ryu/miccai2017/Life/'
subpath = 'Diffusion/Diffusion/'
subjects_list = ['LS5007', 'LS5040', 'LS5049', 'LS6006', 'LS6038',
                 'LS5038', 'LS5041', 'LS6003', 'LS6009', 'LS6046']

models_list = range(8,9)

for model_idx in models_list:
    for subject in subjects_list:
        print('Reconstructing subject %s with model %i' % (subject, model_idx))
        opt['gt_dir'] = os.path.join(base_input_dir, subject, subpath)
        opt['input_file_name'] = 'dt_b1000_lowres_2_'
        opt['recon_dir'] = os.path.join(base_recon_dir, subject)
        name, opt = models_update(model_idx, opt)
        print('with model: ' + name)

        # clear the graph:
        tf.reset_default_graph()
        analysis_miccai2017.nonhcp_reconstruct(opt, dataset_type='life')




