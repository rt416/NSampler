""" Perform MAP-MRI reconstrution on non- HCP dataset (Prisma, MS, Tumour).
It assumes that DTI is available as nifti files."""

import tensorflow as tf
import configuration
import os
import analysis_miccai2017

# Options
opt = configuration.set_default()

# Training
opt['valid'] = False
opt['dropout_rate'] = 0.0

# Data/task:
opt['patchlib_idx'] = 1
opt['subsampling_rate'] = 343
opt['upsampling_rate'] = 2
opt['input_radius'] = 5
opt['receptive_field_radius'] = 2
output_radius = ((2*opt['input_radius']-2*opt['receptive_field_radius']+1)//2)
opt['output_radius'] = output_radius
opt['no_channels'] = 22
if opt['method'] == 'cnn_heteroscedastic':
    opt['mc_no_samples'] = 1
else:
    opt['mc_no_samples'] = 100

# Dir:
opt['data_dir'] = '/SAN/vision/hcp/Ryu/IPMI2016/MAP_patchlibs/'  # '../data/'
opt['save_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/MAP/models'
opt['log_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/MAP/log'
opt['recon_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/MAP/recon'
opt['mask_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/recon'


# Define model updates:
def models_update(idx, opt):
    if idx == 1:
        opt['method'] = 'cnn_heteroscedastic'
        opt['valid'] = False
        opt['dropout_rate'] = 0.0
        name = opt['method']
        opt['save_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/MAP/models'
        opt['mc_no_samples'] = 1
    elif idx == 2:
        opt['method'] = 'cnn_variational_dropout'
        opt['valid'] = False
        opt['dropout_rate'] = 0.0
        name = opt['method']
        opt['save_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/MAP/models'
        opt['mc_no_samples'] = 200
    elif idx == 3:
        opt['method'] = 'cnn_heteroscedastic_variational_hybrid_control'
        opt['valid'] = False
        opt['dropout_rate'] = 0.0
        name = opt['method']
        opt['save_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/MAP/models'
        opt['mc_no_samples'] = 200

    elif idx == 4:
        opt['method'] = 'cnn_heteroscedastic_variational_channelwise_hybrid_control'
        opt['valid'] = False
        opt['dropout_rate'] = 0.0
        name = opt['method']
        opt['save_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/MAP/models'
        opt['mc_no_samples'] = 200

    elif idx == 5:
        opt['method'] = 'cnn_simple'
        opt['valid'] = True
        opt['dropout_rate'] = 0.0
        name = opt['method']
        opt['save_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/MAP/models'
    elif idx == 6:
        opt['method'] = 'cnn_dropout'
        opt['valid'] = False
        opt['dropout_rate'] = 0.1
        name = opt['method'] + '_0.1'
        opt['save_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/MAP/models'
        opt['mc_no_samples'] = 200
    elif idx == 7:
        opt['method'] = 'cnn_gaussian_dropout'
        opt['valid'] = False
        opt['dropout_rate'] = 0.1
        name = opt['method'] + '_0.1'
        opt['save_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/MAP/models'
        opt['mc_no_samples'] = 200
    else:
        raise ValueError('no network for the given idx.')

    return name, opt

# base directories:
base_input_dir = '/SAN/vision/hcp/Ryu/non-HCP/'
base_recon_dir = '/SAN/vision/hcp/Ryu/non-HCP'

opt['input_file_name'] = 'h4_all_lowres_' + str(opt['upsampling_rate']) + '_'
opt['output_file_name'] = 'h4_recon.npy'
opt['gt_header'] = 'h4_all_'

non_HCP = {'prisma':{'subdir':'Prisma/Diffusion_2.5mm',
                     'dt_file':'dt_all_'},
           'prisma_MAP': {'subdir': 'Prisma/Diffusion_2.5mm',
                      'dt_file': 'h4_all_'},
           'tumour':{'subdir':'Tumour/06_FORI',
                     'dt_file':'dt_b700_'},
           'ms':{'subdir':'MS/B0410637-2010-00411',
                 'dt_file':'dt_test_b1200_'},
           'hcp1':{'subdir':'HCP/117324',
                   'dt_file':'dt_b1000_lowres_2_'},
           'hcp2': {'subdir': 'HCP/904044',
                    'dt_file': 'dt_b1000_lowres_2_'},
           }


model_idx = 4
key='prisma_MAP'

for patch_idx in range(1,9):
    print('Reconstructing: %s' %(non_HCP[key]['subdir'],))
    opt['patchlib_idx'] = patch_idx
    opt['gt_dir'] = os.path.join(base_input_dir, non_HCP[key]['subdir'])
    opt['input_file_name'] = non_HCP[key]['dt_file']
    opt['recon_dir'] = os.path.join(base_recon_dir,non_HCP[key]['subdir'],'MAP')
    name, opt = models_update(model_idx,opt)
    print('with model: ' + name)

    # clear the graph:
    tf.reset_default_graph()
    analysis_miccai2017.nonhcp_reconstruct(opt, dataset_type=key)




