""" Perform reconstrution on non- HCP dataset (Prisma, MS, Tumour).
It assumes that DTI is available as nifti files."""

import tensorflow as tf
import configuration
import os
import analysis_miccai2017

# Options
opt = configuration.set_default()

# Update parameters:
opt['method'] = 'cnn_heteroscedastic'
opt['valid'] = False  # pick the best model with the minimal cost (instead of RMSE).

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


# base directories:
base_input_dir = '/Users/ryutarotanno/DeepLearning/nsampler/data/'
base_recon_dir = '/Users/ryutarotanno/DeepLearning/nsampler/recon/non-HCP/'

non_HCP = {'prisma':{'subdir':'Prisma/Diffusion_2.5mm',
                     'dt_file':'dt_all_'},
           'tumour':{'subdir':'Tumour/06_FORI',
                     'dt_file':'dt_b700_'},
           'ms':{'subdir':'MS/B0410637-2010-00411',
                 'dt_file':'dt_b1200_lowres2_'}
            }

# non_HCP = {'ms':{'subdir':'MS/B0410637-2010-00411',
#                 'dt_file':'dt_b1200_'}}

for key in non_HCP:
    print('Reconstructing: %s' %(non_HCP[key]['subdir'],))
    opt['gt_dir'] = os.path.join(base_input_dir,non_HCP[key]['subdir'])
    opt['input_file_name'] = non_HCP[key]['dt_file']
    opt['recon_dir'] = os.path.join(base_recon_dir,non_HCP[key]['subdir'])
    opt['save_dir'] = '/Users/ryutarotanno/tmp/model/'
    # clear the graph:
    tf.reset_default_graph()
    analysis_miccai2017.nonhcp_reconstruct(opt,dataset_type=key)



