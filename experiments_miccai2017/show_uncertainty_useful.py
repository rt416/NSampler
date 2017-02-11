import tensorflow as tf
import configuration
import os
import analysis_miccai2017
import sr_analysis
from train import name_network
import matplotlib.pyplot as plt

# Options
opt = configuration.set_default()
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

# Experiment (local)
# base_input_dir = '/Users/ryutarotanno/DeepLearning/nsampler/data/'
# base_gt_dir = '/Users/ryutarotanno/DeepLearning/nsampler/data/'
base_input_dir = '/Users/ryutarotanno/DeepLearning/nsampler/recon/miccai2017/'
network = True

if network:
    nn_name = name_network(opt)
else:
    nn_name=''

non_HCP = {'hcp': {'subdir':'HCP/904044',
                   'dt_file':'dt_recon_',
                   'std_file': 'dt_std_data_'},
           'prisma':{'subdir':'Prisma/Diffusion_2.5mm',
                     'dt_file':'dt_all_'},
           'tumour':{'subdir':'Tumour/06_FORI',
                     'dt_file':'dt_b700_'},
           'ms':{'subdir':'MS/B0410637-2010-00411',
                 'dt_file':'dt_b1200_lowres2_'}
            }

dataset_type = 'hcp'


# # -------------- plot std vs rmse ------------------------------------------ :
#
# std_file = os.path.join(base_input_dir,
#                         non_HCP[dataset_type]['subdir'],
#                         nn_name,
#                         'dt_recon_MD_std_analytical.nii')
# err_file = os.path.join(base_input_dir,
#                         non_HCP[dataset_type]['subdir'],
#                         nn_name,
#                         'error_dt_recon_MD_dir.nii')
# mask_file =os.path.join(base_input_dir,
#                         non_HCP[dataset_type]['subdir'],
#                         'masks',
#                         'mask_us=2_rec=5.nii')
# print('plotting error vs uncertainty:')
# sr_analysis.plot_twonii(std_file, err_file,
#                         mask_file=mask_file,
#                         no_points=10000,
#                         xlabel='std',
#                         ylabel='rmse',
#                         title='Mean Diffusivity: cnn hetero + variational, ')

# -------------- plot ROC curve ------------------------------------------ :
nii_gt = os.path.join(base_input_dir,
                       non_HCP[dataset_type]['subdir'],
                       'maps',
                       'dt_b1000_MD.nii')

nii_est = os.path.join(base_input_dir,
                       non_HCP[dataset_type]['subdir'],
                       nn_name,
                       'dt_recon_MD_dir.nii')

nii_std = os.path.join(base_input_dir,
                        non_HCP[dataset_type]['subdir'],
                        nn_name,
                        'dt_recon_MD_std_analytical.nii')


mask_file = os.path.join(base_input_dir,
                         non_HCP[dataset_type]['subdir'],
                         'masks',
                         'mask_us=2_rec=5.nii')
sr_analysis.plot_ROC_twonii(nii_gt, nii_est, nii_std,
                            mask_file=mask_file, no_points=100000, acceptable_err=0.00015)

plt.title('ROC')
plt.show()




# analysis_miccai2017._MD_FA(dti_file, std_file, no_samples=1000)
# analysis_miccai2017._MD_FA(dti_file,no_samples=1)
