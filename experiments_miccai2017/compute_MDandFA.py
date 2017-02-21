# Compute MD and FA:

import tensorflow as tf
import configuration
import os
import analysis_miccai2017
from train import name_network

# Options
opt = configuration.set_default()
opt['method'] = 'cnn_heteroscedastic_variational_hybrid_control'
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
base_gt_dir = '/Users/ryutarotanno/DeepLearning/nsampler/data/'
network = True
dataset_type = 'tumour'

if network:
    base_input_dir = '/Users/ryutarotanno/DeepLearning/nsampler/recon/miccai2017/'
    nn_name = name_network(opt)
    non_HCP = {'hcp': {'subdir': 'HCP/904044',
                       'dt_file': 'dt_recon_',
                       'std_file': 'dt_std_'},
               'prisma': {'subdir': 'Prisma/Diffusion_2.5mm',
                          'dt_file': 'dt_recon_'},
               'tumour': {'subdir': 'Tumour/06_FORI',
                          'dt_file': 'dt_recon_',
                          'std_file': 'dt_std_data_'},
               'ms': {'subdir': 'MS/B0410637-2010-00411',
                      'dt_file': 'dt_b1200_lowres2_'}
               }
else:
    base_input_dir = '/Users/ryutarotanno/DeepLearning/nsampler/data/'
    nn_name=''
    non_HCP = {'hcp': {'subdir': 'HCP/904044',
                       'dt_file': 'dt_b1000_lowres_2_',
                       'std_file': 'dt_std_'},
               'prisma': {'subdir': 'Prisma/Diffusion_2.5mm',
                          'dt_file': 'dt_all_'},
               'tumour': {'subdir': 'Tumour/06_FORI',
                          'dt_file': 'dt_b700_',
                          'std_file': 'dt_std_data_'},
               'ms': {'subdir': 'MS/B0410637-2010-00411',
                      'dt_file': 'dt_b1200_lowres2_'}
               }

dti_file = os.path.join(base_input_dir,
                        non_HCP[dataset_type]['subdir'],
                        nn_name,
                        non_HCP[dataset_type]['dt_file'])
std_file = os.path.join(base_input_dir,
                        non_HCP[dataset_type]['subdir'],
                        nn_name,
                        non_HCP[dataset_type]['std_file'])

print('Compute MD and FA of' + dti_file)
recon_method = input("Select the reconstruction option 1, 2, 3, 4: ")

if recon_method == 1:
    # ---------- simply compute MD and FA --------------------------------:
    md_nii, fa_nii = analysis_miccai2017._MD_FA(dti_file, save_tail='_dir')
elif recon_method == 2:
    # --------- compute MD/FA and also uncertainty over MD analytically---:
    md_nii, fa_nii, __ = analysis_miccai2017._MD_FA(dti_file, std_file,
                                                    save_tail='',
                                                    compute_md_analytical=True)
elif recon_method == 3:
    # --------- compute MD/FA and uncertainty through MC estimates -------:
    analysis_miccai2017._MD_FA(dti_file, std_file,
                               no_samples=1000,
                               compute_md_analytical=False)
    # analysis_miccai2017._MD_FA(dti_file,no_samples=1)


# Also compute the errors:
# print('Compute and save the errors:')
# md_gt_nii = os.path.join(base_gt_dir,
#                          non_HCP[dataset_type]['subdir'],
#                          'dt_b1000_MD.nii')
# fa_gt_nii = os.path.join(base_gt_dir,
#                          non_HCP[dataset_type]['subdir'],
#                          'dt_b1000_FA.nii')
#
# # Compute errors:
# analysis_miccai2017._errors_MD_FA(md_nii, md_gt_nii, fa_nii, fa_gt_nii)

#analysis_miccai2017._MD_FA(dti_file, std_file, no_samples=1000)
# analysis_miccai2017._MD_FA(dti_file,no_samples=1)

