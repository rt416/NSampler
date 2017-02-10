""" Compute the average error/uncertainty maps over an ensemble of
the chosen network type.

Also, due to add a feature to compute the average errors:
"""


"""
Perform reconstrution on both HCP/non-HCP dataset (Prisma, MS, Tumour).
Approximate the mean and std over FA/MD with MC sampling on the fly during reconstruction.
"""

import tensorflow as tf
import configuration
import os
import numpy as np
import analysis_miccai2017
import sr_utility
import reconstruct_mc_FA_and_MD
import nibabel as nib
from train import name_network

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
        opt['recon_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/comparison_v1/recon'
        opt['mc_no_samples'] = 200
    elif idx == 2:
        opt['method'] = 'cnn_heteroscedastic_variational_hybrid_control'
        opt['valid'] = False
        opt['dropout_rate'] = 0.0
        name = opt['method']
        opt['save_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/comparison_v2/models'
        opt['recon_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/comparison_v2/recon'
        opt['mc_no_samples'] = 200

    elif idx == 3:
        opt['method'] = 'cnn_heteroscedastic_variational_channelwise_hybrid_control'
        opt['valid'] = False
        opt['dropout_rate'] = 0.0
        name = opt['method']
        opt['save_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/comparison_v2/models'
        opt['recon_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/comparison_v2/recon'
        opt['mc_no_samples'] = 200

    elif idx == 4:
        opt['method'] = 'cnn_dropout'
        opt['valid'] = False
        opt['dropout_rate'] = 0.1
        name = opt['method'] + '_0.1'
        opt['save_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/comparison_v1/models'
        opt['recon_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/comparison_v1/recon'
        opt['mc_no_samples'] = 200
    elif idx == 5:
        opt['method'] = 'cnn_gaussian_dropout'
        opt['valid'] = False
        opt['dropout_rate'] = 0.1
        name = opt['method'] + '_0.1'
        opt['save_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/comparison_v1/models'
        opt['recon_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/comparison_v1/recon'
        opt['mc_no_samples'] = 200
    else:
        raise ValueError('no network for the given idx.')

    return name, opt


# ------------------------- compute the average RMSE/std over DTIs ----------------------:
models_list = range(1,6)
subjects_list = ['904044', '117324']
base_save_dir = '/SAN/vision/hcp/Ryu/miccai2017/10feb2017/average_maps'
# base_save_dir = '/SAN/vision/hcp/Ryu/non-HCP/HCP'


for subject in subjects_list:
    for model_idx in models_list:
        for dti_idx in range(3,9):
            rmse_volume = 0.0
            std_volume = 0.0

            for patchlib_idx in range(1,9):
                name, opt = models_update(model_idx, opt)
                opt['patchlib_idx'] = patchlib_idx
                opt['subject'] = subject

                print('\nsubject: '+ subject +
                      '\nmodel idx: ' + opt['method'] +
                      '\ndti component: ' + str(dti_idx-2))

                nn_name = name_network(opt)
                error_name = 'error_dt_recon_b1000_%i.nii' % (dti_idx,)
                std_name = 'dt_std_b1000_%i.nii' % (dti_idx,)

                error_file = os.path.join(opt['recon_dir'], subject, nn_name, error_name)
                std_file = os.path.join(opt['recon_dir'], subject, nn_name, std_name)

                rmse_volume += np.sqrt(nib.load(error_file).get_data())
                std_volume += nib.load(std_file).get_data()

            # Save the file:
            save_dir = os.path.join(base_save_dir, subject, opt['method'])
            if not(os.path.exists(save_dir)):
                os.makedirs(save_dir)
            rmse_volume/=8.0
            std_volume/=8.0
            img = nib.Nifti1Image(rmse_volume, np.eye(4))
            print('Saving :  '+'average_'+error_name)
            nib.save(img, os.path.join(save_dir,'average_'+error_name))

            img = nib.Nifti1Image(std_volume, np.eye(4))
            print('Saving :  ' + 'average_' + std_name)
            nib.save(img, os.path.join(save_dir, 'average_' + std_name))


# --------------------- compute the average RMSE/std over MD --------------------------:
models_list = range(1,6)
subjects_list = ['904044', '117324']
base_save_dir = '/SAN/vision/hcp/Ryu/miccai2017/10feb2017/average_maps'


for subject in subjects_list:
    # Compute the ground truth FA/MD first:
    md_gt_file = os.path.join(opt['recon_dir'], subject, 'maps', 'dt_b1000_MD.nii')
    fa_gt_file = os.path.join(opt['recon_dir'], subject, 'maps', 'dt_b1000_FA.nii')
    save_maps_dir = os.path.join(opt['recon_dir'], subject, 'maps')
    if not (os.path.exists(fa_gt_file)) or not (os.path.exists(md_gt_file)):
        print('Ground truth FA or MD does not exist. Compute them ...')
        dti_gt_root_dir = os.path.join(opt['gt_dir'], subject, opt['subpath'])
        dti_gt_file = dti_gt_root_dir + '/dt_b1000_'
        os.makedirs(save_maps_dir)
        analysis_miccai2017._MD_FA(dti_gt_file, save_dir=save_maps_dir)

    # Compute the average RMSE/uncertainty over MD:
    for model_idx in models_list:

        rmse_volume = 0.0
        std_volume = 0.0

        for patchlib_idx in range(1,9):
            name, opt = models_update(model_idx, opt)
            opt['patchlib_idx'] = patchlib_idx
            opt['subject'] = subject

            print('\nsubject: '+ subject +
                  '\nmodel idx: ' + opt['method'])

            nn_name = name_network(opt)
            dti_name = 'dt_recon_b1000_'
            std_name = 'dt_std_b1000_'
            dti_file = os.path.join(opt['recon_dir'], subject, nn_name, dti_name)
            std_file = os.path.join(opt['recon_dir'], subject, nn_name, std_name)

            # compute the MD and FA, and std over MD (but not FA!):
            md_nii, __, md_std_nii = analysis_miccai2017._MD_FA(dti_file,
                                                                std_file,
                                                                save_tail='',
                                                                compute_md_analytical=True)

            # compute the rmse:
            rmse_volume += sr_utility.compute_rmse_nii(nii_1=md_gt_file,
                                                       nii_2=md_nii,
                                                       save_file=None)
            std_volume += nib.load(md_std_nii).get_data()

        # Save the file:
        save_dir = os.path.join(base_save_dir, subject, opt['method'])
        if not(os.path.exists(save_dir)):
            os.makedirs(save_dir)
        rmse_volume/=8.0
        std_volume/=8.0
        img = nib.Nifti1Image(rmse_volume, np.eye(4))
        print('Saving :  '+'average_'+ 'error' + dti_name + 'MD.nii' )
        nib.save(img, os.path.join(save_dir,'average_error_' + dti_name + 'MD.nii' ))

        img = nib.Nifti1Image(std_volume, np.eye(4))
        print('Saving :  ' + 'average_' + std_name + 'MD.nii')
        nib.save(img, os.path.join(save_dir, 'average_' + std_name + 'MD.nii'))







