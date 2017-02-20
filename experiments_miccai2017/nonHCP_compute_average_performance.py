""" Perform reconstrution on non-HCP dataset (Prisma, Life).
It assumes that DTI is available as nifti files."""

import tensorflow as tf
import configuration
import cPickle as pkl
import os
import analysis_miccai2017
import numpy as np


# todo: for Edge Reconstruction, make sure you take dt_est[:-1,:,:-1]

# Subjects list:
subjects_list = ['Diffusion_2.5mm/']


# --------------------------------RF (edge reconstruction) --------------------------------------:
analysis_dir = '/Users/ryutarotanno/DeepLearning/nsampler/recon/miccai2017/Prisma/analysis'
experiment_name = '/19feb17_comparison'
experiment_file_int = analysis_dir + experiment_name + '.pkl'

print('Conduct experiment: ' + experiment_file_int)
err_compare = dict()
params=dict()
params['edge'] = False
recon_dict = {'IQT_rf_noedge':'IQT_rf',
              'BIQT_rf_noedge':'BIQT_rf',
              'IQT_cubic':'IQT_cubic',
              'cnn_simple':'cnn_simple_us=2_in=11_rec=5_out=14_opt=AdamOptimizer_drop=0.0_prep=standard_Diverse_TS8_Subsample343_001',
              'cnn_gaussian': 'cnn_gaussian_dropout_us=2_in=11_rec=5_out=14_opt=AdamOptimizer_drop=0.1_prep=standard_Diverse_TS8_Subsample343_001',
              'cnn_dropout': 'cnn_dropout_us=2_in=11_rec=5_out=14_opt=AdamOptimizer_drop=0.1_prep=standard_Diverse_TS8_Subsample343_001',
              'cnn_variational': 'cnn_variational_dropout_us=2_in=11_rec=5_out=14_opt=AdamOptimizer_drop=0.0_prep=standard_Diverse_TS8_Subsample343_001',
              'cnn_hetero': 'cnn_heteroscedastic_us=2_in=11_rec=5_out=14_opt=AdamOptimizer_drop=0.0_prep=standard_Diverse_TS8_Subsample343_001',
              'cnn_heterovar': 'cnn_heteroscedastic_variational_channelwise_hybrid_control_us=2_in=11_rec=5_out=14_opt=AdamOptimizer_drop=0.0_prep=standard_Diverse_TS8_Subsample343_001',
              'cnn_heterovar_channel': 'cnn_heteroscedastic_variational_hybrid_control_us=2_in=11_rec=5_out=14_opt=AdamOptimizer_drop=0.0_prep=standard_Diverse_TS8_Subsample343_001',
              }


for name, nn_dir in recon_dict.iteritems():
    err_mtx = np.zeros((len(subjects_list),6))
    for j, subject in enumerate(subjects_list):

        # set:
        # params['edge'] = True

        # reconstructed files (IQT/BIQT/Interpolation):
        recon_dir = '/Users/ryutarotanno/DeepLearning/nsampler/recon/miccai2017/Prisma/'
        params['recon_file'] = recon_dir + subject + nn_dir + '/dt_recon_resized_'

        # ground truth file:
        gt_dir = '/Users/ryutarotanno/DeepLearning/nsampler/data/Prisma/Diffusion_1.35mm'
        params['gt_file'] = gt_dir + '/dt_b1000_'

        # mask file: no edge and whole
        mask_dir = '/Users/ryutarotanno/DeepLearning/nsampler/recon/miccai2017/Prisma/Diffusion_2.5mm/masks'
        params['mask_file_noedge'] = mask_dir + '/mask_noedge.nii'
        params['mask_file_whole'] = mask_dir + '/mask_whole.nii'

        # save dir:
        save_dir = recon_dir + subject + nn_dir
        params['save_file'] = save_dir + '/error.pkl'

        err=analysis_miccai2017.compute_err_nonhcp(params)
        err_mtx[j, 0] = err['rmse_noedge']
        err_mtx[j, 1] = err['psnr_noedge']
        err_mtx[j, 2] = err['mssim_noedge']
        err_mtx[j, 3] = err['rmse_whole']
        err_mtx[j, 4] = err['psnr_whole']
        err_mtx[j, 5] = err['mssim_whole']

    # Compute the average errors over subjects:
    err_compare[name] = {'error':
                             {'rmse_noedge': err_mtx.mean(axis=0)[0],
                              'psnr_noedge': err_mtx.mean(axis=0)[1],
                              'mssim_noedge': err_mtx.mean(axis=0)[2],
                              'rmse_whole': err_mtx.mean(axis=0)[3],
                              'psnr_whole': err_mtx.mean(axis=0)[4],
                              'mssim_whole': err_mtx.mean(axis=0)[5]
                              }
                         }

    if not os.path.exists(analysis_dir):
        os.makedirs((analysis_dir))

    with open(experiment_file_int, 'wb') as fp:
        pkl.dump(err_compare, fp, protocol=pkl.HIGHEST_PROTOCOL)
    print('Recon metrics saved as %s' % (experiment_file_int,))

