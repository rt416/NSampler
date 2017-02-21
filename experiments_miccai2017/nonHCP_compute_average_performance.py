""" Perform reconstrution on non-HCP dataset (Prisma, Life).
It assumes that DTI is available as nifti files."""

import tensorflow as tf
import configuration
import cPickle as pkl
import os
import analysis_miccai2017
import numpy as np


# todo: for Edge Reconstruction, make sure you take dt_est[:-1,:,:-1]/8

# # -------------------------------- Prisma experiment (local) --------------------------------------:
# # Subjects list:
# subjects_list = ['Diffusion_2.5mm/']
#
# analysis_dir = '/Users/ryutarotanno/DeepLearning/nsampler/recon/miccai2017/Prisma/analysis'
# experiment_name = '/19feb17_comparison'
# experiment_file_int = analysis_dir + experiment_name + '.pkl'
#
# print('Conduct experiment: ' + experiment_file_int)
# err_compare = dict()
# params=dict()
# params['edge'] = False
# recon_dict = {'IQT_rf_noedge':'IQT_rf',
#               'BIQT_rf_noedge':'BIQT_rf',
#               'IQT_cubic':'IQT_cubic',
#               'cnn_simple':'cnn_simple_us=2_in=11_rec=5_out=14_opt=AdamOptimizer_drop=0.0_prep=standard_Diverse_TS8_Subsample343_001',
#               'cnn_gaussian': 'cnn_gaussian_dropout_us=2_in=11_rec=5_out=14_opt=AdamOptimizer_drop=0.1_prep=standard_Diverse_TS8_Subsample343_001',
#               'cnn_dropout': 'cnn_dropout_us=2_in=11_rec=5_out=14_opt=AdamOptimizer_drop=0.1_prep=standard_Diverse_TS8_Subsample343_001',
#               'cnn_variational': 'cnn_variational_dropout_us=2_in=11_rec=5_out=14_opt=AdamOptimizer_drop=0.0_prep=standard_Diverse_TS8_Subsample343_001',
#               'cnn_hetero': 'cnn_heteroscedastic_us=2_in=11_rec=5_out=14_opt=AdamOptimizer_drop=0.0_prep=standard_Diverse_TS8_Subsample343_001',
#               'cnn_heterovar': 'cnn_heteroscedastic_variational_channelwise_hybrid_control_us=2_in=11_rec=5_out=14_opt=AdamOptimizer_drop=0.0_prep=standard_Diverse_TS8_Subsample343_001',
#               'cnn_heterovar_channel': 'cnn_heteroscedastic_variational_hybrid_control_us=2_in=11_rec=5_out=14_opt=AdamOptimizer_drop=0.0_prep=standard_Diverse_TS8_Subsample343_001',
#               }
#
# for name, nn_dir in recon_dict.iteritems():
#     err_mtx = np.zeros((len(subjects_list),6))
#     for j, subject in enumerate(subjects_list):
#
#         # set:
#         # params['edge'] = True
#
#         # reconstructed files (IQT/BIQT/Interpolation):
#         recon_dir = '/Users/ryutarotanno/DeepLearning/nsampler/recon/miccai2017/Prisma/'
#         params['recon_file'] = recon_dir + subject + nn_dir + '/dt_recon_resized_'
#
#         # ground truth file:
#         gt_dir = '/Users/ryutarotanno/DeepLearning/nsampler/data/Prisma/Diffusion_1.35mm'
#         params['gt_file'] = gt_dir + '/dt_b1000_'
#
#         # mask file: no edge and whole
#         mask_dir = '/Users/ryutarotanno/DeepLearning/nsampler/recon/miccai2017/Prisma/Diffusion_2.5mm/masks'
#         params['mask_file_noedge'] = mask_dir + '/mask_noedge.nii'
#         params['mask_file_whole'] = mask_dir + '/mask_whole.nii'
#
#         # save dir:
#         save_dir = recon_dir + subject + nn_dir
#         params['save_file'] = save_dir + '/error.pkl'
#
#         err=analysis_miccai2017.compute_err_nonhcp(params)
#         err_mtx[j, 0] = err['rmse_noedge']
#         err_mtx[j, 1] = err['psnr_noedge']
#         err_mtx[j, 2] = err['mssim_noedge']
#         err_mtx[j, 3] = err['rmse_whole']
#         err_mtx[j, 4] = err['psnr_whole']
#         err_mtx[j, 5] = err['mssim_whole']
#
#     # Compute the average errors over subjects:
#     err_compare[name] = {'error':
#                              {'rmse_noedge': err_mtx.mean(axis=0)[0],
#                               'psnr_noedge': err_mtx.mean(axis=0)[1],
#                               'mssim_noedge': err_mtx.mean(axis=0)[2],
#                               'rmse_whole': err_mtx.mean(axis=0)[3],
#                               'psnr_whole': err_mtx.mean(axis=0)[4],
#                               'mssim_whole': err_mtx.mean(axis=0)[5]
#                               }
#                          }
#
#     if not os.path.exists(analysis_dir):
#         os.makedirs((analysis_dir))
#
#     with open(experiment_file_int, 'wb') as fp:
#         pkl.dump(err_compare, fp, protocol=pkl.HIGHEST_PROTOCOL)
#     print('Recon metrics saved as %s' % (experiment_file_int,))

# # -------------------------------- Life-HCP experiment (local) --------------------------------------:
# print('Compute errors on the Life dataset!!\n')
#
# # subjects_list = ['LS5007', 'LS5040', 'LS5049', 'LS6006', 'LS6038',
# #                  'LS5038', 'LS5041', 'LS6003', 'LS6009', 'LS6046']
#
# subjects_list = ['LS6006', 'LS6038', 'LS6003', 'LS6009', 'LS6046']
#
#
# analysis_dir = '/SAN/vision/hcp/Ryu/miccai2017/Life/analysis'
# experiment_name = '/20feb17_comparison_oldpeople'
# experiment_file_int = analysis_dir + experiment_name + '.pkl'
#
# print('Conduct experiment: ' + experiment_file_int)
# err_compare = dict()
# params=dict()
# params['edge'] = False
# recon_dict = {'IQT_rf_noedge':'/IQT_rf_noedge',
#               'BIQT_rf_noedge':'/BIQT_rf_noedge',
#               'Interp_cubic':'/Interp_cubic',
#               'Interp_spline':'/Interp_spline',
#               'cnn_simple':'/cnn_simple_us=2_in=11_rec=5_out=14_opt=AdamOptimizer_drop=0.0_prep=standard_Diverse_TS8_Subsample343_001',
#               'cnn_gaussian': '/cnn_gaussian_dropout_us=2_in=11_rec=5_out=14_opt=AdamOptimizer_drop=0.1_prep=standard_Diverse_TS8_Subsample343_001',
#               'cnn_dropout': '/cnn_dropout_us=2_in=11_rec=5_out=14_opt=AdamOptimizer_drop=0.1_prep=standard_Diverse_TS8_Subsample343_001',
#               'cnn_variational': '/cnn_variational_dropout_us=2_in=11_rec=5_out=14_opt=AdamOptimizer_drop=0.0_prep=standard_Diverse_TS8_Subsample343_001',
#               'cnn_variational_channel': '/cnn_variational_dropout_channelwise_us=2_in=11_rec=5_out=14_opt=AdamOptimizer_drop=0.0_prep=standard_Diverse_TS8_Subsample343_001',
#               'cnn_hetero': '/cnn_heteroscedastic_us=2_in=11_rec=5_out=14_opt=AdamOptimizer_drop=0.0_prep=standard_Diverse_TS8_Subsample343_001',
#               'cnn_heterovar': '/cnn_heteroscedastic_variational_channelwise_hybrid_control_us=2_in=11_rec=5_out=14_opt=AdamOptimizer_drop=0.0_prep=standard_Diverse_TS8_Subsample343_001',
#               'cnn_heterovar_channel': '/cnn_heteroscedastic_variational_hybrid_control_us=2_in=11_rec=5_out=14_opt=AdamOptimizer_drop=0.0_prep=standard_Diverse_TS8_Subsample343_001',
#               }
#
#
# for name, nn_dir in recon_dict.iteritems():
#     err_mtx = np.zeros((len(subjects_list),6))
#     for j, subject in enumerate(subjects_list):
#
#         # set:
#         # params['edge'] = True
#
#         # reconstructed files (IQT/BIQT/Interpolation):
#         recon_dir = '/SAN/vision/hcp/Ryu/miccai2017/Life/'
#         params['recon_file'] = recon_dir + subject + nn_dir + '/dt_recon_'
#
#         # ground truth file:
#         gt_dir = '/SAN/vision/hcp/DCA_HCP.2013.3_Proc/'
#         subpath = '/Diffusion/Diffusion/'
#         params['gt_file'] = gt_dir + subject + subpath + '/dt_b1000_'
#
#         # mask file: no edge and whole
#         mask_dir = '/Users/ryutarotanno/DeepLearning/nsampler/recon/miccai2017/Prisma/Diffusion_2.5mm/masks'
#         params['mask_file_noedge'] = recon_dir + subject + '/IQT_rf_noedge' + '/dt_recon_1.nii'
#         params['mask_file_whole'] = recon_dir + subject \
#                                     + '/cnn_simple_us=2_in=11_rec=5_out=14_opt=AdamOptimizer_drop=0.0_prep=standard_Diverse_TS8_Subsample343_001'\
#                                     + '/dt_recon_1.nii'
#
#         # save dir:
#         save_dir = recon_dir + subject + nn_dir
#         params['save_file'] = save_dir + '/error.pkl'
#
#         err=analysis_miccai2017.compute_err_nonhcp(params)
#         err_mtx[j, 0] = err['rmse_noedge']
#         err_mtx[j, 1] = err['psnr_noedge']
#         err_mtx[j, 2] = err['mssim_noedge']
#         err_mtx[j, 3] = err['rmse_whole']
#         err_mtx[j, 4] = err['psnr_whole']
#         err_mtx[j, 5] = err['mssim_whole']
#
#     # Compute the average errors over subjects:
#     err_compare[name] = {'error':
#                              {'rmse_noedge': err_mtx.mean(axis=0)[0],
#                               'psnr_noedge': err_mtx.mean(axis=0)[1],
#                               'mssim_noedge': err_mtx.mean(axis=0)[2],
#                               'rmse_whole': err_mtx.mean(axis=0)[3],
#                               'psnr_whole': err_mtx.mean(axis=0)[4],
#                               'mssim_whole': err_mtx.mean(axis=0)[5]
#                               }
#                          }
#
#     if not os.path.exists(analysis_dir):
#         os.makedirs((analysis_dir))
#
#     with open(experiment_file_int, 'wb') as fp:
#         pkl.dump(err_compare, fp, protocol=pkl.HIGHEST_PROTOCOL)
#     print('Recon metrics saved as %s' % (experiment_file_int,))

# -------------------------------- Life-HCP: compute errors with boundary completion --------------------------------------:
print('Compute errors on the Life dataset!!\n')

subjects_list = ['LS5007']
# subjects_list = ['LS6006', 'LS6038', 'LS6003', 'LS6009', 'LS6046']


analysis_dir = '/SAN/vision/hcp/Ryu/miccai2017/Life/analysis'
experiment_name = '/20feb17_comparison_rf_edgecompletion'
experiment_file_int = analysis_dir + experiment_name + '.pkl'

print('Conduct experiment: ' + experiment_file_int)
err_compare = dict()
params=dict()
params['edge'] = False
recon_dict = {'IQT_rf_whole':'/IQT_rf_whole',
              'BIQT_rf_whole':'/BIQT_rf_whole'}


for name, nn_dir in recon_dict.iteritems():
    err_mtx = np.zeros((len(subjects_list),9))
    for j, subject in enumerate(subjects_list):

        # set:
        # params['edge'] = True

        # reconstructed files (IQT/BIQT/Interpolation):
        recon_dir = '/SAN/vision/hcp/Ryu/miccai2017/Life/'
        params['recon_file'] = recon_dir + subject + nn_dir + '/dt_recon_'

        # ground truth file:
        gt_dir = '/SAN/vision/hcp/DCA_HCP.2013.3_Proc/'
        subpath = '/Diffusion/Diffusion/'
        params['gt_file'] = gt_dir + subject + subpath + '/dt_b1000_'

        # mask file: no edge and whole
        mask_dir = '/Users/ryutarotanno/DeepLearning/nsampler/recon/miccai2017/Prisma/Diffusion_2.5mm/masks'
        params['mask_file_noedge'] = recon_dir + subject + '/IQT_rf_noedge' + '/dt_recon_1.nii'
        params['mask_file_whole'] = recon_dir + subject \
                                    + '/cnn_simple_us=2_in=11_rec=5_out=14_opt=AdamOptimizer_drop=0.0_prep=standard_Diverse_TS8_Subsample343_001'\
                                    + '/dt_recon_1.nii'

        # save dir:
        save_dir = recon_dir + subject + nn_dir
        params['save_file'] = save_dir + '/error.pkl'

        err=analysis_miccai2017.compute_err_nonhcp(params)
        err_mtx[j, 0] = err['interior_rmse']
        err_mtx[j, 1] = err['interior_psnr']
        err_mtx[j, 2] = err['interior_mssim']
        err_mtx[j, 3] = err['whole_rmse']
        err_mtx[j, 4] = err['whole_psnr']
        err_mtx[j, 5] = err['whole_mssim']
        err_mtx[j, 6] = err['edge_rmse']
        err_mtx[j, 7] = err['edge_psnr']
        err_mtx[j, 8] = err['edge_mssim']

    # Compute the average errors over subjects:
    err_compare[name] = {'error':
                             {'interior_rmse': err_mtx.mean(axis=0)[0],
                              'interior_psnr': err_mtx.mean(axis=0)[1],
                              'interior_mssim': err_mtx.mean(axis=0)[2],
                              'whole_rmse': err_mtx.mean(axis=0)[3],
                              'whole_psnr': err_mtx.mean(axis=0)[4],
                              'whole_mssim': err_mtx.mean(axis=0)[5],
                              'edge_rmse': err_mtx.mean(axis=0)[6],
                              'edge_psnr': err_mtx.mean(axis=0)[7],
                              'edge_mssim': err_mtx.mean(axis=0)[8],
                              }
                         }

    if not os.path.exists(analysis_dir):
        os.makedirs((analysis_dir))

    with open(experiment_file_int, 'wb') as fp:
        pkl.dump(err_compare, fp, protocol=pkl.HIGHEST_PROTOCOL)
    print('Recon metrics saved as %s' % (experiment_file_int,))