""" Perform reconstrution on non- HCP dataset (Prisma, MS, Tumour).
It assumes that DTI is available as nifti files."""

import tensorflow as tf
import configuration
import cPickle as pkl
import os
from analysis_miccai2017 import compute_err_matlab
import numpy as np


# todo: for Edge Reconstruction, make sure you take dt_est[:-1,:,:-1]

# Subjects list:
subjects_list = ['904044', '165840', '889579', '713239',
                 '899885', '117324', '214423', '857263']


# -------------------------------- Interpolation ---------------------------------------:
analysis_dir = '/SAN/vision/hcp/Ryu/miccai2017/comparison_v2/analysis'
experiment_name = '05feb17_comparison_interpolation'
experiment_file_int = os.path.join(analysis_dir, experiment_name + '.pkl')
print('Conduct experiment: ' + experiment_file_int)
err_compare = dict()
params=dict()
recon_interp = {'int_cubic':'Interpolation_cubic_DS02toDS01_MapDS02_V5.mat',
                'int_spline':'Interpolation_spline_DS02toDS01_MapDS02_V5.mat'}

for name, dt_est_int in recon_interp.iteritems():
    err_mtx = np.zeros((len(subjects_list),6))
    for j, subject in enumerate(subjects_list):
        # reconstructed files (IQT/BIQT/Interpolation):
        recon_dir = '/SAN/vision/hcp/Ryu/miccai2017/RF_recon'
        subpath = 'T1w/Diffusion'
        recon_name = dt_est_int
        params['recon_file']=os.path.join(recon_dir, subject, subpath, recon_name)

        # ground truth file:
        gt_dir = '/SAN/vision/hcp/DCA_HCP.2013.3_Proc/'
        params['gt_file'] = os.path.join(gt_dir, subject, subpath,'dt_b1000_')

        # mask file (no edge):
        mask_dir = '/SAN/vision/hcp/Ryu/miccai2017/recon'
        params['mask_file'] = os.path.join(mask_dir, subject, 'masks', 'mask_us=2_rec=5.nii')

        # reference file:
        ref_dir = '/SAN/vision/hcp/Ryu/miccai2017/comparison_v2/recon'
        ref_name = 'cnn_heteroscedastic_variational_hybrid_control_us=2_in=11_rec=5_out=14_opt=AdamOptimizer_drop=0.0_prep=standard_Diverse_TS8_Subsample343_001'
        params['ref_file'] = os.path.join(ref_dir,subject,ref_name,'dt_recon_b1000.npy')

        # save dir:
        save_dir = recon_dir
        header, __ =os.path.splitext(recon_name)
        save_name = 'error_'+ header + '.pkl'
        params['save_file'] = os.path.join(save_dir, subject, subpath, save_name)

        err=compute_err_matlab(params)
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
    with open(experiment_file_int, 'wb') as fp:
        if not os.path.exists(analysis_dir):
            os.makedirs((analysis_dir))
        pkl.dump(err_compare, fp, protocol=pkl.HIGHEST_PROTOCOL)
    print('Recon metrics saved as %s' % (experiment_file_int,))


# ----------------------------- RF-IQT/BIQT----------------------------------------------:
recon_IQT = ['OrigRecon_NoTree07_SR02_DS02toDS01_MapDS02_V5_randomidx%02i.mat'
             % (i,) for i in range(1, 9)]
recon_BIQT = ['StdBayRecon_NoTree07_SR02_DS02toDS01_MapDS02_V5_randomidx%02i.mat'
              % (i,) for i in range(1, 9)]
method_list = {'RF_original':recon_IQT, 'RF_BIQT':recon_BIQT}

analysis_dir = '/SAN/vision/hcp/Ryu/miccai2017/comparison_v2/analysis'
experiment_name = '05feb17_comparison_randomforests'
experiment_file = os.path.join(analysis_dir, experiment_name + '.pkl')
print('Conduct experiment: ' + experiment_file_int)
err_compare = dict()
params=dict()

for name, recon_file_list in method_list.iteritems():
    err_mtx = np.zeros((len(subjects_list), len(recon_file_list), 6))
    for idx, dt_est in enumerate(recon_file_list):
        for j, subject in enumerate(subjects_list):
            # reconstructed files (IQT/BIQT/Interpolation):
            recon_dir = '/SAN/vision/hcp/Ryu/miccai2017/RF_recon'
            subpath = 'T1w/Diffusion'
            recon_name = dt_est
            params['recon_file'] = os.path.join(recon_dir, subject, subpath, recon_name)

            # ground truth file:
            gt_dir = '/SAN/vision/hcp/DCA_HCP.2013.3_Proc/'
            params['gt_file'] = os.path.join(gt_dir, subject, subpath, 'dt_b1000_')

            # mask file (no edge):
            mask_dir = '/SAN/vision/hcp/Ryu/miccai2017/recon'
            params['mask_file'] = os.path.join(mask_dir, subject, 'masks', 'mask_us=2_rec=5.nii')

            # reference file:
            ref_dir = '/SAN/vision/hcp/Ryu/miccai2017/comparison_v2/recon'
            ref_name = 'cnn_heteroscedastic_variational_hybrid_control_us=2_in=11_rec=5_out=14_opt=AdamOptimizer_drop=0.0_prep=standard_Diverse_TS8_Subsample343_001'
            params['ref_file'] = os.path.join(ref_dir, subject, ref_name, 'dt_recon_b1000.npy')

            # save dir:
            save_dir = recon_dir
            header, __ = os.path.splitext(recon_name)
            save_name = 'error_' + header + '.pkl'
            params['save_file'] = os.path.join(save_dir, subject, subpath, save_name)

            err = compute_err_matlab(params)
            err_mtx[j, idx, 0] = err['rmse_noedge']
            err_mtx[j, idx, 1] = err['psnr_noedge']
            err_mtx[j, idx, 2] = err['mssim_noedge']
            err_mtx[j, idx, 3] = err['rmse_whole']
            err_mtx[j, idx, 4] = err['psnr_whole']
            err_mtx[j, idx, 5] = err['mssim_whole']

    err_compare[name] \
        = {'mean':
               {'rmse_noedge': err_mtx.mean(axis=(0, 1))[0],
                'psnr_noedge': err_mtx.mean(axis=(0, 1))[1],
                'mssim_noedge': err_mtx.mean(axis=(0, 1))[2],
                'rmse_whole': err_mtx.mean(axis=(0, 1))[3],
                'psnr_whole': err_mtx.mean(axis=(0, 1))[4],
                'mssim_whole': err_mtx.mean(axis=(0, 1))[5]
                },
           'std':
               {'rmse_noedge': np.std(err_mtx.mean(axis=0), axis=0)[0],
                'psnr_noedge': np.std(err_mtx.mean(axis=0), axis=0)[1],
                'mssim_noedge': np.std(err_mtx.mean(axis=0), axis=0)[2],
                'rmse_whole': np.std(err_mtx.mean(axis=0), axis=0)[3],
                'psnr_whole': np.std(err_mtx.mean(axis=0), axis=0)[4],
                'mssim_whole': np.std(err_mtx.mean(axis=0), axis=0)[5]
                }
           }

    # Save:
    with open(experiment_file, 'wb') as fp:
        if not os.path.exists(analysis_dir):
            os.makedirs((analysis_dir))
        pkl.dump(err_compare, fp, protocol=pkl.HIGHEST_PROTOCOL)
    print('Recon metrics saved as %s' % (experiment_file,))


## local:
# reconstructed files (IQT/BIQT/Interpolation):
recon_dir = '/SAN/vision/hcp/Ryu/miccai2017/RF_recon'
subpath = 'T1w/Diffusion'
recon_name = dt_est
params['recon_file'] = os.path.join(recon_dir, subject, subpath, recon_name)

# ground truth file:
gt_dir = '/SAN/vision/hcp/DCA_HCP.2013.3_Proc/'
params['gt_file'] = os.path.join(gt_dir, subject, subpath, 'dt_b1000_')

# mask file (no edge):
mask_dir = '/SAN/vision/hcp/Ryu/miccai2017/recon'
params['mask_file'] = os.path.join(mask_dir, subject, 'masks', 'mask_us=2_rec=5.nii')

# reference file:
ref_dir = '/SAN/vision/hcp/Ryu/miccai2017/comparison_v2/recon'
ref_name = 'cnn_simple_us=2_in=11_rec=5_out=14_opt=AdamOptimizer_drop=0.0_prep=standard_Diverse_TS8_Subsample343_001'
params['ref_file'] = os.path.join(ref_dir, subject, ref_name, 'dt_recon_b1000.npy')

# save dir:
save_dir = recon_dir
header, __ = os.path.splitext(recon_name)
save_name = 'error_' + header + '.pkl'
params['save_file'] = os.path.join(save_dir, subject, subpath, save_name)

err = compute_err_matlab(params)



# local dirs:
# opt['data_dir'] = '/SAN/vision/hcp/Ryu/IPMI2016/TrainingSet/' # '../data/'
# opt['recon_dir'] = '/Users/ryutarotanno/tmp/recon'
# opt['gt_dir'] = '/Users/ryutarotanno/DeepLearning/Test_1/data/HCP/' # ground truth dir
# opt['subpath'] = 'T1w/Diffusion'
# opt['mask_dir'] ='/Users/ryutarotanno/tmp/recon'
#
# opt['subject'] = '904044'
