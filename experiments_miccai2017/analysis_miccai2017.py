from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import timeit
from train import define_checkpoint, name_network
from sr_analysis import correlation_plot_and_analyse
from sr_utility import read_dt_volume
import matplotlib.pyplot as plt

# Plot RMSE vs uncertainty:
def plot_rmse_vs_uncertainty(opt):
    print(opt['method'])
    recon_dir = opt['recon_dir']
    gt_dir = opt['gt_dir']
    mask_dir = opt['mask_dir']
    subpath = opt['subpath']
    subject = opt['subject']
    nn_dir = name_network(opt)

    # Compute the reconstruction errors:
    recon_file = os.path.join(recon_dir, subject, nn_dir, 'dt_mcrecon_b1000.npy')
    gt_file = os.path.join(gt_dir, subject, subpath,'dt_b1000_')
    uncertainty_file = os.path.join(recon_dir, subject, nn_dir, 'dt_std_b1000.npy')
    mask_file = os.path.join(mask_dir, subject, 'masks',
                             'mask_us=' + str(opt['upsampling_rate']) + \
                             '_rec=' + str(5) + '.nii')

    dt_gt = read_dt_volume(nameroot=gt_file)
    dt_est = np.load(recon_file)
    dt_std = np.load(uncertainty_file)
    mask = dt_est[:, :, :, 0] == 0

    # Plot uncertainty against error:
    dti_list = range(2,8)

    for idx, dti_idx in enumerate(dti_list):
        start_time = timeit.default_timer()
        plt.subplot(2,3,idx+1)
        img_err = np.sqrt((dt_gt[:,:,:,dti_idx] - dt_est[:,:,:,dti_idx]) ** 2)
        img_unc = dt_std[:,:,:,dti_idx]
        title = '%i:' % (idx + 1,)
        correlation_plot_and_analyse(img_unc, img_err, mask, no_points=300,
                                     xlabel='std', ylabel='rmse', title=title, opt=opt)
        end_time = timeit.default_timer()
        print('component %i: took %f secs' % (idx+1,(end_time - start_time)))

    plt.show()

# Compute rmse, psnr, mssim on the brain and the edge, and store to settings.pkl:




# Received Operating Characteristics:








