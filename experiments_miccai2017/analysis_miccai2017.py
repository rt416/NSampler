from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import timeit

from train import define_checkpoint, name_network
from reconstruct_mcdropout import super_resolve_mcdropout
from reconstruct import super_resolve

from sr_analysis import correlation_plot_and_analyse
import sr_utility
from sr_utility import read_dt_volume
from sr_analysis import compare_images
from sr_analysis import plot_ROC

import matplotlib.pyplot as plt
import nibabel as nib
import cPickle as pkl

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
    recon_file = os.path.join(recon_dir, subject, nn_dir, 'dt_recon_b1000.npy')
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
def compute_err(opt):
    print('Method: %s \nSubject: %s' %
          (opt['method'], opt['subject']))
    recon_dir = opt['recon_dir']
    gt_dir = opt['gt_dir']
    mask_dir = opt['mask_dir']
    subpath = opt['subpath']
    subject = opt['subject']
    nn_dir = name_network(opt)

    # Load the ground truth/estimated high-res volumes:
    recon_file = os.path.join(recon_dir, subject, nn_dir, 'dt_recon_b1000.npy')
    gt_file = os.path.join(gt_dir, subject, subpath,'dt_b1000_')
    uncertainty_file = os.path.join(recon_dir, subject, nn_dir, 'dt_std_b1000.npy')
    mask_file = os.path.join(mask_dir, subject, 'masks',
                             'mask_us=' + str(opt['upsampling_rate']) + \
                             '_rec=' + str(5) + '.nii')

    dt_gt = read_dt_volume(nameroot=gt_file)
    dt_est = np.load(recon_file)

    # Load the masks:
    img = nib.load(os.path.join(mask_dir, mask_file))
    mask_noedge = img.get_data() == 0
    mask_whole = dt_est[:, :, :, 0] == 0

    # Compute rmse, psnr, mssim:
    start_time = timeit.default_timer()
    err = dict()
    err['rmse_noedge'], err['psnr_noedge'], err['mssim_noedge'] = \
        compare_images(dt_gt[...,2:], dt_est[...,2:], mask_noedge)
    print('\n(No edge)\nRMSE: %.10f \nPSNR: %.5f \nMSSIM: %.5f' %
          (err['rmse_noedge'],err['psnr_noedge'],err['mssim_noedge']))
    err['rmse_whole'], err['psnr_whole'], err['mssim_whole'] = \
        compare_images(dt_gt[..., 2:], dt_est[..., 2:], mask_whole)
    print('\n(Whole)\nRMSE: %.10f \nPSNR: %.5f \nMSSIM: %.5f' %
          (err['rmse_whole'],err['psnr_whole'],err['mssim_whole']))
    end_time = timeit.default_timer()
    print('Took %f secs' % (end_time - start_time,))

    # Save this as a separate file:
    err_file = os.path.join(recon_dir, subject, nn_dir, 'error.pkl')
    with open(err_file, 'wb') as fp:
        pkl.dump(err, fp, protocol=pkl.HIGHEST_PROTOCOL)
    print('Errors details saved as %s' %(err_file,))

    return err

# double the size of non-HCP DTI:
def resize_DTI(dti, r):
    """ Resize the size of a brain volume.
    Args:
        dti: (4d np.array)
        r: upsampling rate
    """
    s_old = dti.shape
    s_new = (r*s_old[0],r*s_old[1],r*s_old[2],s_old[3])

    dti_new = np.zeros(s_new)
    shift_indices = [(i, j, k)
                     for k in xrange(r)
                     for j in xrange(r)
                     for i in xrange(r)]

    for (shift_x, shift_y, shift_z) in shift_indices:
        dti_new[shift_x::r, shift_y::r, shift_z::r,:] = dti
    return dti_new


# compute error on non-HCP dataset:
def nonhcp_reconstruct(opt, dataset_type='prisma'):
    # ------------------------ Reconstruct --------------------------------------:
    print('Method: %s' %
          (opt['method'], ))

    # load parameters:
    recon_dir = opt['recon_dir']
    gt_dir = opt['gt_dir']
    input_file_name = opt['input_file_name']

    # Load the input low-res DT image:
    print('... loading the low-res input DTI ...')
    dt_lowres = sr_utility.read_dt_volume(os.path.join(gt_dir,input_file_name))
    if not(dataset_type=='ms'):
        dt_lowres = resize_DTI(dt_lowres,opt['upsampling_rate'])
    else:
        print('MS dataset: no need to resample.')
    # clear the graph (is it necessary?)
    tf.reset_default_graph()

    # Reconstruct & save:
    nn_dir = name_network(opt)
    print('\nReconstruct high-res dti with the network: \n%s.' % nn_dir)

    output_file = os.path.join(recon_dir, nn_dir, 'dt_recon.npy')
    uncertainty_file = os.path.join(recon_dir, nn_dir, 'dt_std.npy')
    __, output_header = os.path.split(output_file)
    __, uncertainty_header = os.path.split(uncertainty_file)

    if not (os.path.exists(os.path.join(recon_dir, nn_dir))):
        os.makedirs(os.path.join(recon_dir, nn_dir))

    start_time = timeit.default_timer()
    if not(opt['method']=='cnn_simple'):
        print('... saving as %s' % output_file)
        dt_hr, dt_std = super_resolve_mcdropout(dt_lowres, opt)
        np.save(output_file, dt_hr)
        np.save(uncertainty_file, dt_std)

        # save as .nii
        print('\nSave each estimated dti/std separately as a nii file ...')
        sr_utility.save_as_nifti(output_header, os.path.join(recon_dir, nn_dir),
                                 gt_dir='', save_as_ijk=True)
        sr_utility.save_as_nifti(uncertainty_header, os.path.join(recon_dir, nn_dir),
                                 gt_dir='', save_as_ijk=True)
    else:
        print('... saving as %s' % output_file)
        dt_hr = super_resolve(dt_lowres, opt)
        np.save(output_file, dt_hr)

        # save as .nii
        print('\nSave each estimated dti separately as a nii file ...')
        sr_utility.save_as_nifti(output_header, os.path.join(recon_dir, nn_dir),
                                 gt_dir='', save_as_ijk=True)

    end_time = timeit.default_timer()
    print('\nIt took %f secs. \n' % (end_time - start_time))


# Receiver Operating Characteristics:
def get_ROC(opt):
    print(opt['method'])
    recon_dir = opt['recon_dir']
    gt_dir = opt['gt_dir']
    mask_dir = opt['mask_dir']
    subpath = opt['subpath']
    subject = opt['subject']
    nn_dir = name_network(opt)

    # Compute the reconstruction errors:
    recon_file = os.path.join(recon_dir, subject, nn_dir, 'dt_mcrecon_b1000.npy')
    gt_file = os.path.join(gt_dir, subject, subpath, 'dt_b1000_')
    uncertainty_file = os.path.join(recon_dir, subject, nn_dir, 'dt_std_b1000.npy')
    mask_file = os.path.join(mask_dir, subject, 'masks',
                             'mask_us=' + str(opt['upsampling_rate']) + \
                             '_rec=' + str(5) + '.nii')

    dt_gt = read_dt_volume(nameroot=gt_file)
    dt_est = np.load(recon_file)
    dt_std = np.load(uncertainty_file)
    mask = dt_est[:, :, :, 0] == 0

    # Plot uncertainty against error:
    dti_list = range(2, 8)

    for idx, dti_idx in enumerate(dti_list):
        start_time = timeit.default_timer()
        plt.subplot(2, 3, idx + 1)
        title = '%i:' % (idx + 1,)
        plot_ROC(dt_gt[:, :, :, dti_idx], dt_est[:, :, :, dti_idx], dt_std[:, :, :, dti_idx],
                 mask, no_points=10000)
        plt.title(title)
        end_time = timeit.default_timer()
        print('component %i: took %f secs' % (idx + 1, (end_time - start_time)))

    plt.show()







