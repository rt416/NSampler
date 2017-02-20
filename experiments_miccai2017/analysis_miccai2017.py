from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import timeit

from train import define_checkpoint, name_network
from reconstruct_mcdropout import super_resolve_mcdropout
from reconstruct_mc_heterovariational import super_resolve_mcdropout_heterovariational
from reconstruct import super_resolve

from sr_analysis import correlation_plot_and_analyse
import sr_utility
from sr_utility import read_dt_volume
from sr_analysis import compare_images
from sr_analysis import plot_ROC

import matplotlib.pyplot as plt
import nibabel as nib
import cPickle as pkl
import scipy.io as sio

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
    gt_file = os.path.join(gt_dir, subject, subpath, 'dt_b1000_')
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


def compute_err_nonhcp(params):
    """ Compute errors for """
    recon_file = params['recon_file']
    gt_file = params['gt_file']
    mask_file_noedge = params['mask_file_noedge']
    mask_file_whole = params['mask_file_whole']
    save_file = params['save_file']

    # Load the ground truth/estimated high-res volumes:
    dt_gt = read_dt_volume(nameroot=gt_file)
    dt_est = read_dt_volume(nameroot=recon_file)

    print('shape of dt_gt and dt_est are: %s and %s' % (dt_gt.shape, dt_est.shape))
    if params['edge']:
        print('Edge reconstruction is selected.')
        dt_est = dt_est[:-1, :, :-1, :] / 7.0
    print('shape of dt_gt and dt_est are: %s and %s' % (dt_gt.shape, dt_est.shape))

    # Get the mask from deep learning reconstruction:
    img = nib.load(mask_file_whole)
    mask_whole = img.get_data() == 0

    # Load the masks with no edges:
    img = nib.load(mask_file_noedge)
    mask_noedge = img.get_data() == 0

    # Compute rmse, psnr, mssim:
    start_time = timeit.default_timer()
    err = dict()
    err['rmse_noedge'], err['psnr_noedge'], err['mssim_noedge'] = \
        compare_images(dt_gt[..., 2:], dt_est[..., 2:], mask_noedge)
    print('\n(No edge)\nRMSE: %.10f \nPSNR: %.5f \nMSSIM: %.5f' %
          (err['rmse_noedge'], err['psnr_noedge'], err['mssim_noedge']))
    err['rmse_whole'], err['psnr_whole'], err['mssim_whole'] = \
        compare_images(dt_gt[..., 2:], dt_est[..., 2:], mask_whole)
    print('\n(Whole)\nRMSE: %.10f \nPSNR: %.5f \nMSSIM: %.5f' %
          (err['rmse_whole'], err['psnr_whole'], err['mssim_whole']))
    end_time = timeit.default_timer()
    print('Took %f secs' % (end_time - start_time,))

    # Save this as a separate file:
    with open(save_file, 'wb') as fp:
        pkl.dump(err, fp, protocol=pkl.HIGHEST_PROTOCOL)
    print('Errors details saved as %s' % (save_file,))
    return err

def compute_err_matlab(params):
    """ Compute errors for """

    recon_file = params['recon_file']
    gt_file = params['gt_file']
    mask_file = params['mask_file']
    ref_file = params['ref_file']
    save_file = params['save_file']

    # Load the ground truth/estimated high-res volumes:
    dt_gt = read_dt_volume(nameroot=gt_file)
    mat_contents = sio.loadmat(recon_file)
    dt_est = mat_contents['img_RFrecon']
    print('shape of dt_gt and dt_est are: %s and %s' %(dt_gt.shape, dt_est.shape))
    if params['edge']:
        print('Edge reconstruction is selected.')
        dt_est = dt_est[:-1,:,:-1,:]/7.0
    print('shape of dt_gt and dt_est are: %s and %s' % (dt_gt.shape, dt_est.shape))

    # Get the mask from deep learning reconstruction:
    ref_est = np.load(ref_file)
    mask_whole = ref_est[:, :, :, 0] == 0
    del ref_est

    # Load the masks with no edges:
    img = nib.load(mask_file)
    mask_noedge = img.get_data() == 0

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
    with open(save_file, 'wb') as fp:
        pkl.dump(err, fp, protocol=pkl.HIGHEST_PROTOCOL)
    print('Errors details saved as %s' %(save_file,))
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



# non-HCP data. Compute the mean and std of FA.
def _MD_FA(dti_file, std_file=None, no_samples=500,
           save_dir=None, save_tail='',
           compute_error=False,
           compute_md_analytical=False):
    """ Compute the mean MD and FA on non-HCP dataset
    Args:
        dti_file (str) : the name of the mean dti nifti files
        std_file (str) : the name of the corresponding std
        save_dir (str) : specify the dir for saving the files.
        Otherwise, MD and FA are saved in the same dir as input.
    """
    if not(save_dir==None):
        _, dti_header = os.path.split(dti_file)
        save_file = os.path.join(save_dir, dti_header)
    else:
        save_file = dti_file

    dti_mean = sr_utility.read_dt_volume(dti_file)
    if std_file == None and compute_md_analytical == False:
        md, fa =sr_utility.compute_MD_and_FA(dti_mean[...,2:])
        md_nii = save_file + 'MD' + save_tail + '.nii'
        sr_utility.ndarray_to_nifti(md, md_nii)
        fa_nii = save_file + 'FA' + save_tail + '.nii'
        sr_utility.ndarray_to_nifti(fa, fa_nii)
        return md_nii, fa_nii
    elif compute_md_analytical == True :
        dti_std = sr_utility.read_dt_volume(std_file)
        md, fa = sr_utility.compute_MD_and_FA(dti_mean[..., 2:])
        dti_std2=dti_std[..., 2:]
        md_std = np.sqrt(dti_std2[..., 0] ** 2 + dti_std2[..., 3] ** 2 + dti_std2[..., 5] ** 2) / 3.0
        # md, md_std = sr_utility.propagate_uncertainty_analytical_MD(dti_mean[..., 2:],
        #                                                         dti_std[...,2:])
        md_nii = save_file + 'MD' + save_tail + '.nii'
        md_std_nii = save_file + 'MD_std' + save_tail + '_analytical' '.nii'

        sr_utility.ndarray_to_nifti(md, md_nii)
        sr_utility.ndarray_to_nifti(md_std, md_std_nii)
        fa_nii = save_file + 'FA' + save_tail + '.nii'
        sr_utility.ndarray_to_nifti(fa, fa_nii)
        return md_nii, fa_nii, md_std_nii
    else:
        dti_std = sr_utility.read_dt_volume(std_file)
        md_mean, md_std, fa_mean, fa_std = \
            sr_utility.mean_and_std_MD_FA(dti_mean[...,2:],
                                          dti_std[...,2:],
                                          no_samples=no_samples)

        md_nii = save_file + 'MD_mean' + save_tail + str(no_samples) + '.nii'
        md_std_nii = save_file + 'MD_std' + save_tail + str(no_samples) + '.nii'
        fa_nii = save_file + 'FA_mean'+save_tail + str(no_samples) + '.nii'
        fa_std_nii = save_file + 'FA_std'+save_tail + + str(no_samples) + '.nii'
        sr_utility.ndarray_to_nifti(md_mean, md_nii)
        sr_utility.ndarray_to_nifti(md_std, md_std_nii)
        sr_utility.ndarray_to_nifti(fa_mean, fa_nii)
        sr_utility.ndarray_to_nifti(fa_std, fa_std_nii)
        return md_nii, fa_nii



def _errors_MD_FA(md_nii, md_gt_nii, fa_nii, fa_gt_nii):
    # compute errors:
    dir, file = os.path.split(md_nii)
    error_md_file = os.path.join(dir, 'error_'+file)

    dir, file = os.path.split(fa_nii)
    error_fa_file = os.path.join(dir, 'error_' + file)

    sr_utility.compute_rmse_nii(nii_1=md_gt_nii, nii_2=md_nii,
                                save_file=error_md_file)

    sr_utility.compute_rmse_nii(nii_1=fa_gt_nii, nii_2=fa_nii,
                                save_file=error_fa_file)


# compute error on non-HCP dataset:
def nonhcp_reconstruct(opt, dataset_type='prisma'):
    # ------------------------ Reconstruct --------------------------------------:
    print('Method: %s' %
          (opt['method'], ))

    if not ('output_file_name' in opt):
        opt['output_file_name'] = 'dt_recon_b1000.npy'
    if not ('gt_header' in opt):
        opt['gt_header'] = 'dt_b1000_'

    # load parameters:
    recon_dir = opt['recon_dir']
    gt_dir = opt['gt_dir']
    # print('gt_dir is ...' + gt_dir)
    input_file_name = opt['input_file_name']

    # Load the input low-res DT image:
    print('... loading the low-res input DTI ...')
    print(os.path.join(gt_dir,input_file_name))
    dt_lowres = sr_utility.read_dt_volume(os.path.join(gt_dir,input_file_name),
                                          no_channels=opt['no_channels'])
    if not(dataset_type=='life' or dataset_type=='hcp1' or
           dataset_type=='hcp2' or dataset_type=='monkey'):
        dt_lowres = resize_DTI(dt_lowres,opt['upsampling_rate'])
    else:
        print('HCP dataset: no need to resample.')
    # clear the graph (is it necessary?)
    tf.reset_default_graph()

    # Reconstruct & save:
    nn_dir = name_network(opt)
    print('\nReconstruct high-res dti with the network: \n%s.' % nn_dir)

    output_file = os.path.join(recon_dir, nn_dir, 'dt_recon.npy')
    uncertainty_file = os.path.join(recon_dir, nn_dir, 'dt_std.npy')
    __, output_header = os.path.split(output_file)
    __, uncertainty_header = os.path.split(uncertainty_file)
    # print(output_file)
    # print(uncertainty_file)

    if not (os.path.exists(os.path.join(recon_dir, nn_dir))):
        os.makedirs(os.path.join(recon_dir, nn_dir))

    start_time = timeit.default_timer()
    if opt['method'] == 'cnn_heteroscedastic_variational' or \
       opt['method'] == 'cnn_heteroscedastic_variational_layerwise' or \
       opt['method'] == 'cnn_heteroscedastic_variational_channelwise' or \
       opt['method'] == 'cnn_heteroscedastic_variational_average' or \
       opt['method'] == 'cnn_heteroscedastic_variational_downsc' or \
       opt['method'] == 'cnn_heteroscedastic_variational_upsc' or \
       opt['method'] == 'cnn_heteroscedastic_variational_layerwise_downsc' or \
       opt['method'] == 'cnn_heteroscedastic_variational_channelwise_downsc' or \
       opt['method'] == 'cnn_heteroscedastic_variational_hybrid_control' or \
       opt['method'] == 'cnn_heteroscedastic_variational_channelwise_hybrid_control' or \
       opt['method'] == 'cnn_heteroscedastic_variational_downsc_control' or \
       opt['method'] == 'cnn_heteroscedastic_variational_upsc_control':

        uncertainty_file_d = os.path.join(recon_dir, nn_dir, 'dt_std_data.npy')
        uncertainty_file_m = os.path.join(recon_dir, nn_dir, 'dt_std_model.npy')

        __, output_header = os.path.split(output_file)
        __, uncertainty_header_d = os.path.split(uncertainty_file_d)
        __, uncertainty_header_m = os.path.split(uncertainty_file_m)

        print('... saving as %s' % output_file)
        dt_hr, dt_std_m, dt_std_d = super_resolve_mcdropout_heterovariational(dt_lowres, opt)
        np.save(output_file, dt_hr)
        np.save(uncertainty_file_d, dt_std_d)
        np.save(uncertainty_file_m, dt_std_m)

        # save as .nii
        print('\nSave each estimated dti/std separately as a nii file ...')
        sr_utility.save_as_nifti(output_header, os.path.join(recon_dir, nn_dir),
                                 gt_dir='', save_as_ijk=True,
                                 no_channels=opt['no_channels'],
                                 gt_header=opt['gt_header'])
        sr_utility.save_as_nifti(uncertainty_header_d, os.path.join(recon_dir, nn_dir),
                                 gt_dir='', save_as_ijk=True,
                                 no_channels=opt['no_channels'],
                                 gt_header=opt['gt_header'])
        sr_utility.save_as_nifti(uncertainty_header_m, os.path.join(recon_dir, nn_dir),
                                 gt_dir='', save_as_ijk=True,
                                 no_channels=opt['no_channels'],
                                 gt_header=opt['gt_header'])
    elif opt['method']=='cnn_heteroscedastic' or \
         opt['method']=='cnn_dropout' or \
         opt['method']=='cnn_gaussian_dropout' or \
         opt['method']=='cnn_variational_dropout' or \
         opt['method']=='cnn_variational_dropout_layerwise' or \
         opt['method']=='cnn_variational_dropout_channelwise' or \
         opt['method']=='cnn_variational_dropout_average':

        print('... saving as %s' % output_file)
        dt_hr, dt_std = super_resolve_mcdropout(dt_lowres, opt)
        np.save(output_file, dt_hr)
        np.save(uncertainty_file, dt_std)

        # save as .nii
        print('\nSave each estimated dti/std separately as a nii file ...')
        sr_utility.save_as_nifti(output_header, os.path.join(recon_dir, nn_dir),
                                 gt_dir='',
                                 save_as_ijk=True,
                                 no_channels=opt['no_channels'],
                                 gt_header=opt['gt_header'])
        sr_utility.save_as_nifti(uncertainty_header, os.path.join(recon_dir, nn_dir),
                                 gt_dir='', save_as_ijk=True,
                                 no_channels=opt['no_channels'],
                                 gt_header=opt['gt_header'])
    elif opt['method']=='cnn_simple':
        print('... saving as %s' % output_file)
        dt_hr = super_resolve(dt_lowres, opt)
        np.save(output_file, dt_hr)

        # save as .nii
        print('\nSave each estimated dti separately as a nii file ...')
        sr_utility.save_as_nifti(output_header, os.path.join(recon_dir, nn_dir),
                                 gt_dir='', save_as_ijk=True,
                                 no_channels=opt['no_channels'],
                                 gt_header=opt['gt_header'])

    else:
        raise ValueError('specified model not available')
    end_time = timeit.default_timer()
    print('\nIt took %f secs. \n' % (end_time - start_time))

# non-HCP data. Compute mean and std of FA.




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

#







