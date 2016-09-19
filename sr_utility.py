"""Utility functions used for loading/preprocessing/postprocessing data. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.cross_validation import train_test_split
import h5py
import os
import nibabel as nib


# Load in a DT volume .nii:
def read_dt_volume(nameroot='/Users/ryutarotanno/DeepLearning/Test_1/data/dt_b1000_lowres_2_'):
    for idx in np.arange(1, 9):
        data_path_new = nameroot + str(idx) + '.nii'
        print("... loading %s" % data_path_new)

        img = nib.load(data_path_new)
        data = img.get_data()
        data_array = np.zeros(data.shape)
        data_array[:] = data[:]

        if idx == 1:
            dti = np.zeros(data.shape + (8,))
            dti[:, :, :, idx-1] = data_array
        else:
            dti[:, :, :, idx - 1] = data_array

        del img, data, data_array
    return dti


# Select the patch-library and load into tensor shared variables:
def load_patchlib(
        patchlib='/Users/ryutarotanno/DeepLearning/Test_1/data/' + 'PatchLibsDiverseDS02_5x5_2x2_TS8_SRi032_0001.mat'):
    """ Split a patch library into trainign and validation sets and
    load them into theano.tensor shared variables
    :type patchlib: string
    :param patchlib: the path to the dataset)
    """
    #############
    # LOAD DATA #
    #############
    data_dir, data_file = os.path.split(patchlib)
    print('Loading %s' % data_file)
    f = h5py.File(patchlib)

    variables_tuple = f.items()
    print('\n Variables in m file include ...')
    for var in variables_tuple:
        name = var[0]
        data = var[1]
        print('   %s ' % name)  # Name

    # Assign variables:
    comipatchlib = variables_tuple[1][1].value
    comipatchlib = comipatchlib.T  # each row represents one input instance

    comopatchlib = variables_tuple[2][1].value
    comopatchlib = comopatchlib.T

    comipatchlib_train, comipatchlib_test, comopatchlib_train, comopatchlib_test \
        = train_test_split(comipatchlib, comopatchlib, test_size=.5)

    rval = (comipatchlib_train, comipatchlib_test, comopatchlib_train, comopatchlib_test)
    return rval


# Normalise training data:
def standardise_data(X_train, Y_train, option='default'):

    # Add ZCA whitening
    if option == 'default':
        # zero mean and unit variance for respective components:
        X_mean, Y_mean = X_train.mean(axis=0), Y_train.mean(axis=0)
        X_std, Y_std = X_train.std(axis=0), Y_train.std(axis=0)
        X_scaled, Y_scaled = (X_train - X_mean)/X_std, (Y_train - Y_mean)/Y_std

        rval = (X_scaled, X_mean, X_std, Y_scaled, Y_mean, Y_std)

    elif option == 'PCA-white':
        X_mean, Y_mean = X_train.mean(axis=0), Y_train.mean(axis=0)

        X_train -= X_mean  # zero-center the data
        X_cov = np.dot(X_train.T, X_train) / X_train.shape[0]  # get the data covariance matrix
        X_U, X_S, X_V = np.linalg.svd(X_cov)
        X_rot = np.dot(X_train, X_U)  # decorrelate the data

        Y_train -= Y_mean  # zero-center the data
        Y_cov = np.dot(Y_train.T, Y_train) / Y_train.shape[0]  # get the data covariance matrix
        Y_U, Y_S, Y_V = np.linalg.svd(Y_cov)
        Y_rot = np.dot(Y_train, Y_U)  # decorrelate the data

        # whiten the data: divide by the eigenvalues (which are square roots of the singular values)
        X_white = X_rot / np.sqrt(X_S + 1e-5)
        Y_white = Y_rot / np.sqrt(X_S + 1e-5)

        rval = (X_white, X_mean, X_U, X_S, Y_white, Y_mean, Y_U, Y_S,)

    else:
        print("The chosen standadization method not available")

    return rval


# Save each estimated dti separately as a nifti file for visualisation
def save_as_nifti(recon_file='mlp_h=1_highres_dti.npy',
                  recon_dir='/Users/ryutarotanno/DeepLearning/Test_1/recon/',
                  gt_dir='/Users/ryutarotanno/DeepLearning/Test_1/data/'):
    """Save each estimated dti separately as a nifti file for visualisation.
    Args:
        recon_file: file name of estimated DTI volume (4D numpy array)
        recon_dir: directory name that contains recon_file
        gt_dir: directory name that contains the ground truth high-res DTI.
    """
    dt_est = np.load(os.path.join(recon_dir, recon_file))  # load the estimated DTI volume
    base, ext = os.path.splitext(recon_file)

    for k in np.arange(6):
        # Save each DT component separately as a nii file:
        dt_gt = nib.load(os.path.join(gt_dir, 'dt_b1000_' + str(k + 3) + '.nii'))  # get the GT k+1 th dt component.
        affine = dt_gt.get_affine()  # fetch its affine transfomation
        header = dt_gt.get_header()  # fetch its header
        img = nib.Nifti1Image(dt_est[:-1, :, :-1, k + 2], affine=affine, header=header)

        print('... saving estimated ' + str(k + 1) + ' th dt element')
        nib.save(img, os.path.join(recon_dir, base + '_' + str(k + 3) + '.nii'))


# Compute reconsturction error:

def compute_rmse(recon_file='mlp_h=1_highres_dti.npy',
                 recon_dir='/Users/ryutarotanno/DeepLearning/nsampler/recon',
                 gt_dir='/Users/ryutarotanno/DeepLearning/Test_1/data'):
    """Compute root mean square error wrt the ground truth DTI.
     Args:
        recon_file: file name of estimated DTI volume (4D numpy array)
        recon_dir: directory name that contains recon_file
        gt_dir: directory name that contains the ground truth high-res DTI.
    Returns:
        reconstruction error (RMSE)
    """
    dt_gt = read_dt_volume(nameroot=os.path.join(gt_dir, 'dt_b1000_'))
    dt_est_tmp = np.load(os.path.join(recon_dir, recon_file))
    dt_est = dt_est_tmp[:-1, :, :-1, :]
    mask = dt_est[:, :, :, 0] == 0  # mask out the background voxels
    mse = np.sum(((dt_gt[:, :, :, 2:] - dt_est[:, :, :, 2:]) ** 2) * mask[..., np.newaxis]) / (mask.sum() * 6.0)
    return np.sqrt(mse)


def name_network(method='linear', n_h1=None, n_h2=None, n_h3=None, cohort='Diverse', no_subjects=8, sample_rate=32, us=2, n=2, m=2,
                 optimisation_method='standard', dropout_rate=0.0):
    """given inputs, return the model name."""

    if method == 'linear':
        nn_file = '%s_%i-%i_opt=%s_drop=%s_%sDS%02i_in=%i_out=%i_TS%i_SRi%03i' % \
                  (method, 6 * (2 * n + 1) ** 3, 6 * m ** 3, optimisation_method, str(dropout_rate), cohort,
                   us, 2 * n + 1, m, no_subjects, sample_rate)
    elif method == 'mlp_h=1':
        nn_file = '%s_%i-%i-%i_opt=%s_drop=%s_%sDS%02i_in=%i_out=%i_TS%i_SRi%03i' % \
                  (method, 6 * (2 * n + 1) ** 3, n_h1, 6 * m ** 3, optimisation_method, str(dropout_rate), cohort,
                   us, 2 * n + 1, m, no_subjects, sample_rate)
    elif method == 'mlp_h=2':
        nn_file = '%s_%i-%i-%i-%i_opt=%s_drop=%s_%sDS%02i_in=%i_out=%i_TS%i_SRi%03i' % \
                  (method, 6 * (2 * n + 1) ** 3, n_h1, n_h2, 6 * m ** 3, optimisation_method, str(dropout_rate), cohort,
                   us, 2 * n + 1, m, no_subjects, sample_rate)
    elif method == 'mlp_h=3':
        nn_file = '%s_%i-%i-%i-%i-%i_opt=%s_drop=%s_%sDS%02i_in=%i_out=%i_TS%i_SRi%03i' % \
                  (method, 6 * (2 * n + 1) ** 3, n_h1, n_h2, n_h3, 6 * m ** 3, optimisation_method, str(dropout_rate),
                   cohort, us, 2 * n + 1, m, no_subjects, sample_rate)
    else:
        raise
    return nn_file