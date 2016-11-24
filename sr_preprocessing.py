""" Preprocessing data """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle
import tensorflow as tf
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


# Extract corresponding patches from DTI volumes. Here we sample all patches centred at a foreground voxel.
def extract_patches(dti_dir='/Users/ryutarotanno/DeepLearning/Test_1/data/',
                    highres_name='dt_b1000_', lowres_name='dt_b1000_lowres_2_',
                    upsampling_rate=2, receptive_field_radius = 2, input_radius = 5, no_channels = 6):
    """
    Args:
        dti_dir (str): the directory of the diffusion tensor images (DTIs) of a single patient
        highres_name:  the file name of the original DTIs
        lowres_name:  the file name of the downsampled DTIs
        upsampling_rate: the upsampling rate
        receptive_field_radius: the width of the receptive field is (2*receptive_field_radius + 1)
        input_radius: the width of the input patch is (2*input_radius + 1)
        no_channels (int) : the number of channels in each voxel

    Returns:
    """
    # Load the original and down-sampled DTI volumes:
    dti_highres = read_dt_volume(nameroot=os.path.join(dti_dir, highres_name))
    dti_lowres = read_dt_volume(nameroot=os.path.join(dti_dir, lowres_name))
    # dti_lowres = dti_lowres[0::us, 0::us, 0::us, :]

    print("The size of HR/LR volumes are: %s and %s", (dti_highres.shape, dti_lowres.shape))

    dim_x_highres, dim_y_highres, dim_z_highres, dim_dt = dti_highres.shape
    brain_mask = dti_highres[:, :, :, 0] == 0

    # Get all the indices of voxels in the brain:
    brain_indices = [(i, j, k) for i in xrange(dim_x_highres)
                               for j in xrange(dim_y_highres)
                               for k in xrange(dim_z_highres) if brain_mask[i, j, k] == 0]

    # Construct patch libraries:
    input_width, receptive_field_width = 2 * input_radius + 1, 2 * receptive_field_radius + 1
    output_radius = upsampling_rate * ((input_width - receptive_field_width + 1)/2)

    input_library = np.ndarray((len(brain_indices),
                                2 * input_radius + 1,
                                2 * input_radius + 1,
                                2 * input_radius + 1,
                                no_channels), dtype='float64')

    output_library = np.ndarray((len(brain_indices),
                                 2 * output_radius + upsampling_rate,
                                 2 * output_radius + upsampling_rate,
                                 2 * output_radius + upsampling_rate,
                                 no_channels), dtype='float64')

    for patch_idx, (i, j, k) in enumerate(brain_indices):
        input_library[patch_idx, :, :, :, :] = \
        dti_lowres[(i - upsampling_rate * input_radius):(i + upsampling_rate * (input_radius + 1)):upsampling_rate,
                   (j - upsampling_rate * input_radius):(j + upsampling_rate * (input_radius + 1)):upsampling_rate,
                   (k - upsampling_rate * input_radius):(k + upsampling_rate * (input_radius + 1)):upsampling_rate, 2:]

        output_library[patch_idx, :, :, :, :] = \
        dti_highres[(i - output_radius): (i + output_radius + (upsampling_rate - 1) + 1),
                    (j - output_radius): (j + output_radius + (upsampling_rate - 1) + 1),
                    (k - output_radius): (k + output_radius + (upsampling_rate - 1) + 1), 2:]

        # reshuffle:
        dti_highres

    return input_library, output_library, brain_indices


# Periodic shuffling:
def forward_periodic_shuffle(patch, upsampling_rate = 2, no_channels = 6):
    """
    Args:
        patch (numpy array): 3 or 4 dimensional array with the last dimension

    Returns:

    """
    if patch.ndim == 3:
        if patch.shape[2] == (upsampling_rate ** 2):
            dim_i, dim_j, dim_filters = patch.shape
            # apply periodic shuffling:
            patch_ps = np.ndarray((dim_i * upsampling_rate,
                                  dim_j * upsampling_rate),
                                  dtype='float64')

            for (i, j) in [(i, j) for i, j in np.ndindex(patch_ps.shape)]:
                patch_ps[i, j] = patch[i // upsampling_rate,
                                       j // upsampling_rate,
                                       np.mod(i, upsampling_rate) +
                                       np.mod(j, upsampling_rate) * upsampling_rate]

        else:
            dim_i, dim_j, dim_filters = patch.shape

            # apply periodic shuffling:
            patch_ps = np.ndarray((dim_i * upsampling_rate,
                                   dim_j * upsampling_rate,
                                   dim_filters / (upsampling_rate**2)), dtype='float64')

            for (i, j, c) in [(i, j, c) for i, j, c in np.ndindex(patch_ps.shape)]:
                patch_ps[i, j, c] = patch[i // upsampling_rate,
                                          j // upsampling_rate,
                                          np.mod(i, upsampling_rate) +
                                          np.mod(j, upsampling_rate) * upsampling_rate +
                                          c * (upsampling_rate**2 - 1)]
    elif patch.ndim == 4:
        dim_i, dim_j, dim_k, dim_filters = patch.shape

        # apply periodic shuffling:
        patch_ps = np.ndarray((dim_i * upsampling_rate,
                               dim_j * upsampling_rate,
                               dim_j * upsampling_rate,
                               dim_filters / (upsampling_rate ** 3)), dtype='float64')

        for (i, j, k, c) in [(i, j, k, c) for i, j, k, c in np.ndindex(patch_ps.shape)]:
            patch_ps[i, j, k, c] = patch[i // upsampling_rate,
                                         j // upsampling_rate,
                                         k // upsampling_rate,
                                         np.mod(i, upsampling_rate) +
                                         np.mod(j, upsampling_rate) * upsampling_rate +
                                         np.mod(k, upsampling_rate) * (upsampling_rate**2) +
                                         c * (upsampling_rate**3 - 1)]
    return patch_ps

# Reverse forward shuffling:
def backward_periodic_shuffle(patch, upsampling_rate = 2, no_channels = 6):
    """
    Args:
        patch (numpy array): 3 or 4 dimensional array with the last dimension

    Returns:

    """
    if patch.ndim == 2:
        dim_i, dim_j = patch.shape

        # apply periodic shuffling:
        patch_bps = np.ndarray((dim_i / upsampling_rate,
                                dim_j / upsampling_rate,
                                upsampling_rate ** 2), dtype='float64')

        for (i, j, c) in [(i, j, c) for i, j, c in np.ndindex(patch_bps.shape)]:
            patch_bps[i, j, c] = patch[upsampling_rate * i + np.mod(c, upsampling_rate),
                                       upsampling_rate * j + np.mod(c // upsampling_rate, upsampling_rate)]

    elif patch.ndim == 3:
        dim_i, dim_j, dim_filters = patch.shape

        # apply periodic shuffling:
        patch_bps = np.ndarray((dim_i / upsampling_rate,
                                dim_j / upsampling_rate,
                                dim_filters * (upsampling_rate**2)), dtype='float64')

        for (i, j, c) in [(i, j, c) for i, j, c in np.ndindex(patch_bps.shape)]:
            patch_bps[i, j, c] = patch[upsampling_rate * i + np.mod(c, upsampling_rate),
                                       upsampling_rate * j + np.mod(c // upsampling_rate, upsampling_rate),
                                       c // (upsampling_rate**2)]
    elif patch.ndim == 4:
        dim_i, dim_j, dim_k, dim_filters = patch.shape

        # apply periodic shuffling:
        patch_bps = np.ndarray((dim_i / upsampling_rate,
                                dim_j / upsampling_rate,
                                dim_k / upsampling_rate,
                                dim_filters * (upsampling_rate ** 3)), dtype='float64')

        for (i, j, k, c) in [(i, j, k, c) for i, j, k, c in np.ndindex(patch_bps.shape)]:
            patch_bps[i, j, k, c] = patch[upsampling_rate * i + np.mod(c, upsampling_rate),
                                          upsampling_rate * j + np.mod(c // upsampling_rate, upsampling_rate),
                                          upsampling_rate * k + np.mod(c // upsampling_rate**2, upsampling_rate),
                                          c // (upsampling_rate ** 3)]
    return patch_bps
















if __name__ == "__main__":
    read_dt_volume()
    extract_patches()