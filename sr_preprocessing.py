""" Preprocessing data:
We store the training data of high-res and low-res subvolumes as HDF5 files.
We use generate 'create_training_data()' to create training set. With scalability in mind, we devise the following
proceadure:

1. Extract a random subset of patches from each subject using 'extract_patches()'.
2. Save it in HDF5 format.
3. Merge all the HDF5 patch-libraries for all subjects into one larger patch-library 'merge_hdf5()'.


Here the output patch of shape (upsampling_rate * m, upsampling_rate * m, upsampling_rate * m, 6) is reshaped into
(m, m, m, 6*(upsampling_rate)**3) through the reverse 3D periodic shuffling. See function 'backward_periodic_shuffle()'.

The starndard 3D periodic shuffling (see eq.4 in MagicPony, CVPR 2016) is implemented in 'forward_periodic_shuffle()'.
"""

# todo: check the training data generation pipeline.
# todo: don't forget to apply the backward periodic shuffling.
# todo: randomised the mini-batch generation.


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
import random
import timeit


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


# Extract randomly patches from given list of patients and save as HDF5.
def create_training_data(data_parent_dir = '/Users/ryutarotanno/DeepLearning/Test_1/data/HCP',
                         data_subpath = 'T1w/Diffusion',
                         save_dir = '/Users/ryutarotanno/tmp',
                         subjects_list = ['992774'],
                         no_randomisation=1, sampling_rate=32,
                         b_value = 1000, upsampling_rate=2, receptive_field_radius=2, input_radius=5, no_channels=6):

    # subjects_list = ['992774', '125525', '205119', '133928', '570243', '448347', '654754', '153025']

    for subject in subjects_list:
        data_path = os.path.join(data_parent_dir, subject, data_subpath)
        highres_name = 'dt_b' + str(b_value) + '_'
        lowres_name = highres_name + 'lowres_' + str(upsampling_rate) + '_'
        input_library, output_library = extract_patches(data_dir=data_path,
                                                        highres_name=highres_name,
                                                        lowres_name=lowres_name,
                                                        upsampling_rate=upsampling_rate,
                                                        receptive_field_radius=receptive_field_radius,
                                                        input_radius=input_radius,
                                                        no_channels=no_channels,
                                                        sampling_rate=sampling_rate)

        filename = os.path.join(data_path, 'patchlibs_sr=32.h5')
        create_hdf5(filename, input_library=input_library, output_library=output_library)


def create_hdf5(filename, input_library, output_library):
    """ Save the given input and output patch library in HDF5 format. In the main training data generation script,
    create_training_data(), to reduce memory burden,  we first create patch library for each patient using this script
    and then merge them together using merge_hdf5().

    Args:
        filename (str): the full path and filename of the output hdf5. Make sure the filename has format '.h5'
        input_library (numpy array): 5D array of structures (batch, i, j, k, dti)
        output_library (numpy array): 5D array of structures (batch, i, j, k, dti)
    """
    shape_in = input_library.shape
    shape_out = output_library.shape

    f = h5py.File(filename, 'w-')
    f.create_dataset("input_lib", data=input_library,
                     maxshape=(None, shape_in(1), shape_in(2), shape_in(3), shape_in(4)))
    f.create_dataset("output_lib", data=output_library,
                     maxshape=(None, shape_out(1), shape_out(2), shape_out(3), shape_out(4)))
    f.close()



def merge_hdf5(global_filename, filenames_list):

    """ Merge patch-libraries (in HDF5 format) for individual subjects into a single h5 file.

    Args:
        global_filename (str) : the full path and filename of the final patch library.

    """

    # First compute the total number of training data points in all the selected files.
    no_data_input, no_data_output = 0, 0

    for file in filenames_list:
        f = h5py.File(file, 'r')
        input_lib = f["input_lib"]
        output_lib = f["output_lib"]
        no_data_input += input_lib.shape[0]
        no_data_output += output_lib.shape[0]
        shape_in, shape_out = input_lib.shape, output_lib.shape
        f.close()

    # Create a global H5 file setting the total length to the sum of all the files.
    if not(no_data_input == no_data_output):
        raise Warning("The number of data in in/ouput library don't match!!")
    else:
        g = h5py.File(global_filename, 'w-')
        g.create_dataset("input_lib", shape=(no_data_input, shape_in(1), shape_in(2), shape_in(3), shape_in(4)))
        g.create_dataset("output_lib", shape=(no_data_input, shape_out(1), shape_out(2), shape_out(3), shape_out(4)))

    # Sequentially fill the global h5 file with small h5 files in 'filenames_list'.
    start_idx = 0

    for idx, file in enumerate(filenames_list):
        start_time = timeit.default_timer()

        f = h5py.File(file, 'r')
        input_lib = f["input_lib"]
        output_lib = f["output_lib"]

        end_idx = start_idx + input_lib.shape[0]

        g["input_lib"][start_idx:end_idx, :, :, :, :] = input_lib
        g["output_lib"][start_idx:end_idx, :, :, :, :] = output_lib

        start_idx += input_lib.shape[0]
        f.close()

        end_time = timeit.default_timer()
        print("%i/%i subjects done. It took %f secs." % (idx, len(filenames_list), end_time - start_time))

    g.close()


# Load training data:
def load_and_shuffle_hdf5(filename, batch_size):
    f = h5py.File(filename, 'r')
    input_lib = f["input_lib"]
    output_lib = f["output_lib"]



# Extract corresponding patches from DTI volumes. Here we sample all patches centred at a foreground voxel.
def extract_patches(data_dir='/Users/ryutarotanno/DeepLearning/Test_1/data/',
                    highres_name='dt_b1000_', lowres_name='dt_b1000_lowres_2_',
                    upsampling_rate=2, receptive_field_radius=2, input_radius=5, no_channels=6,
                    sampling_rate=32):
    """
    Args:
        data_dir (str): the directory of the diffusion tensor images (DTIs) of a single patient
        highres_name:  the file name of the original DTIs
        lowres_name:  the file name of the downsampled DTIs
        upsampling_rate: the upsampling rate
        receptive_field_radius: the width of the receptive field is (2*receptive_field_radius + 1)
        input_radius: the width of the input patch is (2*input_radius + 1)
        no_channels (int) : the number of channels in each voxel
        sampling_rate (int): subsample on the usable patch pairs at rate 1/sampling_rate.

    Returns:
    """
    start_time = timeit.default_timer()

    # Define width of all patches for brevity:
    input_width, receptive_field_width = 2 * input_radius + 1, 2 * receptive_field_radius + 1
    output_radius = upsampling_rate * ((input_width - receptive_field_width + 1) // 2)

    # Load the original and down-sampled DTI volumes, and pad with zeros so all brain-voxel-centred pathces
    # are extractable.
    pad = max(input_radius, output_radius)  # padding width
    dti_highres = read_dt_volume(nameroot=os.path.join(data_dir, highres_name))
    dti_highres[:, :, :, 0] += 1  # adding 1 so brain voxels are valued 1 and background as zero.
    dti_highres = np.pad(dti_highres, pad_width=((pad, pad), (pad, pad), (pad, pad), (0, 0)),
                         mode='constant', constant_values=0)

    dti_lowres = read_dt_volume(nameroot=os.path.join(data_dir, lowres_name))
    dti_lowres[:, :, :, 0] += 1
    dti_lowres = np.pad(dti_lowres, pad_width=((pad, pad), (pad, pad), (pad, pad), (0, 0)),
                        mode='constant', constant_values=0)

    print("The size of HR/LR volumes are: %s and %s" % (dti_highres.shape, dti_lowres.shape))

    dim_x_highres, dim_y_highres, dim_z_highres, dim_dt = dti_highres.shape
    brain_mask = dti_highres[:, :, :, 0] == 1

    # Get all the indices of voxels in the brain and subsample:
    brain_indices = [(i, j, k) for i in xrange(dim_x_highres)
                               for j in xrange(dim_y_highres)
                               for k in xrange(dim_z_highres) if brain_mask[i, j, k] == True]

    brain_indices_subsampled = random.sample(brain_indices, len(brain_indices) // sampling_rate)
    print('number of effective patches = %i' % len(brain_indices))
    print('number of selected patches = %i' % len(brain_indices_subsampled))

    # Construct patch libraries:
    input_library = np.ndarray((len(brain_indices_subsampled),
                                2 * input_radius + 1,
                                2 * input_radius + 1,
                                2 * input_radius + 1,
                                no_channels), dtype='float64')

    output_library = np.ndarray((len(brain_indices_subsampled),
                                 2 * output_radius + upsampling_rate,
                                 2 * output_radius + upsampling_rate,
                                 2 * output_radius + upsampling_rate,
                                 no_channels), dtype='float64')

    for patch_idx, (i, j, k) in enumerate(brain_indices_subsampled):
        input_library[patch_idx, :, :, :, :] = \
        dti_lowres[(i - upsampling_rate * input_radius):(i + upsampling_rate * (input_radius + 1)):upsampling_rate,
                   (j - upsampling_rate * input_radius):(j + upsampling_rate * (input_radius + 1)):upsampling_rate,
                   (k - upsampling_rate * input_radius):(k + upsampling_rate * (input_radius + 1)):upsampling_rate, 2:]

        output_library[patch_idx, :, :, :, :] = \
        dti_highres[(i - output_radius): (i + output_radius + (upsampling_rate - 1) + 1),
                    (j - output_radius): (j + output_radius + (upsampling_rate - 1) + 1),
                    (k - output_radius): (k + output_radius + (upsampling_rate - 1) + 1), 2:]

    end_time = timeit.default_timer()

    print("It took %f secs." % (end_time - start_time))

    return input_library, output_library




# Periodic shuffling:
def forward_periodic_shuffle(patch, upsampling_rate = 2):
    """ This is the 3D extension of periodic shuffling (equation (4) in Magic Pony CVPR 2016).
    Args:
        patch (numpy array): 3 or 4 dimensional array with the last dimension being the dt components
        upsampling_rate (int): upsampling rate

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
def backward_periodic_shuffle(patch, upsampling_rate=2):
    """ Reverses the periodic shuffling defined by forward_periodic_shuffle()
    Args:
        patch (numpy array): 3 or 4 dimensional array with the last dimension
        upsampling_rate (int): upsampling rate
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