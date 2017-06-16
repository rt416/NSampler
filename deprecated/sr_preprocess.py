''' Load and preprocessing script '''

import os
import sys

import cPickle as pkl
import h5py
import numpy as np
import sr_datageneration
import tensorflow as tf

def load_hdf5(opt):
    cohort = opt['cohort']
    no_subjects =opt['no_subjects']
    subsampling_rate = opt['subsampling_rate']
    upsampling_rate = opt['upsampling_rate']
    receptive_field_radius = opt['receptive_field_radius']
    input_radius = opt['input_radius']
    output_radius = opt['output_radius']
    patchlib_idx = opt['patchlib_idx']

    data_dir = opt['data_dir']
    fstr = 'PatchLibs_%s_Upsample%02i_Input%02i_Recep%02i_TS%i_Subsample%03i_%03i.h5'

    filename = data_dir + fstr % (cohort, upsampling_rate,
                                  2*input_radius+1,
                                  2*receptive_field_radius+1,
                                  no_subjects, subsampling_rate,
                                  patchlib_idx)

    # {'in': {'train': <raw_data>, 'valid': <raw_data>,
    # 	'mean': <mean>, 'std': <std>}, 'out' : {...}}

    if not(os.path.exists(filename)):
        print('The specified training set does not exist. Create ...')
        sr_datageneration.create_training_data(opt)
        print('Done!')
        f = h5py.File(filename, 'r')
    else:
        f = h5py.File(filename, 'r')

    data = {}

    print("Loading %s" % (filename,))
    for i in ['in','out']:
        print("\tLoading input and stats")
        data[i] = {}
        X = f[i+"put_lib"]
        xsh = X.shape[0]
        num_train = int((1.-opt['validation_fraction'])*xsh)

        data[i]['train'] = X[:num_train,...]
        data[i]['valid'] = X[num_train:,...]
        data[i]['mean'], data[i]['std'] = moments(opt, data[i]['train'])

    # Save the transforms used for data normalisation:
    print('\tSaving transforms for data normalization for test time')
    transform = {'input_mean': data['in']['mean'],
                 'input_std': data['in']['std'],
                 'output_mean': data['out']['mean'],
                 'output_std': data['out']['std']}
    with open(os.path.join(opt['checkpoint_dir'], 'transforms.pkl'), 'w') as fp:
        pkl.dump(transform, fp, protocol=pkl.HIGHEST_PROTOCOL)
    return data


def moments(opt, x):
    """Per-element whitening on the training set"""
    transform_opt = opt['transform_opt']
    if transform_opt=='standard':
        mean = np.mean(x, axis=0, keepdims=True)
        std = np.std(x, axis=0, keepdims=True)
    elif transform_opt=='scaling':
        shape = np.mean(x, axis=0, keepdims=True).shape
        mean = np.zeros(shape)
        std = 1e-4*np.ones(shape)
    elif transform_opt == 'none':
        shape = np.mean(x, axis=0, keepdims=True).shape
        mean = np.zeros(shape)
        std = np.ones(shape)
    return mean, std

def dict_whiten(data, field1, field2, idx):
	"""Whiten the data at indices idx, under field"""
	x = data[field1][field2][idx]
	return diag_whiten(x, mean=data[field1]['mean'], std=data[field1]['std'])

def diag_whiten(x, mean=0., std=1.):
	"""Whiten on a per-pixel basis"""
	return (x - mean)/std























