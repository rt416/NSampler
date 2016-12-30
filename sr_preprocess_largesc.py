"""Load and preprocessing script """
import os
import sys

import cPickle as pkl
import h5py
import numpy as np
import tensorflow as tf

def load_hdf5(opt):
    cohort = opt['cohort']
    no_subjects =opt['no_subjects']
    subsampling_rate = opt['subsampling_rate']
    upsampling_rate = opt['upsampling_rate']
    receptive_field_radius = opt['receptive_field_radius']
    input_radius = opt['input_radius']
    output_radius = opt['output_radius']

    data_dir = opt['data_dir']
    fstr = 'PatchLibs_%s_Upsample%02i_Input%02i_Recep%02i_TS%i_Subsample%03i_001.h5'
    filename = data_dir + fstr % (cohort, upsampling_rate,
                                  2*input_radius+1,
                                  2*receptive_field_radius+1,
                                  no_subjects, subsampling_rate)

    # {'in': {'train': <raw_data>, 'valid': <raw_data>,
    # 	'mean': <mean>, 'std': <std>}, 'out' : {...}}

    if not(os.path.exists(filename)):
        print('The specified training set does not exist. Create ...')
    else:
        f = h5py.File(filename, 'r')
    data = {}

    print("Loading %s" % (filename,))
    for i in ['in','out']:
        print("\tLoading input and stats")
        data[i] = {}
        X = f[i+"put_lib"]
        xsh = X.shape[0]
        n_train = int((1.-opt['validation_fraction'])*xsh)
        n_valid = xsh-n_train

        data[i]['X'] = X
        data[i]['mean'], data[i]['std'] = moments(opt, data[i]['X'], n_train)

    # Save the transforms used for data normalisation:
    print('\tSaving transforms for data normalization for test time')
    transform = {'input_mean': data['in']['mean'],
                 'input_std': data['in']['std'],
                 'output_mean': data['out']['mean'],
                 'output_std': data['out']['std']}
    with open(os.path.join(opt['checkpoint_dir'], 'transforms.pkl'), 'w') as fp:
        pkl.dump(transform, fp, protocol=pkl.HIGHEST_PROTOCOL)
    return data, n_train, n_valid


def moments(opt, x, n_train):
    """Per-element whitening on the training set"""
    transform_opt = opt['transform_opt']
    if transform_opt=='standard':
        mean, std = mega_moments(x, n_train)
        # mean = np.mean(x, axis=0, keepdims=True)
        # std = np.std(x, axis=0, keepdims=True)
    elif transform_opt=='scaling':
        mean, std = 0., 1e-4
        # shape = np.mean(x, axis=0, keepdims=True).shape
        # mean = np.zeros(shape)
        # std = 1e-4*np.ones(shape)
    return mean, std


def dict_whiten(data, field1, idx):
    """ Whiten the data at indices idx, under field """
    x = data[field1]['X'][idx]
    return diag_whiten(x, mean=data[field1]['mean'], std=data[field1]['std'])


def diag_whiten(x, mean=0., std=1.):
    """Whiten on a per-pixel basis"""
    return (x - mean)/std


def mega_moments(x, n_train, chunk_size=1000, axis=0):
    """Compute moments of very large matrix"""
    print('\tComputing moments of massive matrix')
    n_chunks = int(np.ceil((n_train*1.)/chunk_size))

    sum_x = 0
    sum_x2 = 0

    for i in xrange(n_chunks):
        sys.stdout.write('\tChunk progress: %d/%d\r' % (i+1,n_chunks))
        sys.stdout.flush()

        current_size = np.minimum((i+1)*chunk_size,n_train) - i*chunk_size
        chunk = x[i*chunk_size:i*chunk_size+current_size,...]

        sum_x  += 1.*np.sum(chunk, axis=axis)
        sum_x2 += 1.*np.sum(chunk**2, axis=axis)

    mean = sum_x/n_train
    var = (sum_x2 - 2*mean*sum_x + n_train*mean**2)/n_train

    return (mean, np.sqrt(var))


