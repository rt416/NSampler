from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np



def axisAngle_from_rotmat(R):
    angle = np.arccos(( R[0,0] + R[1,1] + R[2,2] - 1)/2)
    x = (R[2,1] - R[1,2])/np.sqrt((R[2,1] - R[1,2])**2+(R[0,2] - R[2,0])**2+(R[1,0] - R[0,1])**2)
    y = (R[0,2] - R[2,0])/np.sqrt((R[2,1] - R[1,2])**2+(R[0,2] - R[2,0])**2+(R[1,0] - R[0,1])**2)
    z = (R[1,0] - R[0,1])/np.sqrt((R[2,1] - R[1,2])**2+(R[0,2] - R[2,0])**2+(R[1,0] - R[0,1])**2)

    return angle, np.array([x, y, z])



def unique_rows(A):
    return np.unique(A.view(np.dtype((np.void, 
            A.dtype.itemsize*A.shape[1])))).view(A.dtype).reshape(-1, A.shape[1])



def entropy(counts):
    '''Compute entropy.'''
    ps = counts/float(np.sum(counts))  # coerce to float and normalize
    ps = ps[np.nonzero(ps)]            # toss out zeros
    H = -np.sum(ps * np.log2(ps))   # compute entropy
    return H



def MI(x, y):
    '''Compute mutual information'''
    x = x.flatten()
    y = y.flatten()
    assert len(x) == len(y)
    bins = np.sqrt(len(x))
    counts_xy = np.histogram2d(x, y, bins=bins)[0]
    counts_x  = np.histogram(  x,    bins='sqrt')[0]
    counts_y  = np.histogram(  y,    bins='sqrt')[0]
    
    H_xy = entropy(counts_xy)
    H_x  = entropy(counts_x)
    H_y  = entropy(counts_y)
    return H_x + H_y - H_xy


def forward_periodic_shuffle(patch, upsampling_rate=2):
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

            # Apply reverse shuffling (optional):
            shuffle_indices = [(i, j)
                               for j in xrange(upsampling_rate)
                               for i in xrange(upsampling_rate)]

            no_channels = dim_filters / (upsampling_rate ** 3)

            for (i, j) in shuffle_indices:
                patch_ps[i::upsampling_rate,
                         j::upsampling_rate] \
                    = patch[:, :, np.mod(i, upsampling_rate) +
                                  np.mod(j, upsampling_rate) * upsampling_rate]

        else:
            dim_i, dim_j, dim_filters = patch.shape

            # apply periodic shuffling:
            patch_ps = np.ndarray((dim_i * upsampling_rate,
                                   dim_j * upsampling_rate,
                                   dim_filters / (upsampling_rate**2)), dtype='float64')

            shuffle_indices = [(i, j)
                               for j in xrange(upsampling_rate)
                               for i in xrange(upsampling_rate)]

            no_channels = dim_filters / (upsampling_rate ** 2)

            for c in xrange(dim_filters // (upsampling_rate ** 2)):
                for (i, j) in shuffle_indices:
                    patch_ps[i::upsampling_rate,
                             j::upsampling_rate,
                             c] = patch[:, :, np.mod(i, upsampling_rate) +
                                              np.mod(j, upsampling_rate) * upsampling_rate +
                                              c * (upsampling_rate**2)]

    elif patch.ndim == 4:
        dim_i, dim_j, dim_k, dim_filters = patch.shape

        # apply periodic shuffling:
        patch_ps = np.ndarray((dim_i * upsampling_rate,
                               dim_j * upsampling_rate,
                               dim_j * upsampling_rate,
                               dim_filters // (upsampling_rate ** 3)), dtype='float64')

        shuffle_indices = [(i, j, k) for k in xrange(upsampling_rate)
                                     for j in xrange(upsampling_rate)
                                     for i in xrange(upsampling_rate)]

        no_channels = dim_filters / (upsampling_rate ** 3)

        for c in xrange(dim_filters // (upsampling_rate ** 3)):
            for (i, j, k) in shuffle_indices:
                patch_ps[i::upsampling_rate, j::upsampling_rate, k::upsampling_rate, c] \
                    = patch[:, :, :, np.mod(i, upsampling_rate) +
                                     np.mod(j, upsampling_rate) * upsampling_rate +
                                     np.mod(k, upsampling_rate) * (upsampling_rate**2) +
                                     c * (upsampling_rate**3)]

    elif patch.ndim == 5:  # apply periodic shuffling to a batch of examples.
        batch_size, dim_i, dim_j, dim_k, dim_filters = patch.shape

        # Apply reverse shuffling (optional):
        shuffle_indices = [(i, j, k) for k in xrange(upsampling_rate)
                                     for j in xrange(upsampling_rate)
                                     for i in xrange(upsampling_rate)]

        patch_ps = np.ndarray((batch_size,
                               dim_i * upsampling_rate,
                               dim_j * upsampling_rate,
                               dim_j * upsampling_rate,
                               dim_filters // (upsampling_rate ** 3)), dtype='float64')

        no_channels = dim_filters // (upsampling_rate ** 3)

        for c in xrange(dim_filters // (upsampling_rate ** 3)):
            for (i, j, k) in shuffle_indices:
                patch_ps[:, i::upsampling_rate, j::upsampling_rate, k::upsampling_rate, c] \
                    = patch[:, :, :, :, np.mod(i, upsampling_rate) +
                                        np.mod(j, upsampling_rate) * upsampling_rate +
                                        np.mod(k, upsampling_rate) * (upsampling_rate**2) +
                                        c * (upsampling_rate**3)]
    return patch_ps

