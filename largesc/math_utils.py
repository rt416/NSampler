from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from debug import set_trace



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



