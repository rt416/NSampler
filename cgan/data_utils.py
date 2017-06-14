from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import nibabel as nib
import largesc.math_utils as mu
import os as os
import sys as sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil
import random

# from debug import set_trace


def gendata_centroid(mask, bgval=0):
    """
    Generates the centroid based coordinate images for the input image.

    Args:
        mask (3D image): masked foreground voxels.
        bgval (float): background value outside mask.

    Returns:
        xyz (4D image): where each voxel contains the xyz coordinates in the
                        centroid space. Voxels outside the mask are left zero.

    """
    ijk = np.where(mask!=bgval)
    mui = np.mean(ijk[0])
    muj = np.mean(ijk[1])
    muk = np.mean(ijk[2])    
    cijk = np.array(ijk,dtype=float).T
    cijk[:,0] -= mui
    cijk[:,1] -= muj
    cijk[:,2] -= muk
    X = np.matmul(cijk.T,cijk)
    V,R = np.linalg.eigh(X)
    S = np.diag(1/np.sqrt(V))
    T = np.matmul(S,R.T)
    cijk = np.matmul(T,cijk.T).T
    sh = mask.shape
    xyz = np.zeros((sh[0],sh[1],sh[2],3))
    for i in range(0,3):
        o1s = np.ones(ijk[0].shape,dtype=int) * i
        coord = (ijk[0],ijk[1],ijk[2],o1s)
        xyz[coord] = cijk[:,i]
    return xyz,V,R,S,T,ijk,cijk



def gendata_coordinate(mask, bgval=0):
    """
    Generates the coordinates data of images for the input image.

    Args:
        mask (3D image): masked foreground voxels.
        bgval (float): background value outside mask.

    Returns:
        xyz (4D image): where each voxel contains the xyz coordinates in the
                        coordinate space. Voxels outside the mask are left zero.

    """
    ijk = np.where(mask!=bgval)
    sh = mask.shape
    xyz = np.zeros((sh[0],sh[1],sh[2],3))
    for i in range(0,3):
        o1s = np.ones(ijk[0].shape,dtype=int) * i
        coord = (ijk[0],ijk[1],ijk[2],o1s)
        xyz[coord] = ijk[i]
    return xyz



def prog_epoch(mesg, step, stepval):
    """
    Print the progress message in place.

    Args: 
        mesg (str): progress message
        step (int): step=stepval for first call, after that step>stepval
        stepval (int): first value of step
    """
    update = mesg + "                          "
    print(update, end="\r")
    sys.stdout.flush()



def prog(perc, step, stepval, mesg=''):
    """
    Print the progress percentage in place.

    Args: 
        perc (number): Percentage to print.
        step (int): step=stepval for first call, after that step>stepval
        stepval (int): first value of step
        mesg (str): any short message
    """
    update = "{:.1f}%: {}                          ".format(perc, mesg)
    print(update, end="\r")
    sys.stdout.flush()



def sanitise_imgdata(imgdat, val=0., neg=False, nan=True, inf=True):
    """
    Removes, negative, nan, inf value voxels from an image and set to the
    requested value. The change is done in place.

    Args:
        imgdat (np.ndarray): the image data.
        val (float): the default voxel value.
        neg (bool): if true set all voxels with neg values to val.
        nan (bool): if true set all voxels with nan values to val.
        inf (bool): if true set all voxels with inf values to val.

    """
    if nan:
        imgdat[np.isnan(imgdat)] = val
    if inf:
        imgdat[np.isinf(imgdat)] = val
    if neg:
        imgdat[imgdat<0] = val



def show_slices(slices, sz=0, sz_2=0, figsize=(6,6)):
    """ Function to display row of image slices
    Args:
        slices (list): list of 2d numpy arrays alternating between input/output
        sz (int): radius of the central rectangle in LR space
        sz_hr (int): radius of the cen
    """
    fig, axes = plt.subplots(1, len(slices), figsize=figsize)
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
        if sz>0 and sz_2>2:
            bgz = slice.shape[0]//2
            sz_tmp = sz if i%2==0 else sz_2
            off = bgz - sz_tmp
            axes[i].add_patch(patches.Rectangle(
                    (off, off), 2*sz_tmp+1, 2*sz_tmp+1, fill=False, edgecolor='red'))
    fig.show()



def image_subsample(dat, mask, ds=2, bgval=0):
    """
    Subsamples the image by average ds voxels. Retains the original
    resolution of the input image. Subsampling is done in place.

    Args:
        dat (np.ndarray): 3D or 4D image.
        mask (np.ndarray): 3D image of masked voxels.
        ds (int): Subsampling rate.
        bgval (float): background value outside mask.

    """
    ijk = np.argwhere(mask!=bgval)
    cop = np.copy(dat)
    if len(dat.shape)==3:
        for i,j,k in ijk: 
            bmask = mask[i:(i+ds),j:(j+ds),k:(k+ds)]
            if bgval not in bmask:
                dat[i,j,k]=np.mean(cop[i:(i+ds),j:(j+ds),k:(k+ds)])
    elif len(dat.shape)==4:
        for i,j,k in ijk: 
            bmask = mask[i:(i+ds),j:(j+ds),k:(k+ds)]
            if bgval not in bmask:
                dat[i,j,k,:]=np.mean(cop[i:(i+ds),j:(j+ds),k:(k+ds),:])
    else:
        raise ValueError('Only 3D or 4D images are handled')



def image_subsample2(dat, mask, ds=2, bgval=0):
    """
    Subsamples the image by average ds voxels. Retains the original
    resolution of the input image. Subsampling is done in place.

    Args:
        dat (np.ndarray): 3D or 4D image.
        mask (np.ndarray): 3D image of masked voxels.
        ds (int): Subsampling rate.
        bgval (float): background value outside mask.

    """
    ijk = np.argwhere(mask!=bgval)
    cop = np.copy(dat)
    if len(dat.shape)==3:
        for i,j,k in ijk: 
            dat[i,j,k]=np.mean(cop[i:(i+ds),j:(j+ds),k:(k+ds)])
    elif len(dat.shape)==4:
        for i,j,k in ijk: 
            dat[i,j,k,:]=np.mean(cop[i:(i+ds),j:(j+ds),k:(k+ds),:])
    else:
        raise ValueError('Only 3D or 4D images are handled')



def load_series_nii(namepat, series=[], dtype='float32'):
    """
    Loads a series of NIFTI files. For example:
        file_01.nii, file_02.nii ...

    Args:
        namepat (string): Generic file name with format pattern
        series (range): Filenames to load would be:
                        for i in series:
                            filename = namepat.format(i)
                        if series=[], assumes namepat is the filename (single image).
        dtype (string): default float32 (GPU)

    Returns:
        img (np.array): image array with series loaded in the 4th dim
        hdr (nifti header): Contains the meta data of the nii image
    """
    print ('Converting to:', dtype)
    if not series:
        print ('Loading:', namepat)
        nii = nib.load(namepat)
        img = nii.get_data().astype(dtype)
        hdr = nii.get_header()
    elif len(series) == 1:
        filename = namepat.format(series[0])
        print ('Loading single channel:', filename)
        nii = nib.load(filename)
        img = nii.get_data().astype(dtype)
        hdr = nii.get_header()
    else:
        filename = namepat.format(series[0])
        print ('Loading:', filename, 0)
        nii = nib.load(filename)
        dat = nii.get_data().astype(dtype)
        hdr = nii.get_header()
        shplen = len(dat.shape)
        shape = [0,] * (shplen + 1)
        for i in range(shplen):
            shape[i] = dat.shape[i]
        shape[shplen] = len(series)
        img = np.zeros(shape)
        img[..., 0] = dat
        cnt = 1
        for i in series[1:]:
            filename = namepat.format(i)
            print ('Loading:', filename, cnt)
            nii = nib.load(filename)
            img[..., cnt] = nii.get_data().astype(dtype)
            cnt += 1
    return img, hdr



def write_series_nii(namepat, img, hdr=None, series=[], dtype='float32'):
    """
    Writes a series of NIFTI files using the last dimension of the img data
    for the series. Assumes the hdr information is correct and appropriate.

    Args:
        namepat (string): Generic file name with format pattern
        img (np.array): ND array, like output of load_series_nii
        hdr (nifti header): like out put of load_sereis_nii
        series (range): Filenames to write would be:
                        for i in series:
                            filename = namepat.format(i)
                        if series=[], assumes namepat is the filename (single image).
        dtype (string): default float32
    """
    print ('Converting to:', dtype)
    img = img.astype(dtype)
    if hdr is None:
        aff = np.diag([1, 1, 1, 1])
    else:
        aff = hdr.get_best_affine()
    cnt = 0
    if not series:
        print('Writing:', namepat)
        nii = nib.Nifti1Image(img, aff)
        nii.to_filename(namepat)
    else:
        for i in series:
            filename = namepat.format(i)
            print('Writing:', filename, cnt)
            nii = nib.Nifti1Image(img[..., cnt], aff)
            cnt += 1
            nii.to_filename(filename)



def get_root_pepys():
    """
    Returns: root as string
    """
    r1 = '/Users/aghosh/Data/ClusterDRIVE01/hcp/Auro/'
    r2 = '/home/aghosh/Data/ClusterDRIVE/' 
    r3 = '/Users/aghosh/Data/'
    if os.path.exists(r1):
        root = r1
    elif os.path.exists(r2):
        root = r2
    elif os.path.exists(r3):
        root = r3
    else:
        raise ValueError('Cannot find ROOT folder')

    return root



def logdir_reset(log_dir):
    """
    Cleans up log_dir
    """
    if os.path.exists(log_dir):
        print ('Deleting log-dir:', log_dir)
        shutil.rmtree(log_dir)
    print ('Creating log-dir:', log_dir)
    os.makedirs(log_dir)



def block_match(inp, out):
    """
    Block matching algorithm for an output patch within a larger
    input patch
    """
    sh1 = inp.shape
    sh2 = out.shape
    inpN = (sh1[0] - 1) // 2
    if np.linalg.norm(inp)==0 or np.linalg.norm(out)==0:
        ind = np.array([0, 0, 0])
        MI = np.zeros((sh1[0]-sh2[0]+1, sh1[1]-sh2[1]+1, sh1[2]-sh2[2]+1))
    else:
        outM = (sh2[0] - 1) // 2
        MI = np.array([[[mu.MI(inp[i-outM:i+outM+1, j-outM:j+outM+1, k-outM:k+outM+1], out)
                      for k in range(outM, sh1[2]-outM)]
                      for j in range(outM, sh1[1]-outM)]
                      for i in range(outM, sh1[0]-outM)])
        #MI2 = np.zeros((sh1[0]-sh2[0]+1, sh1[1]-sh2[1]+1, sh1[2]-sh2[2]+1))
        #for i in range(outM, sh1[0]-outM):
        #    for j in range(outM, sh1[1]-outM):
        #        for k in range(outM, sh1[2]-outM):
        #            MI2[i-outM, j-outM, k-outM] = mu.MI(inp[i-outM:i+outM+1, j-outM:j+outM+1, k-outM:k+outM+1], out)
        #print (np.linalg.norm(MI-MI2))
        ind = np.argmax(MI)
        ind = np.unravel_index(ind, MI.shape)
        ind = np.array(ind) - inpN + outM
    return ind, MI



def backward_shuffle_img(imglist, ds):
    """
    Bring hi-res image to low-res dimension through reverse/backward shuffling.
    Shuffling done in place.

    Args:
        imglist (list): List of images to shuffle
        ds (int): Downsample/Upsample rate

    Returns:
        shuff_images (list): reverse shuffled images
    """
    print ('Reverse shuffling hi-res images')
    
    shuffle_indices = [(i, j, k) for k in range(ds)
                                 for j in range(ds)
                                 for i in range(ds)]
    is3D = True
    channelsN = 1
    if len(imglist[0].shape) == 3: 
        pass
    elif len(imglist[0].shape) == 4:
        channelsN = imglist[0].shape[3]
        is3D = False
    else:
        raise ValueError('Only 3D or 4D images handled.')

    shuff_images = []
    for img in imglist:
        shuff_arrays = []
        if is3D:
            for (i, j, k) in shuffle_indices:
                shuff_arrays.append(img[i::ds, j::ds, k::ds])
        else:
            for c in range(channelsN):
                for (i, j, k) in shuffle_indices:
                    shuff_arrays.append(img[i::ds, j::ds, k::ds, c])
        img = np.stack(shuff_arrays, axis=3)
        shuff_images.append(img)

    return shuff_images


def forward_shuffle_img(imglist, us):
    """
    Bring low-res multi-channel images to hi-res dimension through forward
    shuffling. Shuffling done in place.

    Args:
        imglist (list): List of images to shuffle
        us (int): Downsample/Upsample rate

    Return:
        shuff_images (list): list of forward shuffled images
    """

    assert len(imglist[0].shape) == 4
    print ('Forward shuffling shuffled images')
    N = imglist[0].shape[3]
    if (N % (us**3)) > 0:
        raise ValueError('Incompatible number of shuffled layers for upsampling')

    channelsN = N // (us**3)
    shuffle_indices = [(i, j, k) for k in range(us)
                                 for j in range(us)
                                 for i in range(us)]
    shuff_images = []
    for img in imglist:
        sh = img.shape
        newsh = (sh[0]*us, sh[1]*us, sh[2]*us, channelsN)
        img2 = np.zeros(newsh).astype('float32')
        for c in range(channelsN):
            for cnt, (i, j, k) in enumerate(shuffle_indices):
                img2[i::us, j::us, k::us, c] = img[..., c*(us**3) + cnt]
        shuff_images.append(img2)

    return shuff_images


def fetch_subjects(no_subjects=8, shuffle=False, test=False):
    if test:
        subj_list = ['904044', '165840', '889579', '713239',
                     '899885', '117324', '214423', '857263']
    else:
        subj_list = ['992774', '125525', '205119', '133928',  # first 8 are the original Diverse  dataset
                     '570243', '448347', '654754', '153025',
                     '101915', '106016', '120111', '122317',  # original 8 training subjects
                     '130316', '148335', '153025', '159340',
                     '162733', '163129', '178950', '188347',  # original 8 test subjects
                     '189450', '199655', '211720', '280739',
                     '106319', '117122', '133827', '140824',  # random 8 subjects
                     '158540', '196750', '205826', '366446']

    assert no_subjects <= len(subj_list)

    if shuffle: random.shuffle(subj_list)
    return subj_list[:no_subjects]
