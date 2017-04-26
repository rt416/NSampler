from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import nibabel as nib
import os as os
import sys as sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil



def normalise_minmax(imglist, bgval=0., maxval=100., minval=10.):
    """
    Min-max normalises the image data list. The exact process is:
    - work only in voxels with val != bgval (bgval=None: full image)
    - find the maximum value in the list of images: M1
    - find the minimum value in the list of images: M2
    - U1 = maxval
    - U2 = minval
    - normalise all images in the list by:
        x = [U1(x-M2)-U2(x-M1)]/(M1-M2).

    Args:
        imglist (np.ndarray list): list of image data.
        bgval (float): foreground is vox != bgval (bgval=None: full image)
        maxval (float): output maximum.
        minval (float): output minimum.

    Returns:
        M1 (float): input maximum over the list.
        M2 (float): input minimum over the list.

    """
    if bgval==None:
        print ('  including the background')
    else:
        print ('  excluding the background')
    M1 = M2 = 0.
    first = True
    #find M1 M2
    for img in imglist:
        vals = img if bgval==None else img[img!=bgval]
        if first:
            M1 = np.amax(vals)
            M2 = np.amin(vals)
            first = False
        else:
            m1 = np.amax(vals)
            m2 = np.amin(vals)
            M1 = m1 if m1 > M1 else M1
            M2 = m2 if m2 < M2 else M2
    #normalise images
    for img in imglist:
        if bgval==None:
            img[...] = (maxval*(img-M2)-minval*(img-M1))/(M1-M2)
        else:
            indx = img!=bgval
            img[indx] = (maxval*(img[indx]-M2)-minval*(img[indx]-M1))/(M1-M2)
    return M1, M2



def rescale_normalised(imglist, M1, M2, bgval=0, maxval=1000., minval=100.):
    """
    Rescales the images when they have been min-max normalised.

    Args:
        imglist (np.ndarray list): list of image data.
        M1 (float): output maximum.
        M2 (float): output minimum.
        bgval (float): foreground is vox != bgval (bgval=None: full image)
        maxval (float): input maximum.
        minval (float): input minimum.

    """
    if bgval==None:
        print ('  including the background')
    else:
        print ('  excluding the background')
    for img in imglist:
        if bgval==None:
            img[...] = (M1*(img-minval)-M2*(img-maxval))/(maxval-minval)
        else:
            indx = img!=bgval
            img[indx] = (M1*(img[indx]-minval)-M2*(img[indx]-maxval))/(maxval-minval)



def minmax_image_channels(imglist, maxval=2., minval=-2.):
    """
    Min-max normalise the image by channels
    """
    is3D = True
    if len(imglist[0].shape) == 3:
        pass
    elif len(imglist[0].shape) == 4:
        is3D = False
        dimV = imglist[0].shape[3]
    else:
        raise ValueError('Only 3D or 4D images handled.')

    M1 = []
    M2 = []
    first = True
    if is3D:
        print ('  minmax normalising image channels: 3D')
        for img in imglist:
            if first:
                _M1 = np.amax(img)
                _M2 = np.amin(img)
            else:
                m1 = np.amax(img)
                m2 = np.amin(img)
                _M1 = m1 if m1 > _M1 else _M1
                _M2 = m2 if m2 < _M2 else _M2
        M1.append(_M1)
        M2.append(_M2)
        for img in imglist:
            img[...] = (maxval*(img-_M2)-minval*(img-_M1))/(_M1-_M2)
    else:
        print ('  minmax normalising image channels:', dimV)
        for it in range(dimV):
            first = True
            for img in imglist:
                if first:
                    _M1 = np.amax(img[...,it])
                    _M2 = np.amin(img[...,it])
                else:
                    m1 = np.amax(img[...,it])
                    m2 = np.amin(img[...,it])
                    _M1 = m1 if m1 > _M1 else _M1
                    _M2 = m2 if m2 < _M2 else _M2
            M1.append(_M1)
            M2.append(_M2)
            for img in imglist:
                img[...,it] = \
                    (maxval*(img[...,it]-_M2)-minval*(img[...,it]-_M1))/(_M1-_M2)
    return M1, M2



def rescale_minmaximage_channels(imglist, M1, M2, maxval=2., minval=-2.):
    is3D = True
    if len(imglist[0].shape) == 3:
        pass
    elif len(imglist[0].shape) == 4:
        is3D = False
        dimV = imglist[0].shape[3]
        assert dimV == len(mean)
        assert dimV == len(std)
    else:
        raise ValueError('Only 3D or 4D images handled.')

    if is3D:
        print ('  rescaling minmax image channels: 3D')
        m1 = M1[0]
        m2 = M2[0]
        for img in imglist:
            img[...] = (m1*(img-minval)-m2*(img-maxval))/(maxval-minval)
    else:
        print ('  rescaling minmax image channels:', dimV)
        for it in range(dimV):
            m1 = M1[it]
            m2 = M2[it]
            for img in imglist:
                img[...,it] = \
                    (m1*(img[...,it]-minval)-m2*(img[...,it]-maxval))/(maxval-minval)



def minmax_patchlib_channels(patchlib, maxval=2., minval=-2.):
    """
    Min-max normalise patchlib by channels
    """
    raise RuntimeError('Untested buggy code')
    dimV = patchlib.shape[-1]
    print ('  minmax normalising patch channels:', patchlib.shape)
    M1 = []
    M2 = []
    for it in range(dimV):
        m1 = np.amax(patchlib[...,it])
        m2 = np.amin(patchlib[...,it])
        M1.append( m1 )
        M2.append( m2 )
        patchlib[...,it] = \
            (maxval*(patchlib[...,it]-M2)-minval*(patchlib[...,it]-M1))/(m1 - m2)
    return M1, M2



def rescale_minmaxpatchlib_channels(patchlib, M1, M2, maxval=2., minval=-2.):
    raise RuntimeError('Untested buggy code')
    dimV = patchlib.shape[-1]
    assert dimV == len(M1)
    assert dimV == len(M2)
    print ('  rescaling minmax patch channels:', patchlib.shape)
    for it in range(dimV):
        m1 = M1[it]
        m2 = M2[it]
        patchlib[...,it] = \
            (m1*(patchlib[...,it]-minval)-m2*(patchlib[...,it]-maxval))/(maxval-minval)



def whiten_image_channels(imglist):
    """
    Whiten the data: mean=0 std=1. Done in place.

    Args:
        imglist (np.ndarray list): list of image data.

    Returns:
        mean (list of list): mean across all the images and channels
        std  (list of list): std across all the images and channels
    """
    raise RuntimeError('Untested buggy code')
    is3D = True
    if len(imglist[0].shape) == 3:
        pass
    elif len(imglist[0].shape) == 4:
        is3D = False
        dimV = imglist[0].shape[3]
    else:
        raise ValueError('Only 3D or 4D images handled.')

    mean = []
    std  = []
    if is3D:
        print ('  whitening image channels: 3D')
        data = np.array([])
        for img in imglist:
            data = np.append(data, img.flatten(), axis=0)
        m = np.mean(data)
        s = np.std(data)
        del data
        mean.append(m)
        std.append(s)
        for img in imglist:
            img[...] = (img - m) / s
    else:
        print ('  whitening image channels:', dimV)
        for it in range(dimV):
            data = np.array([])
            for img in imglist:
                data = np.append(data, img[...,it].flatten(), axis=0)
            m = np.mean(data)
            s = np.std(data)
            del data
            mean.append(m)
            std.append(s)
            for img in imglist:
                img[...,it] = (img[...,it] - m) / s

    return mean, std



def rescale_whiteimage_channels(imglist, mean, std):
    """
    Rescale whitened data: mean=0 std=1. Done in place.

    Args:
        imglist (np.ndarray list): list of image data.
        mean (list of list): mean across all the images and channels
        std  (list of list): std across all the images and channels
    """
    raise RuntimeError('Untested buggy code')
    is3D = True
    if len(imglist[0].shape) == 3:
        pass
    elif len(imglist[0].shape) == 4:
        is3D = False
        dimV = imglist[0].shape[3]
        assert dimV == len(mean)
        assert dimV == len(std)
    else:
        raise ValueError('Only 3D or 4D images handled.')

    if is3D:
        print ('  rescaling white image channels: 3D')
        m = mean[0]
        s = std[0]
        for img in imglist:
            img[...] = img * s + m
    else:
        print ('  rescaling white image channels:', dimV)
        for it in range(dimV):
            m = mean[it]
            s = std[it]
            for img in imglist:
                img[...,it] = img[...,it] * s + m




def whiten_patchlib_channels(patchlib):
    raise RuntimeError('Untested buggy code')
    dimV = patchlib.shape[-1]
    print ('  whitening patch channels:', patchlib.shape)
    mean = []
    std  = []
    for i in range(dimV):
        mean.append( np.mean(patchlib[..., i]) )
        std.append ( np.std (patchlib[..., i]) )
        patchlib[..., i] = (patchlib[..., i] - mean[i]) / std[i]
    return mean, std



def rescale_whitepatchlib_channels(patchlib, mean, std):
    raise RuntimeError('Untested buggy code')
    dimV = patchlib.shape[-1]
    assert dimV == len(mean)
    assert dimV == len(std)
    print ('  rescaling white patch channels:', patchlib.shape)
    for i in range(dimV):
        patchlib[..., i] = patchlib[..., i] * std[i] + mean[i]



def scale_images(srclist, destlist, bgval=0):
    """
    Scale normalise images by equating medians
    """
    print ('  scaling input training images to output images')
    med2list = []
    for img in destlist:
        med2list.append( np.median(img[img != bgval]) )
    med2 = np.mean(med2list)
    for img1, img2 in zip(srclist, destlist):
        m1 = np.median(img1[img1 != bgval])
        img1[img1 != bgval] *= med2 / m1
        #m2 = np.median(img2[img2 != bgval])
        #img2[img2 != bgval] *= med2 / m2
    return med2



def scale_src_image(imglist, med2, bgval=0):
    print ('  scaling input test images')
    for img in imglist:
        m1 = np.median(img[img != bgval])
        img[img != bgval] *= med2 / m1

