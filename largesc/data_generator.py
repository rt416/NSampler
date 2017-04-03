"""
Data preparation and patch generation for HCP diffusion data.
We assume that all diffusion components are stored separately as nifti files.

Outputs the Data class that provides a next_batch function to call for training.

Does:
(Data preparation)
1. Loads the datasets.
2. Sanitises the data (for inf and nan).
(Patch generation)
3. Returns the data_patchlib.Data class object with this dataset for patch
   generation.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import os
import largesc.patch_sampler as patch_sampler
import largesc.data_utils as dutils

# import data_restore as restore
import numpy as np

def load_patchlib(filename, train_folder):
    """
    Looks and loads patchlib

    Args:
        filename (str): Filename to save patchlib (will be saved in Training data
                        folder)
        train_folder (str): subpath is added between training data folder and filename

    Returns:
        dataset: data_patchlib.Data, which provides a next_batch function
                 (dataset = None if no patchlib is found)
    """
    trainfile = train_folder + filename
    dataset = None
    if os.path.isfile(trainfile):
        print ('Loading patch library:', trainfile)
        dataset = patch_sampler.Data().load(trainfile)
    else:
        raise RuntimeError('Patch library not found:', trainfile)
    return dataset


# the basic data preparation subfunction:
def prepare_data(size,
                 eval_frac,
                 inpN,
                 outM,
                 filename,
                 whiten,
                 is_Blockmatch=False,
                 sub_path='',
                 train_index=[],
                 bgval=0,
                 inp_channels=4,
                 is_reset=False,
                 sample_sz=10,
                 us_rate = 1,
                 data_dir_root='',
                 save_dir_root=''):
    """
    Data preparation and patch generation for Pepys rat data.
    Outputs the Data class that provides a next_batch function to call for training.

    Does:
    (Data preparation)
    1. Loads the datasets.
    2. Sanitises the data (for inf and nan).
    (Patch generation)
    3. Returns the data_patchlib.Data class object with this dataset for patch
       generation.
    Note: Data is loaded only once and will not be reloaded. To force reload
          reload the module.

    Args:
        size (int): Total size of the patchlib
        eval_frac (float: [0,1]): fraction of the dataset used for evaluation
        inpN (int): input patch size = (2*inpN + 1)
        outM (int): output patch size = (2*outM + 1)
        filename (str): Filename to save patchlib (will be saved in Training data
                        folder)
        sub_path (str): subpath is added between training data folder and filename
        train_index (list): subject list to include in the training data
        bgval (float): Background value: voxels outside the mask
        inp_channels (int): Number of input data channels
        whiten (whiten type): Whiten data or not
        is_reset (bool): if set will recompute patchlib even if file exists and
                        overwrite the old patchlib
        sample_sz (int, optional): Used internally to sample the voxles randomly
                                     within the list of subjects
        us_rate (int) : Upsampling rate
        data_dir_root (str) : root dir where source images are stored from which
                            patche pairs are extracted.
        save_dir_root (str) : root dir where the patch library or extractor are
                            saved.

    Returns:
        dataset: data_patchlib.Data, which provides a next_batch function
        TraininDataFolder (str): path to training data folder
    """

    # Get the list of training subjects.
    if not train_index:
        print ('Training index not provided, using default ')
        train_index = ['117324', '904044']

    # Create the dir for saving the training data details
    train_folder = save_dir_root + 'TrainingData/' + sub_path
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)

    # Load/create training sets.
    trainfile = train_folder + filename
    if os.path.isfile(trainfile) and not is_reset:
        print('Loading patch library:', trainfile)
        dataset = patch_sampler.Data().load(trainfile)
    else:
        # load the images into memory
        inp_channels = range(3,9)
        out_channels = range(3,9)
        inp_images, out_images = load_data(data_dir_root,
                                           train_index,
                                           inp_channels,
                                           out_channels)

        # Check if there're any nan/inf
        print ('Sanitising data...')
        for i in range(len(train_index)):
            dutils.sanitise_imgdata(inp_images[i])
            dutils.sanitise_imgdata(out_images[i])

        new_whiten = whiten
        # normalise the image.
        # if whiten == flags.NORM_SCALE_IMG:
        #     inp = []
        #     for img in inp_images:
        #         inp.append(img[...,0])
        #     med2 = dwh.scale_images(inp, out_images, bgval=bgval)
        #     for it in range(1, inp_channels):
        #         inp = []
        #         for img in inp_images:
        #             inp.append(img[...,it])
        #         dwh.normalise_minmax(inp, maxval=1000, minval=0, bgval=bgval)
        #         dwh.scale_src_image(inp, med2, bgval=bgval)
        #     new_whiten = flags.NORM_NONE

        ext = sub_path.replace('/', '')
        patfile = train_folder + filename.replace('.p', '_') + ext + '_patches.p'

        # Feed the data into patch extractor:
        if os.path.isfile(patfile) and not is_reset:
            print ('Loading patch indices...')
            dataset = patch_sampler.Data().load_patch_indices(patfile,
                                                     inp_images, out_images)
        else:
            print ('Computing patch library...')
            print(us_rate)
            dataset = patch_sampler.Data().create_patch_lib(size,
                                                   eval_frac,
                                                   inpN,
                                                   outM,
                                                   inp_images,
                                                   out_images,
                                                   ds=us_rate,
                                                   whiten=new_whiten,
                                                   bgval=bgval,
                                                   sample_sz=sample_sz)
            # if whiten == flags.NORM_SCALE_IMG:
            #     dataset._sparams.med2 = med2
            spfile = train_folder + filename.replace('.p', '_') + ext + '_spars.p'
            print ('Saving patch-scale-params:', spfile)
            dataset.save_scale_params(spfile)
            dataset.save_patch_indices(patfile)

    return dataset, train_folder


def load_data(data_dir_root,
              train_index,
              inp_channels,
              out_channels):
    """ load a sequence of nii files.

    Args:
        data_dir_root (str) : root dir for data
        train_index: subjects list e.g. ['117324', '904044']
        inp_channels (list of indices): for DTI e.g. [3,4,5,6,7,8]
        out_channels (list of int): for DTI e.g. [3,4,5,6,7,8]

    Returns:
        inp_images (list): list of numpy arrays

    """

    print ('Loading data...')
    inp_images = [0,] * len(train_index)
    out_images = [0,] * len(train_index)
    ind = 0

    for subject in train_index:
        inp_file = (data_dir_root + subject + '/dt_b1000_lowres_2_{0:d}.nii')
        out_file = (data_dir_root + subject + '/dt_b1000_{0:d}.nii')
        inp_images[ind], hdr = dutils.load_series_nii(inp_file,
                                                      inp_channels,
                                                      dtype='float32')
        out_images[ind], _   = dutils.load_series_nii(out_file,
                                                      out_channels,
                                                      dtype='float32')
        ind += 1

    # print ('  Truncating 9.4T histogram...')
    # for img in out_images:
    #     img[img > MAX_VAL] = MAX_VAL
    return inp_images, out_images


# # Main function for generator based reconstruction function:
# def prepare_restore(restM,
#                     restore_index,
#                     sub_path,
#                     bgval=0,
#                     inp_channels=4,
#                     is_gt=False,
#                     gt_channels=1):
#     """
#     Finds all the patches that need to be restored of the given size and not
#     in the background
#
#     Args:
#         restM (int): patch size = (2*restM + 1)
#         restore_index (int): subject index to include for restoring
#         sub_path (str): subpath is added between training data folder and filename
#         bgval (float): Background value: voxels outside the mask
#         inp_channels (int): Number of input data channels
#         is_gt (bool): If set loads the 9.4T GT image
#         gt_channels (int): Number of ground truth data channels
#
#     Returns:
#         indexlist: index-list of voxels in the image to restore
#         img (np.ndarray): Sanitised and normalised image
#         hdr (nifti header): Nifti header information
#         train_folder (str): path to training data folder
#         imgtest (np.ndarray): Sanitised and normalised 9.4T image
#     """
#     root = dutils.get_root_hcp() + 'Pepys_1T_vs_9-4T/'
#     train_folder = root + 'TrainingData/' + sub_path
#     train_folder_pref = root + 'TrainingData/'
#     if not os.path.exists(train_folder_pref):
#         os.makedirs(train_folder_pref)
#
#     print ('Loading Pepys data...')
#     cnt = 0
#     for datfile1T, datfile9p4T in zip(datanames_1T, datanames_9p4T):
#         inp_file = root + out_subpath1T + pref_1T + datfile1T
#         if is_gt:
#             inp9_file = root + out_subpath9p4T + pref_9T + datfile9p4T
#         if (cnt+1) == restore_index:
#             img, hdr = dutils.load_series_nii(inp_file,
#                                                      range(1, inp_channels+1))
#             if is_gt:
#                 imgtest, _ = dutils.load_series_nii(inp9_file,
#                                                     range(1, gt_channels+1))
#                 # print ('  Truncating 9.4T histogram...')
#                 # imgtest[imgtest > MAX_VAL] = MAX_VAL
#             break
#         cnt += 1
#
#     print ('Sanitising data...')
#     dutils.sanitise_imgdata(img)
#     imglist = [img]
#     if is_gt:
#         dutils.sanitise_imgdata(imgtest)
#
#     filename = ( train_folder_pref +
#         'PatchtestGrid_S{0:02d}_{1:d}x{1:d}.p'.format(restore_index, restM) )
#
#     if os.path.isfile(filename):
#         print ('Loading patch-list:', filename)
#         indexlist = pickle.load(open(filename, 'rb'))
#     else:
#         print ('Computing test patches...')
#         index_list = restore.get_gridrestore_indices(imglist, restM, bgval=bgval)
#
#         indexlist = index_list[0]
#         print ('Saving test patches...', filename)
#         pickle.dump(indexlist, open(filename, 'wb'))
#
#     return indexlist, img, hdr, train_folder, imgtest