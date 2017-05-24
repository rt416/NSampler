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

import cPickle as pkl
import os
import largesc.patch_sampler as patch_sampler
import largesc.data_utils as dutils

# import data_restore as restore
import numpy as np

# def load_patchlib(filename, train_folder):
#     """
#     Looks and loads patchlib
#
#     Args:
#         filename (str): Filename to save patchlib (will be saved in Training data
#                         folder)
#         train_folder (str): subpath is added between training data folder and filename
#
#     Returns:
#         dataset: data_patchlib.Data, which provides a next_batch function
#                  (dataset = None if no patchlib is found)
#     """
#     trainfile = train_folder + filename
#     dataset = None
#     if os.path.isfile(trainfile):
#         print ('Loading patch library:', trainfile)
#         dataset = patch_sampler.Data().load(trainfile)
#     else:
#         raise RuntimeError('Patch library not found:', trainfile)
#     return dataset
#

# the basic data preparation subfunction:
def prepare_data(size,
                 eval_frac,
                 inpN,
                 outM,
                 no_channels,
                 patchlib_name,
                 whiten,
                 inp_header='dt_b1000_lowres_2_',
                 out_header='dt_b1000_',
                 method='default',
                 train_index=[],
                 bgval=0,
                 is_reset=False,
                 clip=False,
                 us_rate=2,
                 data_dir_root='',
                 save_dir_root='',
                 subpath=''):
    """
    Data preparation and patch generation for diffusion data.
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
        patchlib_name (str): name of the patchlib subdir to save patchlib indices and transformation.

        train_index (list): subject list to include in the training data
        bgval (float): background value: voxels outside the mask
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
        raise ('Training index not provided, using default ')


    # Create the dir for saving the training data details
    train_folder = save_dir_root + patchlib_name

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)


    # load the images into memory (as a list of numpy arrays):
    inp_channels = range(3,no_channels+3)
    out_channels = range(3,no_channels+3)
    inp_images, out_images = load_data(data_dir_root,
                                       subpath,
                                       train_index,
                                       inp_channels,
                                       out_channels,
                                       inp_header,
                                       out_header)

    # Check if there're any nan/inf
    print ('Sanitising data...')
    for i in range(len(train_index)):
        dutils.sanitise_imgdata(inp_images[i])
        dutils.sanitise_imgdata(out_images[i])

    # Feed the data into patch extractor:
    patfile = train_folder + '/patchlib_indices.pkl'
    transfile = train_folder + '/transforms.pkl'

    if os.path.isfile(patfile) and os.path.isfile(transfile) and not is_reset:
        print ('Loading patch indices...')
        dataset = patch_sampler.Data().load_patch_indices(patfile,
                                                          transfile,
                                                          inp_images,
                                                          out_images,
                                                          inpN,
                                                          ds=us_rate,
                                                          whiten=whiten,
                                                          clip=clip)
        print('Save transformation:' + transfile)
        dataset.save_transform(transfile)
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
                                                       whiten=whiten,
                                                       bgval=bgval,
                                                       method=method,
                                                       clip=clip)
        print ('Saving patch indices:' + patfile)
        dataset.save_patch_indices(patfile)
        print('Saving transformation:' + transfile)
        dataset.save_transform(transfile)

    return dataset, train_folder


def load_data(data_dir_root,
              subpath,
              train_index,
              inp_channels,
              out_channels,
              inp_header,
              out_header):
    """ load a sequence of nii files.

    Args:
        data_dir_root (str) : root dir for data
        train_index: subjects list e.g. ['117324', '904044']
        inp_channels (list of indices): for DTI e.g. [3,4,5,6,7,8]
        out_channels (list of int): for DTI e.g. [3,4,5,6,7,8]
        inp_header (str): header of input nii files e.g. 'dt_b1000_lowres_2_'
        out_header (str): header of output nii files e.g. 'dt_b1000_'

    Returns:
        inp_images (list): list of numpy arrays

    """

    print ('Loading data...')
    inp_images = [0,] * len(train_index)
    out_images = [0,] * len(train_index)
    ind = 0

    # todo: need to make the naming more general - currently specific to DTIs
    for subject in train_index:
        inp_file = (data_dir_root + subject + subpath +
                    '/'+inp_header)
        out_file = (data_dir_root + subject + subpath +
                    '/'+out_header)
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