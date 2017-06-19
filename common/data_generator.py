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
import os
import patch_sampler as patch_sampler
import  data_utils as dutils


# The main function for define the patch loader:
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
                 shuffle=True,
                 pad_size=-1,
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
    train_folder = os.path.join(save_dir_root,patchlib_name)

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
    patfile = os.path.join(train_folder,'patchlib_indices.pkl')
    transfile = os.path.join(train_folder,'transforms.pkl')

    if os.path.isfile(patfile) and os.path.isfile(transfile) and not is_reset:
        print ('Loading patch indices...')
        dataset = patch_sampler.Data().load_patch_indices(patfile,
                                                          transfile,
                                                          inp_images,
                                                          out_images,
                                                          inpN,
                                                          us_rate=us_rate,
                                                          whiten=whiten,
                                                          pad_size=pad_size,
                                                          clip=clip,
                                                          shuffle=shuffle)
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
                                                        us_rate=us_rate,
                                                        whiten=whiten,
                                                        bgval=bgval,
                                                        method=method,
                                                        pad_size=pad_size,
                                                        clip=clip,
                                                        shuffle=shuffle)
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
    """Load a sequence of nifti files.

    Load a sequence of nifti files, which should be stored in the HCP folder

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

    for subject in train_index:
        inp_file = os.path.join(data_dir_root, subject, subpath, inp_header)
        out_file = os.path.join(data_dir_root, subject, subpath, out_header)
        inp_images[ind], hdr = dutils.load_series_nii(inp_file, inp_channels, dtype='float32')
        out_images[ind], _   = dutils.load_series_nii(out_file, out_channels, dtype='float32')
        ind += 1

    return inp_images, out_images
