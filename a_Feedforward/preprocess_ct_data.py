import os
import glob
import numpy as np
from common.sr_utility import ndarray_to_nifti
import pydicom


def stack_dicom_files(list_of_files):
    """Read dicom files and stack them into a 3D volume."""
    img_list = []
    for f in list_of_files:
        ds = pydicom.read_file(f)
        img = ds.pixel_array
        img_list.append(img)
    return np.stack(img_list, axis=-1)

# ###### Pre-process the CT images ######
# TODO (prachi): Replace data_dir with where you've saved the raw DICOM files from Joe
data_dir = "/Users/ryutarotanno/Data/prachi_msc_project/raw"

# TODO (prachi): Replace save_dir with where you want to save the  preprocessed CT images i.e. training data.
save_dir = "/Users/ryutarotanno/Data/prachi_msc_project/processed"

subjects = ["IPF-" + str(i) for i in range(1, 10)] + ["IPF-11"]

for subject in subjects:
    print("\n Processing subject: %s" %(subject,))

    # Load all normal, low, ultra-low-dose scans
    files = glob.glob(os.path.join(data_dir, subject+'-N') + '/*.DICOM')
    img_normal = stack_dicom_files(files)

    files = glob.glob(os.path.join(data_dir, subject+'-L') + '/*.DICOM')
    img_low = stack_dicom_files(files)

    files = glob.glob(os.path.join(data_dir, subject+'-UL') + '/*.DICOM')
    img_ultralow = stack_dicom_files(files)

    print("    Volume shapes: NORMAL, LOW, ULTRALOW", img_normal.shape, img_low.shape, img_ultralow.shape)

    # Create a directory to save:
    if not os.path.exists(os.path.join(save_dir, subject)):
        os.makedirs(os.path.join(save_dir, subject))
    else:
        print("    path exists:" + os.path.join(save_dir, subject))

    # Save normal-dose scans
    ndarray_to_nifti(img_normal, nifti_file=os.path.join(save_dir, subject) + '/image_stack_normal_1')
    ndarray_to_nifti(img_normal, nifti_file=os.path.join(save_dir, subject) + '/image_stack_normal_2')
    ndarray_to_nifti(img_normal, nifti_file=os.path.join(save_dir, subject) + '/image_stack_normal_3')

    # Save low-dose scans if size matches that of normal-dose
    if img_normal.shape == img_low.shape:
        ndarray_to_nifti(img_low, nifti_file=os.path.join(save_dir, subject) + '/image_stack_low_1')
        ndarray_to_nifti(img_low, nifti_file=os.path.join(save_dir, subject) + '/image_stack_low_2')
        ndarray_to_nifti(img_low, nifti_file=os.path.join(save_dir, subject) + '/image_stack_low_3')
    else:
        print("    Shape mismatch with low-dose volume")

    # Save ultra-low-dose scans if size matches that of normal-dose
    if img_normal.shape == img_ultralow.shape:
        ndarray_to_nifti(img_ultralow, nifti_file=os.path.join(save_dir, subject) + '/image_stack_ultralow_1')
        ndarray_to_nifti(img_ultralow, nifti_file=os.path.join(save_dir, subject) + '/image_stack_ultralow_2')
        ndarray_to_nifti(img_ultralow, nifti_file=os.path.join(save_dir, subject) + '/image_stack_ultralow_3')
    else:
        print("    Shape mismatch with ultralow-dose volume")
