"""Ryu: main experiments script"""

import argparse
import os, sys
sys.path.append('./..')
from common.data_utils import fetch_subjects
import common.stats as stats
import train
import reconstruct


# -------------------------- Set up configurations ----------------------------
# Basic settings
parser = argparse.ArgumentParser(description='dliqt-tensorflow-implementation')
parser.add_argument('-e', '--experiment', dest='experiment', type=str, default='experiment_1', help='name of the experiment')
parser.add_argument('--gpu', type=str, default="0", help='which GPU to use')
parser.add_argument('--overwrite', action='store_true', help='restart the training completelu')
parser.add_argument('--continue', action='store_true', help='continue training from previous epoch')
parser.add_argument('--is_reset', action='store_true', help='reset the patch library?')
parser.add_argument('--is_samples', action='store_true', help='want to monitor intermediate samples?')
parser.add_argument('--not_save', action='store_true', help='evoke not to save reconstructed output.')
parser.add_argument('--disp', action='store_true', help='save the displayed outputs?')
parser.add_argument('-pp', '--postprocess', dest='postprocess', action='store_true', help='post-process the estimated highres output?')

# Directories:
parser.add_argument('--base_dir', type=str, default='/SAN/vision/hcp/Ryu/miccai2017', help='base directory')
parser.add_argument('--gt_dir', type=str, default='/SAN/vision/hcp/DCA_HCP.2013.3_Proc', help='ground truth directory')
parser.add_argument('--subpath', type=str, default='T1w/Diffusion', help='subdirectory in gt_dir')
parser.add_argument('--mask_dir', type=str, default='/SAN/vision/hcp/Ryu/miccai2017/hcp_masks', help='directory of segmentation masks')
parser.add_argument('--mask_subpath', type=str, default='', help='subdirectory in mask_dir')
parser.add_argument('--mask_name', type=str, default='', help='mask nifti name')

# Model:
parser.add_argument('-m', '--method', dest='method', type=str, default='espcn', help='network type')
parser.add_argument('--no_filters', type=int, default=50, help='number of initial filters')
parser.add_argument('--no_layers', type=int, default=2, help='number of hidden layers')
parser.add_argument('--mc_no_samples', type=int, default=50, help='number of MC samples at reconstruction')

parser.add_argument('--hetero', action='store_true', help='want to perform heteroscedastic training?')
parser.add_argument('--vardrop', action='store_true', help='want to perform variational dropout?')
parser.add_argument('--params', type=str, default='weight', help='parameters of var. dropout. Other options: layer, channel')
parser.add_argument('--cov_on', action='store_true', help='want to perform variational dropout on the covariance network?')
parser.add_argument('--hybrid_on', action='store_true', help='want to perform hybrid training for var + hetero models?')
parser.add_argument('--decompose', action='store_true', help='want to decompose uncertainty?')


parser.add_argument('--is_BN', action='store_true', help='want to use batch normalisation?')
parser.add_argument('-dr', '--dropout_rate', dest='dropout_rate', type=float, default='0.0', help='drop-out rate')

parser.add_argument('--optimizer', type=str, default='adam', help='optimization method')
parser.add_argument('-lr', '--learning_rate', dest='learning_rate', type=float, default='1e-3', help='learning rate')
parser.add_argument('--no_epochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=12, help='batch size')
parser.add_argument('--validation_fraction', type=float, default=0.5, help='fraction of validation data')
parser.add_argument('--valid', action='store_true', help='pick the best model based on the loss, not the MSE?')


# Data/task
parser.add_argument('--is_map', action='store_true', help='MAP-SR?')
parser.add_argument('-pl', '--no_patches', dest='no_patches', type=int, default=2250, help='number of patches sampled from each train subject')
parser.add_argument('-ts', '--no_subjects', dest="no_subjects", type=int, default='8', help='background value')
parser.add_argument('-bgval', '--background_value', dest="background_value", type=float, default='0', help='background value')
parser.add_argument('--no_channels', type=int, default=6, help='number of channels')
parser.add_argument('-us', '--upsampling_rate', dest="upsampling_rate", type=int, default=2, help='upsampling rate')
parser.add_argument('-ir', '--input_radius', dest="input_radius", type=int, default=5, help='input radius')
parser.add_argument('--patch_sampling_opt', type=str, default='default', help='sampling scheme for patche extraction')
parser.add_argument('--transform_opt', type=str, default='standard', help='normalisation transform')
parser.add_argument('--pad_size', type=int, default=-1, help='size of padding applied before patch extraction. Set -1 to apply maximal padding.')
parser.add_argument('--is_clip', action='store_true', help='want to clip the images (0.1% - 99.9% percentile) for preprocessing?')
parser.add_argument('--is_shuffle', action='store_true', help='want to reverse shuffle the HR output into LR space?')
parser.add_argument('--is_dt_all', action='store_true', help='use the dt_all files')
parser.add_argument('--is_mdfacfa', action='store_true', help='compute MD, FA and CFA')


arg = parser.parse_args()
opt = vars(arg)
if opt['continue']==True or opt['overwrite'] ==True: assert opt['continue']!= opt['overwrite']

# GPUs devices:
os.environ["CUDA_VISIBLE_DEVICES"]=opt["gpu"]
from tensorflow.python.client import device_lib
print device_lib.list_local_devices()

# data/task:
opt['train_size']=int(opt['no_patches']*opt['no_subjects'])
opt['train_subjects'] = fetch_subjects(no_subjects=opt['no_subjects'], shuffle=False, test=False)
opt['patchlib_idx'] = 1

# Make directories to store results:
base_dir = os.path.join(opt['base_dir'],opt['experiment'])

# Update directories in args
opt.update({
    "data_dir": os.path.join(base_dir,"data"),
    "save_dir": os.path.join(base_dir,"models"),
    "log_dir": os.path.join(base_dir,"log"),
    "recon_dir": os.path.join(base_dir,"recon"),
    "stats_dir": os.path.join(base_dir, "stats")
})

if not(os.path.exists(base_dir)):
    os.makedirs(base_dir)
    os.makedirs(opt["data_dir"])
    os.makedirs(opt["save_dir"])
    os.makedirs(opt["log_dir"])
    os.makedirs(opt["recon_dir"])
    os.makedirs(opt["stats_dir"])

# Mean Apparent Propagator MRI
opt['input_file_name'] = 'dt_b1000_lowres_'+str(opt['upsampling_rate'])+'_{:d}.nii'
opt['gt_header'] = 'dt_b1000_{:d}.nii'
opt['output_file_name'] = 'dt_recon_mc=%i.npy' % opt["mc_no_samples"]
opt['output_std_file_name'] = 'dt_std_mc=%i.npy' % opt["mc_no_samples"]
opt['output_var_random_file_name'] = 'var_random_mc=%i.npy' % opt["mc_no_samples"]
opt['output_var_model_file_name'] = 'var_model_mc=%i.npy' % opt["mc_no_samples"]


if opt['is_map']:
    opt['input_file_name'] = 'h4_all_lowres_'+str(opt['upsampling_rate'])+'_{:02d}.nii'
    opt['output_file_name'] = 'h4_recon_mc=%i.npy' % opt["mc_no_samples"]
    opt['output_std_file_name'] = 'h4_std_mc=%i.npy' % opt["mc_no_samples"]
    opt['gt_header'] = 'h4_all_{:02d}.nii'
    opt['no_channels'] = 22

if opt['is_dt_all']:
    opt['input_file_name'] = 'dt_all_lowres_'+str(opt['upsampling_rate'])+'_{:d}.nii'
    opt['gt_header'] = 'dt_all_{:d}.nii'
    opt['output_file_name'] = 'dt_recon_all.npy'
    opt['output_std_file_name'] = 'dt_std_all.npy'

# Others:
opt['save_as_ijk'] = False
opt['gt_available'] = True

# Print options
for key, val in opt.iteritems():
    print("{}: {}".format(key, val))

# TRAIN
train.train_cnn(opt)


# RECONSTRUCT:
subjects_list = fetch_subjects(no_subjects=8, shuffle=False, test=True)
for subject in subjects_list:
    opt['subject'] = subject
    reconstruct.sr_reconstruct(opt)

# STATS:
stats.compute_stats(opt, subjects_list)
