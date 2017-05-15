"""Ryu: main experiments script"""

import argparse
import os
from largesc.train_v2 import train_cnn
import largesc.reconstruct_v2 as reconstruct
import sys
from data_utils import fetch_subjects
from train_v2 import name_network

# Settings
parser = argparse.ArgumentParser(description='dliqt-tensorflow-implementation')
parser.add_argument('-e', '--experiment', dest='experiment', type=str, default='25Apr2017', help='name of the experiment')
parser.add_argument('-m', '--method', dest='method', type=str, default='cnn_simple', help='network type')
parser.add_argument('--valid', action='store_true', help='pick the best model based on the loss, not the MSE?')
parser.add_argument('--overwrite', action='store_true', help='restart the training completelu')
parser.add_argument('--continue', action='store_true', help='continue training from previous epoch')
parser.add_argument('--is_reset', action='store_true', help='reset the patch library?')
parser.add_argument('--save', action='store_true', help='save the reconstructed output?')

parser.add_argument('--optimizer', type=str, default='adam', help='optimization method')
parser.add_argument('-lr', '--learning_rate', dest='learning_rate', type=float, default='1e-3', help='learning rate')
parser.add_argument('-dr', '--dropout_rate', dest='dropout_rate', type=float, default='0.0', help='drop-out rate')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=12, help='batch size')
parser.add_argument('--validation_fraction', type=float, default=0.5, help='fraction of validation data')
parser.add_argument('--train_size', type=int, default=18000, help='total number of patches')
parser.add_argument('--patch_sampling_opt', type=str, default='default', help='sampling scheme for patche extraction')
parser.add_argument('--transform_opt', type=str, default='standard', help='normalisation transform')

# Data/task
parser.add_argument('--is_map', action='store_true', help='MAP-SR?')
parser.add_argument('-ts', '--no_subjects', dest="no_subjects", type=int, default='8', help='background value')
parser.add_argument('--no_channels', type=int, default=6, help='number of channels')
parser.add_argument('-us', '--upsampling_rate', dest="upsampling_rate", type=int, default=2, help='upsampling rate')
parser.add_argument('-ir', '--input_radius', dest="input_radius", type=int, default=5, help='input radius')
parser.add_argument('-rr', '--receptive_field_radius', dest="receptive_field_radius", type=int, default=2, help='receptive field radius')

# Directories:
parser.add_argument('--base_dir', type=str, default='/SAN/vision/hcp/Ryu/miccai2017', help='base directory')

arg = parser.parse_args()
opt = vars(arg)
if opt['continue']==True or opt['overwrite'] ==True:
    assert opt['continue']!= opt['overwrite']

# Other micellaneous parameters:
opt['n_h1'] = 50
opt['n_h2'] = 2 * opt['n_h1']
opt['n_h3'] = 10
# opt['L1_reg'] = 0.00
# opt['L2_reg'] = 1e-5

# data/task:
opt['train_subjects'] = fetch_subjects(no_subjects=opt['no_subjects'], shuffle=False, test=False)
opt['b_value'] = 1000
opt['patchlib_idx'] = 1
opt['no_randomisation'] = 1
opt['output_radius'] = ((2*opt['input_radius']-2*opt['receptive_field_radius']+1)//2)

# directories:
base_dir = opt['base_dir']+'/'+opt['experiment']+'/'
if not(os.path.exists(base_dir)):
    os.makedirs(base_dir)
    os.makedirs(base_dir + 'data/')
    os.makedirs(base_dir + 'log/')
    os.makedirs(base_dir + 'models/')
    os.makedirs(base_dir + 'recon/')

opt['data_dir'] = base_dir + 'data/'
opt['save_dir'] = base_dir + 'models/'
opt['log_dir'] = base_dir + 'log/'
opt['recon_dir'] = base_dir + 'recon/'

opt['mask_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/recon/'
opt['gt_dir'] = '/SAN/vision/hcp/DCA_HCP.2013.3_Proc/'  # ground truth dir
opt['subpath'] = '/T1w/Diffusion/'

# Mean Apparent Propagater MRI
if opt['is_map']:
    opt['input_file_name'] = 'h4_all_lowres_'+str(opt['upsampling_rate'])+'_'
    opt['output_file_name'] = 'h4_recon.npy'
    opt['gt_header'] = 'h4_all_'
    opt['no_channels'] = 22
else:
    opt['input_file_name'] = 'dt_b1000_lowres_'+str(opt['upsampling_rate'])+'_'
    opt['gt_header'] = 'dt_b1000_'


# -------------------- Train and test --------------------------------:
with open(opt['save_dir']+name_network(opt)+'/output.txt', 'w') as f:
    # Redirect all the outputs to the text file:
    print("Redirecting the output to: "
          +opt['save_dir']+name_network(opt)+"/output.txt")
    sys.stdout = f

    # Train:
    print(opt)
    train_cnn(opt)

    # Reconstruct:
    subjects_list = fetch_subjects(no_subjects=8, shuffle=False, test=True)
    rmse_average = 0
    for subject in subjects_list:
        opt['subject'] = subject
        rmse, _ = reconstruct.sr_reconstruct(opt)
        rmse_average += rmse
    print('\n Average RMSE on Diverse dataset is %.15f.'
          % (rmse_average / len(subjects_list),))