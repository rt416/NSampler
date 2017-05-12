"""Ryu: main experiments script"""

import argparse
import os
from largesc.train_v2 import train_cnn
import largesc.reconstruct_v2 as reconstruct
import sys
from train_v2 import name_network

# Training settings
parser = argparse.ArgumentParser(description='dliqt-tensorflow-implementation')
parser.add_argument('--experiment', type=str, default='25Apr2017', help='name of the experiment')
parser.add_argument('--method', type=str, default='cnn_simple', help='network type')
parser.add_argument('--valid', action='store_true', help='pick the best model based on the loss, not the MSE?')
parser.add_argument('--overwrite', action='store_true', help='restart the training completelu')
parser.add_argument('--continue', action='store_true', help='continue training from previous epoch')
parser.add_argument('--is_reset', action='store_true', help='reset the patch library?')

parser.add_argument('--optimizer', type=str, default='adam', help='optimization method')
parser.add_argument('--learning_rate', type=float, default='1e-3', help='learning rate')
parser.add_argument('--dropout_rate', type=float, default='0.0', help='drop-out rate')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=12, help='batch size')
parser.add_argument('--validation_fraction', type=float, default=0.5, help='fraction of validation data')
parser.add_argument('--train_size', type=int, default=18000, help='total number of patches')
parser.add_argument('--patch_sampling_opt', type=str, default='default', help='sampling scheme for patche extraction')
parser.add_argument('--transform_opt', type=str, default='standard', help='normalisation transform')

# Data/task
parser.add_argument('--background_value', type=float, default='0', help='background value')
parser.add_argument('--no_channels', type=int, default=6, help='number of channels')
parser.add_argument('--upsampling_rate', type=int, default=2, help='upsampling rate')
parser.add_argument('--input_radius', type=int, default=5, help='input radius')
parser.add_argument('--receptive_field_radius', type=int, default=2, help='receptive field radius')

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
opt['train_subjects']=['117324', '904044']
# opt['train_subjects'] = ['992774', '125525', '205119', '133928', # first 8 are the original Diverse  dataset
#                          '570243', '448347', '654754', '153025']
                         # '101915', '106016', '120111', '122317', # original 8 training subjects
                         # '130316', '148335', '153025', '159340',
                         # '162733', '163129', '178950', '188347', # original 8 test subjects
                         # '189450', '199655', '211720', '280739',
                         # '106319', '117122', '133827', '140824', # random 8 subjects
                         # '158540', '196750', '205826', '366446']
                         # '351938', '390645', '545345', '586460',
                         # '705341', '749361', '765056', '951457']
opt['no_subjects'] = len(opt['train_subjects'])
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

# ''/SAN/vision/hcp/Ryu/miccai2017/25Apr2017/'
opt['data_dir'] = base_dir + 'data/'
opt['save_dir'] = base_dir + 'models/'
opt['log_dir'] = base_dir + 'log/'
opt['recon_dir'] = base_dir + 'recon/'

opt['mask_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/recon/'
opt['gt_dir'] = '/SAN/vision/hcp/DCA_HCP.2013.3_Proc/'  # ground truth dir
opt['subpath'] = '/T1w/Diffusion/'
opt['input_file_name'] = 'dt_b1000_lowres_' + str(opt['upsampling_rate']) + '_'


with open(opt['save_dir']+name_network(opt)+'/output.txt', 'w') as f:
    # Redirect all the outputs to the text file:
    sys.stdout = f

    # Train:
    print(opt)
    train_cnn(opt)

    # Reconstruct:
    subjects_list = ['904044', '165840', '889579', '713239',
                     '899885', '117324', '214423', '857263']
    rmse_average = 0
    for subject in subjects_list:
        opt['subject'] = subject
        rmse, _ = reconstruct.sr_reconstruct(opt)
        rmse_average += rmse
    print('\n Average RMSE on Diverse dataset is %.15f.'
          % (rmse_average / len(subjects_list),))
