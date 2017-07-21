"""Ryu: main experiments script for non-HCP data reconstruction """
import argparse
import os
from common.data_utils import fetch_subjects
import configuration

rec = __import__('3_MC/reconstruct')
tr = __import__('3_MC/train')


# ---------------- Configurations ----------------------------
# Settings
parser = argparse.ArgumentParser(description='dliqt-tensorflow-implementation')
parser = configuration.add_arguments_standard(parser=parser)
parser.add_argument('--base_input_dir', type=str, default='/SAN/vision/hcp/Ryu/non-HCP', help='base directory where the input low-res images are stored')
parser.add_argument('--base_recon_dir', type=str, default='/SAN/vision/hcp/Ryu/non-HCP', help='base directory where the output images are saved')
parser.add_argument('--dataset', type=str, default='tumour', help='options availble: prisma, tumoour, ms, hcp1, hcp2')
arg = parser.parse_args()
opt = vars(arg)

# GPUs devices:
os.environ["CUDA_VISIBLE_DEVICES"] = opt["gpu"]

# data/task:
opt['train_size']=int(opt['no_patches']*opt['no_subjects'])
opt['patchlib_idx'] = 1

# Print options
for key, val in opt.iteritems():
    print("{}: {}".format(key, val))

# ----------------- Directories set-up --------------------------
# base directories:
# base_input_dir = '/Users/ryutarotanno/DeepLearning/nsampler/data'
# base_recon_dir = '/Users/ryutarotanno/DeepLearning/nsampler/recon/miccai2017'

non_HCP = {'prisma':{'subdir':'Prisma/Diffusion_2.5mm',
                     'dt_file':'dt_all_'},
           'tumour':{'subdir':'Tumour/06_FORI',
                     'dt_file':'dt_b700_'},
           'ms':{'subdir':'MS/B0410637-2010-00411',
                 'dt_file':'dt_test_b1200_'},
           'hcp1':{'subdir':'HCP/117324',
                   'dt_file':'dt_b1000_lowres_2_'},
           'hcp2': {'subdir': 'HCP/904044',
                    'dt_file': 'dt_b1000_lowres_2_'},
           }

# # Make directories to store results:
# base_dir = os.path.join(opt['base_dir'], opt['experiment'], )
# opt.update({
#     "data_dir": os.path.join(base_dir,"data"),
#     "save_dir": os.path.join(base_dir,"models"),
#     "log_dir": os.path.join(base_dir,"log"),
#     "recon_dir": os.path.join(base_dir,"recon")})


# Reconstruct:
key = opt['dataset']
print('Reconstructing: %s' %(non_HCP[key]['subdir'],))
opt['subject']=non_HCP[key]['subdir']
opt['gt_dir'] = os.path.join(opt['base_input_dir'], opt['experiment'])
opt['input_file_name'] = non_HCP[key]['dt_file']
opt['recon_dir'] = os.path.join(opt['base_recon_dir'], opt['experiment'])
rec.sr_reconstruct_nonhcp(opt, dataset_type=key)



