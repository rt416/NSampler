"""Ryu: main experiments script for non-HCP data reconstruction """
import argparse
import os
import configuration
from common.data_utils import fetch_subjects
import b_Probabilistic.reconstruct as reconstruct


# ---------------- Configurations ----------------------------
# Settings
parser = argparse.ArgumentParser(description='dliqt-tensorflow-implementation')
parser = configuration.add_arguments_standard(parser=parser)
parser.add_argument('--base_input_dir', type=str, default='/SAN/vision/hcp/Ryu/non-HCP', help='base directory where the input low-res images are stored')
parser.add_argument('--base_recon_dir', type=str, default='/SAN/vision/hcp/Ryu/non-HCP/recon', help='base directory where the output images are saved')
parser.add_argument('--dataset', type=str, default='tumour', help='options availble: prisma, tumoour, ms, hcp1, hcp2')
parser.add_argument('--subject', type=str, default=None, help='subject name')

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

if opt['is_map']:
    opt['no_channels'] = 22

# ----------------- Directories set-up --------------------------
# base directories:
# base_input_dir = '/Users/ryutarotanno/DeepLearning/nsampler/data'
# base_recon_dir = '/Users/ryutarotanno/DeepLearning/nsampler/recon/miccai2017'

non_HCP = {'prisma':{'subdir':'Prisma/Diffusion_2.5mm',
                     'dt_file':'dt_all_'},
           'prisma_map': {'subdir': 'Prisma/Diffusion_2.5mm',
                          'dt_file': 'h4_all_'},
           'tumour':{'subdir':'Tumour/06_FORI',
                     'dt_file':'dt_b700_'},
           'ms':{'subdir':'MS/B0410637-2010-00411',
                 'dt_file':'dt_test_b1200_'},
           'hcp1':{'subdir':'HCP/117324',
                   'dt_file':'dt_b1000_lowres_2_',
                   'gt_header':'dt_b1000_'},
           'hcp2': {'subdir': 'HCP/904044',
                    'dt_file': 'dt_b1000_lowres_2_',
                    'gt_header': 'dt_b1000_'},
           'hcp1_map': {'subdir': 'HCP/117324',
                        'dt_file': 'h4_all_lowres_2_',
                        'gt_header': 'h4_all_'},
           'hcp2_map': {'subdir': 'HCP/904044',
                        'dt_file': 'h4_all_lowres_2_',
                        'gt_header': 'h4_all_'},
           'hcp1_x2': {'subdir': 'HCP/117324',
                       'dt_file': 'dt_b1000_'},
           'hcp2_x2': {'subdir': 'HCP/904044',
                       'dt_file': 'dt_b1000_'},
           'hcp1_map_x2': {'subdir': 'HCP/117324',
                           'dt_file': 'h4_all_'},
           'hcp2_map_x2': {'subdir': 'HCP/904044',
                           'dt_file': 'h4_all_'},
           'hcp_abnormal': {'subdir': 'HCP.S1200/105620',
                            'dt_file': 'dt_b1000_lowres_2_',
                            'gt_header': 'dt_b1000_'},
           'hcp_abnormal_map': {'subdir': 'HCP.S1200/105620',
                                'dt_file': 'h4_all_lowres_2_',
                                'gt_header': 'h4_all_'},
           'hcp_abnormal_x2': {'subdir': 'HCP.S1200/105620',
                               'dt_file': 'dt_b1000_'},
           'hcp_abnormal_map_x2': {'subdir': 'HCP.S1200/105620',
                                   'dt_file': 'h4_all_'}
           }

# Make directories to store results:
base_dir = os.path.join(opt['base_dir'], opt['experiment'], )
opt.update({
    "data_dir": os.path.join(base_dir,"data"),
    "save_dir": os.path.join(base_dir,"models"),
    "log_dir": os.path.join(base_dir,"log"),
    "recon_dir": os.path.join(base_dir,"recon"),
    "stats_dir": os.path.join(base_dir, "stats")})

# Reconstruct:
key = opt['dataset']
print('Reconstructing: %s' %(non_HCP[key]['subdir'],))
if opt['subject'] is not None:
    print("Subject ID specified: %s " % (opt['subject'],))
else:
    opt['subject'] = non_HCP[key]['subdir']

opt['gt_dir'] = os.path.join(opt['base_input_dir'])
opt['recon_dir'] = os.path.join(opt['base_recon_dir'], opt['experiment'])

opt['input_file_name'] = non_HCP[key]['dt_file']

if 'gt_header' in non_HCP[key].keys():  # for some datasets, ground truths are not available.
    opt['gt_header'] = non_HCP[key]['gt_header']
    opt['save_as_ijk'] = False
    opt['gt_available'] = True
else:
    opt['gt_header'] = None
    opt['save_as_ijk'] = True
    opt['gt_available'] = False

opt['output_file_name'] = opt['input_file_name']+'x%i_recon_mc=%i.npy' % (opt['upsampling_rate'], opt["mc_no_samples"])
opt['output_std_file_name'] = 'std_'+opt['output_file_name']

reconstruct.sr_reconstruct_nonhcp(opt, dataset_type=key)



