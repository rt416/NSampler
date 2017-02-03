""" Perform reconstrution on non- HCP dataset (Prisma, MS, Tumour).
It assumes that DTI is available as nifti files."""

import configuration

# Options
opt = configuration.set_default()

# Update parameters:
opt['method'] = 'cnn_heteroscedastic'
opt['valid'] = True  # pick the best model with the minimal cost (instead of RMSE).

# Training
opt['dropout_rate'] = 0.0

# Data/task:
opt['patchlib_idx'] = 1
opt['subsampling_rate'] = 343
opt['upsampling_rate'] = 2
opt['input_radius'] = 5
opt['receptive_field_radius'] = 2
output_radius = ((2*opt['input_radius']-2*opt['receptive_field_radius']+1)//2)
opt['output_radius'] = output_radius
opt['no_channels'] = 6

# Dir:
opt['gt_dir'] = '/SAN/vision/hcp/DCA_HCP.2013.3_Proc/'  # ground truth dir
opt['subpath'] = 'T1w/Diffusion'
opt['data_dir'] = '/SAN/vision/hcp/Ryu/IPMI2016/TrainingSet/'
opt['recon_dir'] = '/Users/ryutarotanno/tmp/recon'
opt['gt_dir'] = '/Users/ryutarotanno/DeepLearning/Test_1/data/HCP/' # ground truth dir
opt['subpath'] = 'T1w/Diffusion'
opt['mask_dir'] ='/Users/ryutarotanno/tmp/recon'

base_dir = '/Users/ryutarotanno/DeepLearning/nsampler/data/'
non_HCP = {'prisma':{'dir':base_dir+'Prisma/Diffusion_2.5mm',
                     'dt_file':'dt_all_'},
           'tumour':{'dir':base_dir+'Tumour/06_FORI',
                     'dt_file':'dt_b700_'},
           'ms':{'dir':base_dir+'MS/B0410637-2010-00411',
                 'dt_file':'dt_b1200_'}
           }

for key in non_HCP:
    opt['gt_dir'] = '/Users/ryutarotanno/DeepLearning/Test_1/data/HCP/'  # ground truth dir

opt['input_file_name'] = 'dt_b1000_lowres_' + str(opt['upsampling_rate']) + '_'

