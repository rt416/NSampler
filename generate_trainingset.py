""" Main script for creating training data """

import sr_preprocessing_new

# define the configurations of the training data:
opt = {}

opt['data_parent_dir'] = '/home/rtanno/Shared/HDD/SuperRes/HCP'
opt['data_subpath'] = 'T1w/Diffusion'
opt['save_dir'] = '/home/rtanno/Shared/HDD/SuperRes/Training/IPMI'
opt['cohort'] = "Diverse"

opt['no_randomisation'] = 1  # number of distinct training sets you want to create.
opt['sampling_rate'] = 32
opt['b_value'] = 1000

opt['upsampling_rate'] = 2
opt['receptive_field_radius'] = 2
opt['input_radius'] = 5
opt['no_channels'] = 6

opt['no_chunks'] = 100
opt['shuffle'] = True

sr_preprocessing_new.create_training_data(opt)

