""" Main script for creating training data """

import sr_datageneration

opt = {}
opt['data_parent_dir'] = '/SAN/vision/hcp/DCA_HCP.2013.3_Proc'
opt['data_subpath'] = 'T1w/Diffusion'
opt['save_parent_dir'] = '/SAN/vision/hcp/Ryu/IPMI2016/HCP'
opt['save_dir'] = '/SAN/vision/hcp/Ryu/IPMI2016/TrainingSet'
opt['cohort'] = 'Diverse'

opt['no_randomisation'] = 1
opt['subsampling_rate'] = input("Enter subsampling rate: ")
opt['b_value'] = 1000

opt['upsampling_rate'] = input("Enter upsampling rate: ")
opt['receptive_field_radius'] = input("Enter receptive field radius: ")
opt['input_radius'] = input("Enter input radius: ")
opt['no_channels'] = 6
opt['shuffle'] = True

chunks = input("Press 1 if you want to chunk the HDF5 file.")

if chunks == 1:
    opt['chunks'] = True
else:
    opt['chunks'] = False

sr_datageneration.create_training_data(opt)