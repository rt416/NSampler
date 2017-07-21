""" Main script for running statistical analysis for MICCAI 2017 """
import tensorflow as tf

# Options
opt = {}

# Network:
opt['method'] = 'cnn_heteroscedastic'
opt['valid'] = False  # pick the best model with the minimal cost (instead of RMSE).
opt['n_h1'] = 50
opt['n_h2'] = 2*opt['n_h1']
opt['n_h3'] = 10

# Training
opt['optimizer'] = tf.train.AdamOptimizer
opt['dropout_rate'] = 0.0
opt['learning_rate'] = 1e-3
opt['L1_reg'] = 0.00
opt['L2_reg'] = 1e-5
opt['n_epochs'] = 200
opt['batch_size'] = 12
opt['validation_fraction'] = 0.5
opt['shuffle'] = True
opt['validation_fraction'] = 0.5

# Data/task:
opt['cohort'] ='Diverse'
opt['no_subjects'] = 8
opt['patchlib_idx'] = 1
opt['b_value'] = 1000
opt['no_randomisation'] = 1
opt['shuffle_data'] = True
opt['chunks'] = True  # set True if you want to chunk the HDF5 file.

opt['subsampling_rate'] = 343
opt['upsampling_rate'] = 2
opt['input_radius'] = 5
opt['receptive_field_radius'] = 2
output_radius = ((2*opt['input_radius']-2*opt['receptive_field_radius']+1)//2)
opt['output_radius'] = output_radius
opt['no_channels'] = 6
opt['transform_opt'] = 'standard'  # preprocessing of input/output variables

# Dir:
opt['data_dir'] = '/SAN/vision/hcp/Ryu/IPMI2016/TrainingSet/' # '../data/'
opt['recon_dir'] = '/Users/ryutarotanno/tmp/recon'

opt['gt_dir'] = '/Users/ryutarotanno/DeepLearning/Test_1/data/HCP/' # ground truth dir
opt['subpath'] = 'T1w/Diffusion'

opt['mask_dir'] ='/Users/ryutarotanno/tmp/recon'

# Others:
opt['mc_no_samples'] = 100


# Plot rmse
opt['subject'] = '904044'
choose = input("Press: "
               "\n1 - plot rmse vs uncertainty "
               "\n2 - compute rmse, psnr, mssim"
               "\n3 - plot ROC curves"
               "\nselect - ")

import analysis_miccai2017


if choose == 1:
    analysis_miccai2017.plot_rmse_vs_uncertainty(opt)
elif choose==2:
    err = analysis_miccai2017.compute_err(opt)
elif choose == 3:
    analysis_miccai2017.get_ROC(opt)


