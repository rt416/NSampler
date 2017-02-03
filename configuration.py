""" Set default configuration """
import tensorflow as tf

def set_default():
    # Options
    opt = {}

    # Network:
    opt['method'] = 'cnn_simple'
    opt['valid'] = False  # True if the best model achives the lowest loss intead of MSE.
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
    opt['b_value'] = 1000
    opt['patchlib_idx'] = 1
    opt['no_randomisation'] = 1
    opt['shuffle_data'] = True
    opt['chunks'] = True  # set True if you want to chunk the HDF5 file.
    opt['subject'] = '904044'

    opt['subsampling_rate'] = 343
    opt['upsampling_rate'] = 2
    opt['input_radius'] = 5
    opt['receptive_field_radius'] = 2
    output_radius = ((2*opt['input_radius']-2*opt['receptive_field_radius']+1)//2)
    opt['output_radius'] = output_radius
    opt['no_channels'] = 6
    opt['transform_opt'] = 'standard'  # preprocessing of input/output variables
    opt['mc_no_samples'] = 100

    # Dir:
    opt['data_dir'] = '/SAN/vision/hcp/Ryu/IPMI2016/TrainingSet/' # '../data/'
    opt['save_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/01feb2017/models'
    opt['log_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/01feb2017/log'
    opt['recon_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/01feb2017/recon'
    opt['mask_dir'] = '/SAN/vision/hcp/Ryu/miccai2017/recon'

    opt['save_train_dir_tmp'] = '/SAN/vision/hcp/Ryu/IPMI2016/HCP'
    opt['save_train_dir'] = '/SAN/vision/hcp/Ryu/IPMI2016/TrainingSet/'

    opt['gt_dir'] = '/SAN/vision/hcp/DCA_HCP.2013.3_Proc/'  # ground truth dir
    opt['subpath'] = 'T1w/Diffusion'
    opt['input_file_name'] = 'dt_b1000_lowres_' + str(opt['upsampling_rate']) + '_'
    return opt

