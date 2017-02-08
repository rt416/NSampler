""" Reconstruction file """

import os
import sys
import timeit

import cPickle as pkl
import h5py
import numpy as np
import tensorflow as tf
import analysis_miccai2017

import sr_preprocess as pp
import sr_utility
import models
from train import define_checkpoint, name_network
from sr_datageneration import forward_periodic_shuffle


# Pad the volumes:
def dt_pad(dt_volume, opt):
    """ Pad a volume with zeros before reconstruction """

    # -------------------- Load parameters ------------------------------- :
    upsampling_rate = opt['upsampling_rate']
    receptive_field_radius = opt['receptive_field_radius']
    input_radius = opt['input_radius']
    output_radius = opt['output_radius']

    # --------------------- Pad ---------------:
    # Pad with zeros so all brain-voxel-centred pathces are extractable and
    # each dimension is divisible by upsampling rate.
    dim_x_highres, dim_y_highres, dim_z_highres, dim_channels = dt_volume.shape
    pad_min = max((input_radius + 1) * upsampling_rate,
                  (output_radius + 1) * upsampling_rate)  # padding width

    pad_x = pad_min if np.mod(2*pad_min + dim_x_highres, upsampling_rate) == 0 \
        else pad_min + \
             (upsampling_rate - np.mod(2*pad_min + dim_x_highres, upsampling_rate))

    pad_y = pad_min if np.mod(2*pad_min + dim_y_highres, upsampling_rate) == 0 \
        else pad_min + \
             (upsampling_rate - np.mod(2*pad_min + dim_y_highres, upsampling_rate))

    pad_z = pad_min if np.mod(2*pad_min + dim_z_highres, upsampling_rate) == 0 \
        else pad_min + \
             (upsampling_rate - np.mod(2*pad_min + dim_z_highres, upsampling_rate))

    dt_volume[:, :, :, 1] += 1

    pd = ((pad_min, pad_x),
          (pad_min, pad_y),
          (pad_min, pad_z), (0, 0))

    dt_volume = np.pad(dt_volume,
                       pad_width=pd,
                       mode='constant', constant_values=0)

    dt_volume[:, :, :, 1] -= 1

    return dt_volume, pd

# Trim the volume:
def dt_trim(dt_volume, pd):
    """ Trim the dt volume back to the original size
    according to the padding applied

    Args:
        dt_volume (numpy array): 4D numpy dt volume
        pd (tuple): padding applied to dt_volume
    """
    dt_volume = dt_volume[pd[0][0]:-pd[0][1],
                          pd[1][0]:-pd[1][1],
                          pd[2][0]:-pd[2][1]]
    return dt_volume


def mc_inference_hetero_FA_and_MD(fn, fn_std, fd, opt):
    """ Compute the mean and std of samples drawn from stochastic function"""
    no_samples = opt['mc_no_samples']
    if opt['method']=='cnn_dropout' or \
       opt['method']=='cnn_gaussian_dropout' or \
       opt['method']=='cnn_variational_dropout' or \
       opt['method']=='cnn_variational_dropout_layerwise' or \
       opt['method']=='cnn_variational_dropout_channelwise' or \
       opt['method']=='cnn_variational_dropout_average':

        md_sum_out = 0.0
        md_sum_out2 = 0.0
        fa_sum_out = 0.0
        fa_sum_out2 = 0.0

        for i in xrange(no_samples):
            current = 1. * fn.eval(feed_dict=fd)
            current = forward_periodic_shuffle(current, opt['upsampling_rate'])
            md_sample, fa_sample = sr_utility.compute_MD_and_FA(current)
            md_sum_out += md_sample
            md_sum_out2 += md_sample ** 2
            fa_sum_out += fa_sample
            fa_sum_out2 += fa_sample ** 2

        md_mean = md_sum_out / no_samples
        md_std = np.sqrt(np.abs(md_sum_out2 -
                                2 * md_mean * md_sum_out +
                                no_samples * md_mean ** 2) / no_samples)

        fa_mean = fa_sum_out / no_samples
        fa_std = np.sqrt(np.abs(fa_sum_out2 -
                                2 * fa_mean * fa_sum_out +
                                no_samples * fa_mean ** 2) / no_samples)

    elif opt['method'] == 'cnn_heteroscedastic_variational' or \
       opt['method'] == 'cnn_heteroscedastic_variational_layerwise' or \
       opt['method'] == 'cnn_heteroscedastic_variational_channelwise' or \
       opt['method'] == 'cnn_heteroscedastic_variational_average' or \
       opt['method'] == 'cnn_heteroscedastic_variational_downsc' or \
       opt['method'] == 'cnn_heteroscedastic_variational_upsc' or \
       opt['method'] == 'cnn_heteroscedastic_variational_layerwise_downsc' or \
       opt['method'] == 'cnn_heteroscedastic_variational_channelwise_downsc' or \
       opt['method'] == 'cnn_heteroscedastic_variational_hybrid_control' or \
       opt['method'] == 'cnn_heteroscedastic_variational_channelwise_hybrid_control' or \
       opt['method'] == 'cnn_heteroscedastic_variational_downsc_control' or \
       opt['method'] == 'cnn_heteroscedastic_variational_upsc_control':

        md_sum_out = 0.0
        md_sum_out2 = 0.0
        fa_sum_out = 0.0
        fa_sum_out2 = 0.0

        like_std = fn_std.eval(feed_dict=fd)  # add noise from the likelihood model.

        for i in xrange(no_samples):
            dti_sample = np.random.normal(0, like_std)
            current = 1. * fn.eval(feed_dict=fd) + dti_sample
            current = forward_periodic_shuffle(current, opt['upsampling_rate'])
            md_sample, fa_sample = sr_utility.compute_MD_and_FA(current)
            md_sum_out += md_sample
            md_sum_out2 += md_sample ** 2
            fa_sum_out += fa_sample
            fa_sum_out2 += fa_sample ** 2

        md_mean = md_sum_out / no_samples
        md_std = np.sqrt(np.abs(md_sum_out2 -
                                2 * md_mean * md_sum_out +
                                no_samples * md_mean ** 2) / no_samples)

        fa_mean = fa_sum_out / no_samples
        fa_std = np.sqrt(np.abs(fa_sum_out2 -
                                2 * fa_mean * fa_sum_out +
                                no_samples * fa_mean ** 2) / no_samples)

    else:
        raise Exception('The specified method does not support MC inference.')
        mean = fn.eval(feed_dict=fd)
        std_m = None
        std_d = None
    return md_mean, md_std, fa_mean, fa_std


# Reconstruct using the specified NN:
def super_resolve_FA_and_MD(dt_lowres, opt):

    """Perform a patch-based super-resolution on a given low-res image.
    Args:
        dt_lowres (numpy array): a low-res diffusion tensor image volume
        opt (dict):
    Returns:
        the estimated FA and MD.
    """
    # -------------------------- Load in the parameters ----------------------:

    # Network details:
    method = opt['method']
    dropout_rate = opt['dropout_rate']
    n_h1 = opt['n_h1']
    n_h2 = opt['n_h2']
    n_h3 = opt['n_h3']

    # Training set details:
    cohort = opt['cohort']
    no_subjects = opt['no_subjects']
    no_channels = opt['no_channels']
    subsampling_rate = opt['subsampling_rate']

    # Input/Output details:
    upsampling_rate = opt['upsampling_rate']
    receptive_field_radius = opt['receptive_field_radius']
    input_radius = opt['input_radius']
    output_radius = opt['output_radius']

    # get the dir where the network is saved
    network_dir = define_checkpoint(opt)

    # --------------------------- Define the model--------------------------:

    # Specify the network parameters to be restored:
    model_details = pkl.load(open(os.path.join(network_dir,'settings.pkl'), 'rb'))
    nn_file = os.path.join(network_dir, "model-" + str(model_details['step_save']))
    opt.update(model_details)

    print('... defining the network model %s .' % method)
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None,
                                        2 * input_radius + 1,
                                        2 * input_radius + 1,
                                        2 * input_radius + 1,
                                        no_channels],
                           name='lo_res')
        y = tf.placeholder(tf.float32, [None,
                                        2 * output_radius + 1,
                                        2 * output_radius + 1,
                                        2 * output_radius + 1,
                                        no_channels * (upsampling_rate ** 3)],
                           name='hi_res')

    with tf.name_scope('learning_rate'):
        lr = tf.placeholder(tf.float32, [], name='learning_rate')

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)  # keep probability for dropout

    with tf.name_scope('tradeoff'):
        trade_off = tf.placeholder(tf.float32)  # keep probability for dropout

    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Load normalisation parameters and define prediction:
    transform = pkl.load(open(os.path.join(network_dir, 'transforms.pkl'), 'rb'))
    y_pred, y_pred_std = models.scaled_prediction(method, x, y,
                                                  keep_prob, transform,
                                                  opt, trade_off)

    # -------------------------- Reconstruct --------------------------------:
    # Restore all the variables and perform reconstruction:
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, nn_file)
        print("Model restored.")

        # Apply padding:
        # print("Size of dt_lowres before padding: %s", (dt_lowres.shape,))
        dt_lowres, padding = dt_pad(dt_volume=dt_lowres, opt=opt)
        mask = dt_lowres[...,0]!= -1

        # Prepare high-res skeleton:
        dt_md_mean = np.zeros(dt_lowres.shape[:-1])  # data uncertainty

        dt_md_std = np.zeros(dt_lowres.shape[:-1])  # data uncertainty

        dt_fa_mean = np.zeros(dt_lowres.shape[:-1])

        dt_fa_std = np.zeros(dt_lowres.shape[:-1])  # model uncertainty

        # Down-sample:
        dt_lowres = dt_lowres[::upsampling_rate,
                              ::upsampling_rate,
                              ::upsampling_rate, :]

        # Reconstruct:
        (xsize, ysize, zsize, comp) = dt_lowres.shape
        recon_indx = [(i, j, k) for k in np.arange(input_radius + 1,
                                                   zsize - input_radius + 1,
                                                   2 * output_radius + 1)
                                for j in np.arange(input_radius + 1,
                                                   ysize - input_radius + 1,
                                                   2 * output_radius + 1)
                                for i in np.arange(input_radius + 1,
                                                   xsize - input_radius + 1,
                                                   2 * output_radius + 1)]
        for i, j, k in recon_indx:
            sys.stdout.flush()
            sys.stdout.write('\tSlice %i of %i.\r' % (k, zsize))

            ipatch_tmp = dt_lowres[(i - input_radius - 1):(i + input_radius),
                                   (j - input_radius - 1):(j + input_radius),
                                   (k - input_radius - 1):(k + input_radius), 2:comp]

            ipatch = ipatch_tmp[np.newaxis, ...]

            # Predict high-res patch:
            fd = {x: ipatch, keep_prob: (1.0 - dropout_rate), trade_off: 0.0}
            md_mean, md_std, fa_mean, fa_std = \
                mc_inference_hetero_FA_and_MD(y_pred, y_pred_std, fd, opt)

            dt_md_mean[upsampling_rate * (i - output_radius - 1):
                       upsampling_rate * (i + output_radius),
                       upsampling_rate * (j - output_radius - 1):
                       upsampling_rate * (j + output_radius),
                       upsampling_rate * (k - output_radius - 1):
                       upsampling_rate * (k + output_radius)] \
            = md_mean

            dt_md_std[upsampling_rate * (i - output_radius - 1):
            upsampling_rate * (i + output_radius),
            upsampling_rate * (j - output_radius - 1):
            upsampling_rate * (j + output_radius),
            upsampling_rate * (k - output_radius - 1):
            upsampling_rate * (k + output_radius)] \
            = md_std

            dt_fa_mean[upsampling_rate * (i - output_radius - 1):
                       upsampling_rate * (i + output_radius),
                       upsampling_rate * (j - output_radius - 1):
                       upsampling_rate * (j + output_radius),
                       upsampling_rate * (k - output_radius - 1):
                       upsampling_rate * (k + output_radius)] \
            = fa_mean

            dt_fa_std[upsampling_rate * (i - output_radius - 1):
            upsampling_rate * (i + output_radius),
            upsampling_rate * (j - output_radius - 1):
            upsampling_rate * (j + output_radius),
            upsampling_rate * (k - output_radius - 1):
            upsampling_rate * (k + output_radius)] \
            = fa_std

        # Trim unnecessary padding:
        dt_md_mean = dt_trim(dt_md_mean, padding)
        dt_md_std = dt_trim(dt_md_std, padding)
        dt_fa_mean = dt_trim(dt_fa_mean, padding)
        dt_fa_std = dt_trim(dt_fa_std, padding)


        dt_md_mean = dt_md_mean * mask
        dt_md_std = dt_md_std * mask
        dt_fa_mean = dt_fa_mean * mask
        dt_fa_std = dt_fa_std * mask
        print("Size of dt_hires after trimming: %s", (dt_md_mean.shape,))
    return dt_md_mean, dt_md_std, dt_fa_mean, dt_fa_std


# Main reconstruction code:
def sr_reconstruct_FA_and_MD(opt):
    # load parameters:
    recon_dir = opt['recon_dir']
    gt_dir = opt['gt_dir']
    subpath = opt['subpath']
    subject = opt['subject']
    input_file_name = opt['input_file_name']

    # Load the input low-res DT image:
    print('... loading the test low-res image ...')
    dt_lowres = sr_utility.read_dt_volume(os.path.join(gt_dir, subject,
                                                       subpath, input_file_name))

    # clear the graph (is it necessary?)
    tf.reset_default_graph()

    # Reconstruct:
    start_time = timeit.default_timer()
    nn_dir = name_network(opt)
    print('\nPerformn MC-reconstruction of high-res dti with the network: \n%s.' % nn_dir)
    dt_md_mean, dt_md_std, dt_fa_mean, dt_fa_std = super_resolve_FA_and_MD(dt_lowres, opt)

    # Save:
    md_mean_file = os.path.join(recon_dir, subject, nn_dir, 'dt_recon_MD_mean.npy')
    md_std_file = os.path.join(recon_dir, subject, nn_dir, 'dt_recon_MD_std.npy')
    fa_mean_file = os.path.join(recon_dir, subject, nn_dir, 'dt_recon_FA_mean.npy')
    fa_std_file = os.path.join(recon_dir, subject, nn_dir, 'dt_recon_FA_std.npy')

    print('... saving MC-predicted MD as %s' % md_mean_file)
    if not (os.path.exists(os.path.join(recon_dir, subject))):
        os.mkdir(os.path.join(recon_dir, subject))
    if not (os.path.exists(os.path.join(recon_dir, subject, nn_dir))):
        os.mkdir(os.path.join(recon_dir, subject, nn_dir))
    np.save(md_mean_file, dt_md_mean)
    np.save(md_std_file, dt_md_std)
    np.save(fa_mean_file, dt_fa_mean)
    np.save(fa_std_file, dt_fa_std)
    end_time = timeit.default_timer()
    print('\nIt took %f secs. \n' % (end_time - start_time))

    # Save each estimated dti/std separately as a nifti file for visualisation:
    print('\nSave as nii files...')
    mean_md, base = os.path.splitext(md_mean_file)
    std_md, base = os.path.splitext(md_std_file)
    mean_fa, base = os.path.splitext(fa_mean_file)
    std_fa, base = os.path.splitext(fa_std_file)
    sr_utility.ndarray_to_nifti(dt_md_mean, mean_md+'.nii')
    sr_utility.ndarray_to_nifti(dt_md_std, std_md + '.nii')
    sr_utility.ndarray_to_nifti(dt_fa_mean, mean_fa + '.nii')
    sr_utility.ndarray_to_nifti(dt_fa_std, std_fa + '.nii')

    # Compute the reconstruction error:
    mask_file = 'mask_us=' + str(opt['upsampling_rate']) + \
                '_rec=' + str(5) + '.nii'
    mask_dir = opt['mask_dir']

    # calculate the errors too for FA and MD, and save them:
    md_gt_file = os.path.join(recon_dir, subject, 'maps','dt_b1000_MD.nii')
    fa_gt_file = os.path.join(recon_dir, subject, 'maps','dt_b1000_FA.nii')
    save_maps_dir = os.path.join(recon_dir, subject, 'maps')
    if not(os.path.exists(fa_gt_file)) or not(os.path.exists(md_gt_file)):
        print('Ground truth FA or MD does not exist. Compute them ...')
        dti_gt_root_dir =  os.path.join(gt_dir, subject, subpath)
        dti_gt_file = dti_gt_root_dir + '/dt_b1000_'
        os.makedirs(save_maps_dir)
        analysis_miccai2017._MD_FA(dti_gt_file, save_dir=save_maps_dir)

    sr_utility.compute_rmse_nii(nii_1=md_gt_file, nii_2=mean_md+'.nii')
    sr_utility.compute_rmse_nii(nii_1=fa_gt_file, nii_2=mean_fa+'.nii')
