""" Reconstruction file """
import timeit
import cPickle as pkl
import numpy as np
import tensorflow as tf
import os
import sys
import nibabel as nib

from train import get_output_radius
import common.sr_utility as sr_utility
from common.sr_utility import forward_periodic_shuffle
from common.utils import name_network, name_patchlib, set_network_config, define_checkpoint, mc_inference, mc_inference_decompose, mc_inference_MD_CFA_MD, dt_trim, dt_pad, clip_image, save_stats
from common.sr_analysis import compare_images_and_get_stats, compute_differencemaps


# Main reconstruction code:
def sr_reconstruct(opt):
    # Save displayed output to a text file:
    if opt['disp']:
        f = open(opt['save_dir'] + '/' + name_network(opt) + '/output_recon.txt', 'ab')
        # Redirect all the outputs to the text file:
        print("Redirecting the output to: "
              + opt['save_dir'] + '/' + name_network(opt) + "/output_recon.txt")
        sys.stdout = f

    # Define directory and file names:
    print('\nStart reconstruction! \n')
    recon_dir = opt['recon_dir']
    gt_dir = opt['gt_dir']
    subpath = opt['subpath']
    subject = opt['subject']
    no_channels = opt['no_channels']
    input_file_name, _ = opt['input_file_name'].split('{')
    gt_header, _ = opt['gt_header'].split('{')
    nn_dir = name_network(opt)
    output_file = os.path.join(recon_dir, subject, nn_dir, opt['output_file_name'])
    save_stats_dir = os.path.join(opt['stats_dir'], nn_dir)
    if not (os.path.exists(save_stats_dir)):
        os.makedirs(save_stats_dir)
    # ------------------------- Perform synthesis -----------------------------
    print('\n ... reconstructing high-res dti with network: \n%s.' % nn_dir)
    if os.path.exists(output_file):
        print("Reconstruction already exists: " + output_file)
        print("Move on. ")
    else:
        # Load the input low-res DT image:
        print('... loading the test low-res image ...')
        input_file = os.path.join(gt_dir, subject, subpath, input_file_name)
        dt_lowres = sr_utility.read_dt_volume(input_file, no_channels=no_channels)

        # Seems to aggravate performance, so currently ignored.
        # Clip the input DTI:
        # if opt["is_clip"]:
        #     print('... clipping the input image')
        #     dt_lowres[...,-no_channels:]=clip_image(dt_lowres[...,-no_channels:],
        #                                             bkgv=opt["background_value"])

        # Reconstruct:
        tf.reset_default_graph()
        start_time = timeit.default_timer()
        if opt['decompose']:
            dt_hr, dt_var_model, dt_var_random = super_resolve_decompose(dt_lowres, opt)
        else:
            dt_hr, dt_std = super_resolve(dt_lowres, opt)

        end_time = timeit.default_timer()
        print('\nIt took %f secs. \n' % (end_time - start_time))

        # Post-processing:
        if opt["postprocess"]:
            # Clipping:
            print('... post-processing the output image')
            dt_hr[..., -no_channels:] = clip_image(dt_hr[..., -no_channels:], bkgv=opt["background_value"], tail_perc=0.01, head_perc=99.99)

    # ---------------------------  Save stuff --------------------------------
    if opt["not_save"]:
        print("Selected not to save the outputs")
    elif os.path.exists(output_file):
        print("Reconstruction already exists: " + output_file)
    else:
        print('... saving MC-estimated high-res volume and its uncertainty as %s' % output_file)
        if not (os.path.exists(os.path.join(recon_dir, subject,nn_dir))):
            os.makedirs(os.path.join(recon_dir, subject, nn_dir))

        # Save predicted high-res brain volume:
        np.save(output_file, dt_hr)
        print('\nSave each super-resolved channel separately as a nii file ...')
        __, recon_file = os.path.split(output_file)
        sr_utility.save_as_nifti(recon_file,
                                 os.path.join(recon_dir, subject, nn_dir),
                                 os.path.join(gt_dir, subject, subpath),
                                 save_as_ijk=opt['save_as_ijk'],
                                 no_channels=no_channels,
                                 gt_header=gt_header)

        # Save uncertainty for probabilistic models:
        if opt['hetero'] or opt['vardrop']:
            if opt['decompose']:
                uncertainty_random_file = os.path.join(recon_dir, subject, nn_dir, opt['output_var_random_file_name'])
                uncertainty_model_file = os.path.join(recon_dir, subject, nn_dir,  opt['output_var_model_file_name'])
                print('... saving random uncertainty as %s' % uncertainty_random_file)
                print('... saving model uncertainty as %s' % uncertainty_model_file)
                np.save(uncertainty_random_file, dt_var_random)
                np.save(uncertainty_model_file, dt_var_model)
                __, var_random_file = os.path.split(uncertainty_random_file)
                __, var_model_file = os.path.split(uncertainty_model_file)
                sr_utility.save_as_nifti(var_random_file,
                                         os.path.join(recon_dir, subject, nn_dir),
                                         os.path.join(gt_dir, subject, subpath),
                                         save_as_ijk=opt['save_as_ijk'],
                                         no_channels=no_channels,
                                         gt_header=gt_header)

                sr_utility.save_as_nifti(var_model_file,
                                         os.path.join(recon_dir, subject, nn_dir),
                                         os.path.join(gt_dir, subject, subpath),
                                         save_as_ijk=opt['save_as_ijk'],
                                         no_channels=no_channels,
                                         gt_header=gt_header)

            else:
                uncertainty_file = os.path.join(recon_dir, subject, nn_dir, opt['output_std_file_name'])
                print('... saving its uncertainty as %s' % uncertainty_file)
                np.save(uncertainty_file, dt_std)
                __, std_file = os.path.split(uncertainty_file)
                print(
                '\nSave the uncertainty separately for respective channels as a nii file ...')
                sr_utility.save_as_nifti(std_file,
                                         os.path.join(recon_dir, subject, nn_dir),
                                         os.path.join(gt_dir, subject, subpath),
                                         save_as_ijk=opt['save_as_ijk'],
                                         no_channels=no_channels,
                                         gt_header=gt_header)

    # ------------- Compute stats (e.g. errors, difference maps, etc) ---------
    print('\nCompute the evaluation statistics ...')
    if not (os.path.exists(opt["recon_dir"])):
        os.makedirs(opt["stats_dir"])

    # load the ground truth image and mask:
    dt_gt = sr_utility.read_dt_volume(nameroot=os.path.join(gt_dir, subject, subpath, gt_header), no_channels=no_channels)
    if os.path.exists(output_file): dt_hr = np.load(output_file)
    mask_file = "mask_us={:d}_rec={:d}.nii".format(opt["upsampling_rate"], 5)
    mask_dir_local = os.path.join(opt["mask_dir"], subject, opt["mask_subpath"], "masks")

    if os.path.exists(os.path.join(mask_dir_local, mask_file)):
        img = nib.load(os.path.join(mask_dir_local, mask_file))
        mask_interior = img.get_data() == 0
        mask = dt_hr[:, :, :, 0] == 0
        complement_mask_noedge = img.get_data() != 0
        mask_edge = mask * complement_mask_noedge

        m, m2, p, s = compare_images_and_get_stats(dt_gt[...,2:], dt_hr[...,2:], mask, "whole")
        m_int, m2_int, p_int, s_int = compare_images_and_get_stats(dt_gt[...,2:], dt_hr[...,2:], mask_interior, "interior")
        m_ed, m2_ed, p_ed, s_ed = compare_images_and_get_stats(dt_gt[...,2:], dt_hr[...,2:], mask_edge, "edge")

        csv_file = os.path.join(save_stats_dir, 'stats.csv')
        headers = ['subject',
                   'RMSE(interior)', 'RMSE(edge)', 'RMSE(whole)',
                   'Median(interior)', 'Median(edge)', 'Median(whole)',
                   'PSNR(interior)', 'PSNR(edge)', 'PSNR(whole)',
                   'MSSIM(interior)', 'MSSIM(edge)', 'MSSIM(whole)']
        stats = [m_int, m_ed, m, m2_int, m2_ed, m2, p_int, p_ed, p, s_int, s_ed, s]
    else:
        print("Mask for the interior region NOT FOUND")
        mask = dt_hr[:, :, :, 0] == 0
        m, m2, p, s = compare_images_and_get_stats(dt_gt[...,2:], dt_hr[...,2:], mask, "whole")
        csv_file = os.path.join(save_stats_dir, 'stats_brain.csv')
        headers = ['subject','RMSE(whole)', 'Median(whole)','PSNR(whole)','MSSIM(whole)']
        stats = [m, m2, p, s]

    # Save the stats to a CSV file:
    save_stats(csv_file, opt['subject'], headers, stats)

    # Compute difference maps and save:
    if opt["not_save"]:
        print(" Difference maps not computed")
    else:
        compute_differencemaps(dt_gt[...,2:], dt_hr[...,2:],
                               mask, output_file, no_channels,
                               save_as_ijk=opt['save_as_ijk'],
                               gt_dir=os.path.join(gt_dir, subject, subpath),
                               gt_header=gt_header)


# ------------------ default reconstruction function -------------------------
def super_resolve(dt_lowres, opt):
    """Perform a patch-based super-resolution on a given low-res image.
    Args:
        dt_lowres (numpy array): a low-res diffusion tensor image volume
        opt (dict):
    Returns:
        the estimated high-res volume
    """

    # --------------------------- Define the model--------------------------:
    # placeholders
    print()
    print("--------------------------")
    print("...Setting up placeholders")
    print('... defining the network model %s .' % opt['method'])
    side = 2*opt["input_radius"] + 1
    x = tf.placeholder(tf.float32,
                       shape=[1,side,side,side,opt['no_channels']],
                       name='input_x')
    phase_train = tf.placeholder(tf.bool, name='phase_train')
    keep_prob = tf.placeholder(tf.float32, name='dropout_rate')
    trade_off = tf.placeholder(tf.float32, name='trade_off')
    num_data = tf.placeholder(tf.float32, name='num_train_data')

    # define network and inference:
    print("...Constructing network: %s \n" % opt['method'])
    net = set_network_config(opt)
    transfile = os.path.join(opt['data_dir'], name_patchlib(opt), 'transforms.pkl')
    transform = pkl.load(open(transfile, 'rb'))
    y_pred, y_std = net.scaled_prediction_mc(x, phase_train, keep_prob,
                                             transform=transform,
                                             trade_off=trade_off,
                                             num_data=num_data,
                                             params=opt["params"],
                                             cov_on=opt["cov_on"],
                                             hetero=opt["hetero"],
                                             vardrop=opt["vardrop"])
    # Compute the output radius:
    opt['output_radius'] = get_output_radius(y_pred, opt['upsampling_rate'], opt['is_shuffle'])

    # Specify the network parameters to be restored:
    network_dir = define_checkpoint(opt)
    model_details = pkl.load(open(os.path.join(network_dir,'settings.pkl'), 'rb'))
    nn_file = os.path.join(network_dir, "model-" + str(model_details['step_save']))

    # -------------------------- Reconstruct --------------------------------:
    # Restore all the variables and perform reconstruction:
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, nn_file)
        print("Model restored.")

        # Apply padding:
        print("Size of dt_lowres before padding: %s", (dt_lowres.shape,))
        dt_lowres, padding = dt_pad(dt_lowres, opt['upsampling_rate'], opt['input_radius'])

        print("Size of dt_lowres after padding: %s", (dt_lowres.shape,))

        # Prepare high-res skeleton:
        dt_hires = np.zeros(dt_lowres.shape)
        dt_hires[:, :, :, 0] = dt_lowres[:, :, :, 0]  # same brain mask as input
        dt_hires[:, :, :, 1] = dt_lowres[:, :, :, 1]

        dt_hires_std = np.zeros(dt_lowres.shape)
        dt_hires_std[:, :, :, 0] = dt_lowres[:, :, :, 0]
        print("Size of dt_hires after padding: %s", (dt_hires.shape,))

        # Downsample:
        dt_lowres = dt_lowres[::opt['upsampling_rate'],
                              ::opt['upsampling_rate'],
                              ::opt['upsampling_rate'], :]

        # Reconstruct:
        (xsize, ysize, zsize, comp) = dt_lowres.shape
        recon_indx = [(i, j, k) for k in np.arange(opt['input_radius']+1,
                                                   zsize-opt['input_radius']+1,
                                                   2*opt['output_radius']+1)
                                for j in np.arange(opt['input_radius']+1,
                                                   ysize-opt['input_radius']+1,
                                                   2*opt['output_radius']+1)
                                for i in np.arange(opt['input_radius']+1,
                                                   xsize-opt['input_radius']+1,
                                                   2*opt['output_radius']+1)]
        for i, j, k in recon_indx:
            sys.stdout.flush()
            sys.stdout.write('\tSlice %i of %i.\r' % (k, zsize))

            ipatch_tmp = dt_lowres[(i - opt['input_radius'] - 1):(i + opt['input_radius']),
                                   (j - opt['input_radius'] - 1):(j + opt['input_radius']),
                                   (k - opt['input_radius'] - 1):(k + opt['input_radius']),
                                    2:comp]

            ipatch_mask = dt_lowres[(i - opt['output_radius'] - 1):(i + opt['output_radius']),
                                    (j - opt['output_radius'] - 1):(j + opt['output_radius']),
                                    (k - opt['output_radius'] - 1):(k + opt['output_radius']),
                                    0]

            # only process if any pixel in the output patch is in the brain.
            if np.max(ipatch_mask) >= 0:
                ipatch = ipatch_tmp[np.newaxis, ...]

                # Estimate high-res patch and its associeated uncertainty:
                fd = {x: ipatch,
                      keep_prob: 1.0-opt['dropout_rate'],
                      trade_off: 1.0,
                      phase_train: False}

                opatch, opatch_std = mc_inference(y_pred, y_std, fd, opt, sess)

                if opt["is_shuffle"]:  # only apply shuffling if necessary
                    opatch = forward_periodic_shuffle(opatch, opt['upsampling_rate'])
                    opatch_std = forward_periodic_shuffle(opatch_std, opt['upsampling_rate'])

                dt_hires[opt['upsampling_rate'] * (i - opt['output_radius'] - 1):
                         opt['upsampling_rate'] * (i + opt['output_radius']),
                         opt['upsampling_rate'] * (j - opt['output_radius'] - 1):
                         opt['upsampling_rate'] * (j + opt['output_radius']),
                         opt['upsampling_rate'] * (k - opt['output_radius'] - 1):
                         opt['upsampling_rate'] * (k + opt['output_radius']),
                         2:] \
                = opatch

                dt_hires_std[opt['upsampling_rate'] * (i - opt['output_radius'] - 1):
                             opt['upsampling_rate'] * (i + opt['output_radius']),
                             opt['upsampling_rate'] * (j - opt['output_radius'] - 1):
                             opt['upsampling_rate'] * (j + opt['output_radius']),
                             opt['upsampling_rate'] * (k - opt['output_radius'] - 1):
                             opt['upsampling_rate'] * (k + opt['output_radius']),
                             2:] \
                = opatch_std

        # Trim unnecessary padding:
        dt_hires = dt_trim(dt_hires, padding)
        dt_hires_std = dt_trim(dt_hires_std, padding)
        mask = dt_hires[:, :, :, 0] !=-1
        dt_hires[...,2:]=dt_hires[...,2:]*mask[..., np.newaxis]
        dt_hires_std[..., 2:] = dt_hires_std[..., 2:] * mask[..., np.newaxis]

        print("Size of dt_hires after trimming: %s", (dt_hires.shape,))
    return dt_hires, dt_hires_std


# --------------- reconstruct with decomposed uncertainty -------------------
def super_resolve_decompose(dt_lowres, opt):
    """Perform a patch-based super-resolution on a given low-res image.
    Args:
        dt_lowres (numpy array): a low-res diffusion tensor image volume
        opt (dict):
    Returns:
        the estimated high-res volume
    """

    # --------------------------- Define the model--------------------------:
    # placeholders
    print()
    print("--------------------------")
    print("...Setting up placeholders")
    print('... defining the network model %s .' % opt['method'])
    side = 2*opt["input_radius"] + 1
    x = tf.placeholder(tf.float32,
                       shape=[1,side,side,side,opt['no_channels']],
                       name='input_x')
    phase_train = tf.placeholder(tf.bool, name='phase_train')
    keep_prob = tf.placeholder(tf.float32, name='dropout_rate')
    trade_off = tf.placeholder(tf.float32, name='trade_off')
    num_data = tf.placeholder(tf.float32, name='num_train_data')

    # define network and inference:
    print("...Constructing network: %s \n" % opt['method'])
    net = set_network_config(opt)
    transfile = os.path.join(opt['data_dir'], name_patchlib(opt), 'transforms.pkl')
    transform = pkl.load(open(transfile, 'rb'))
    y_pred, y_std = net.scaled_prediction_mc(x, phase_train, keep_prob,
                                             transform=transform,
                                             trade_off=trade_off,
                                             num_data=num_data,
                                             params=opt["params"],
                                             cov_on=opt["cov_on"],
                                             hetero=opt["hetero"],
                                             vardrop=opt["vardrop"])
    # Compute the output radius:
    opt['output_radius'] = get_output_radius(y_pred, opt['upsampling_rate'], opt['is_shuffle'])

    # Specify the network parameters to be restored:
    network_dir = define_checkpoint(opt)
    model_details = pkl.load(open(os.path.join(network_dir,'settings.pkl'), 'rb'))
    nn_file = os.path.join(network_dir, "model-" + str(model_details['step_save']))

    # -------------------------- Reconstruct --------------------------------:
    # Restore all the variables and perform reconstruction:
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, nn_file)
        print("Model restored.")

        # Apply padding:
        print("Size of dt_lowres before padding: %s", (dt_lowres.shape,))
        dt_lowres, padding = dt_pad(dt_lowres, opt['upsampling_rate'], opt['input_radius'])

        print("Size of dt_lowres after padding: %s", (dt_lowres.shape,))

        # Prepare high-res skeleton:
        dt_hires = np.zeros(dt_lowres.shape)
        dt_hires[:, :, :, 0] = dt_lowres[:, :, :, 0]  # same brain mask as input
        dt_hires[:, :, :, 1] = dt_lowres[:, :, :, 1]

        dt_var_model = np.zeros(dt_lowres.shape)
        dt_var_model[:, :, :, 0] = dt_lowres[:, :, :, 0]
        dt_var_random = np.zeros(dt_lowres.shape)
        dt_var_random[:, :, :, 0] = dt_lowres[:, :, :, 0]

        print("Size of dt_hires after padding: %s", (dt_hires.shape,))

        # Downsample:
        dt_lowres = dt_lowres[::opt['upsampling_rate'],
                              ::opt['upsampling_rate'],
                              ::opt['upsampling_rate'], :]

        # Reconstruct:
        (xsize, ysize, zsize, comp) = dt_lowres.shape
        recon_indx = [(i, j, k) for k in np.arange(opt['input_radius']+1,
                                                   zsize-opt['input_radius']+1,
                                                   2*opt['output_radius']+1)
                                for j in np.arange(opt['input_radius']+1,
                                                   ysize-opt['input_radius']+1,
                                                   2*opt['output_radius']+1)
                                for i in np.arange(opt['input_radius']+1,
                                                   xsize-opt['input_radius']+1,
                                                   2*opt['output_radius']+1)]
        for i, j, k in recon_indx:
            sys.stdout.flush()
            sys.stdout.write('\tSlice %i of %i.\r' % (k, zsize))

            ipatch_tmp = dt_lowres[(i - opt['input_radius'] - 1):(i + opt['input_radius']),
                                   (j - opt['input_radius'] - 1):(j + opt['input_radius']),
                                   (k - opt['input_radius'] - 1):(k + opt['input_radius']),
                                    2:comp]

            ipatch_mask = dt_lowres[(i - opt['output_radius'] - 1):(i + opt['output_radius']),
                                    (j - opt['output_radius'] - 1):(j + opt['output_radius']),
                                    (k - opt['output_radius'] - 1):(k + opt['output_radius']),
                                    0]

            # only process if any pixel in the output patch is in the brain.
            if np.max(ipatch_mask) >= 0:
                ipatch = ipatch_tmp[np.newaxis, ...]

                # Estimate high-res patch and its associeated uncertainty:
                fd = {x: ipatch,
                      keep_prob: 1.0-opt['dropout_rate'],
                      trade_off: 1.0,
                      phase_train: False}

                opatch, ovar_model, ovar_random = mc_inference_decompose(y_pred, y_std, fd, opt, sess)

                if opt["is_shuffle"]:  # only apply shuffling if necessary
                    opatch = forward_periodic_shuffle(opatch, opt['upsampling_rate'])
                    ovar_model = forward_periodic_shuffle(ovar_model, opt['upsampling_rate'])
                    ovar_random = forward_periodic_shuffle(ovar_random, opt['upsampling_rate'])

                dt_hires[opt['upsampling_rate'] * (i - opt['output_radius'] - 1):
                         opt['upsampling_rate'] * (i + opt['output_radius']),
                         opt['upsampling_rate'] * (j - opt['output_radius'] - 1):
                         opt['upsampling_rate'] * (j + opt['output_radius']),
                         opt['upsampling_rate'] * (k - opt['output_radius'] - 1):
                         opt['upsampling_rate'] * (k + opt['output_radius']),
                         2:] \
                = opatch

                dt_var_model[opt['upsampling_rate'] * (i - opt['output_radius'] - 1):
                             opt['upsampling_rate'] * (i + opt['output_radius']),
                             opt['upsampling_rate'] * (j - opt['output_radius'] - 1):
                             opt['upsampling_rate'] * (j + opt['output_radius']),
                             opt['upsampling_rate'] * (k - opt['output_radius'] - 1):
                             opt['upsampling_rate'] * (k + opt['output_radius']),
                             2:] \
                = ovar_model

                dt_var_random[opt['upsampling_rate'] * (i - opt['output_radius'] - 1):
                              opt['upsampling_rate'] * (i + opt['output_radius']),
                              opt['upsampling_rate'] * (j - opt['output_radius'] - 1):
                              opt['upsampling_rate'] * (j + opt['output_radius']),
                              opt['upsampling_rate'] * (k - opt['output_radius'] - 1):
                              opt['upsampling_rate'] * (k + opt['output_radius']),
                              2:] \
                = ovar_random

        # Trim unnecessary padding:
        dt_hires = dt_trim(dt_hires, padding)
        dt_var_model = dt_trim(dt_var_model, padding)
        dt_var_random = dt_trim(dt_var_random, padding)

        mask = dt_hires[:, :, :, 0] !=-1
        dt_hires[...,2:]=dt_hires[...,2:]*mask[..., np.newaxis]
        dt_var_model[..., 2:] = dt_var_model[..., 2:] * mask[..., np.newaxis]
        dt_var_random[..., 2:] = dt_var_random[..., 2:] * mask[..., np.newaxis]

        print("Size of dt_hires after trimming: %s", (dt_hires.shape,))
    return dt_hires, dt_var_model, dt_var_random


# --------------- reconstruct on non-HCP dataset  ----------------------
def sr_reconstruct_nonhcp(opt, dataset_type):
    # Define directory and file names:
    print('\nStart reconstruction! \n')
    recon_dir = opt['recon_dir']
    gt_dir = opt['gt_dir']
    subpath = opt['subpath']
    subject = opt['subject']
    no_channels = opt['no_channels']
    input_file_name = opt['input_file_name']
    gt_header = opt['gt_header']
    nn_dir = name_network(opt)
    output_file = os.path.join(recon_dir, subject, nn_dir, opt['output_file_name'])
    save_stats_dir = os.path.join(opt['stats_dir'], nn_dir)
    if not (os.path.exists(save_stats_dir)):
        os.makedirs(save_stats_dir)
    # ------------------------- Perform synthesis -----------------------------
    tf.reset_default_graph()
    print('\n ... reconstructing high-res dti \n')

    if os.path.exists(output_file):
        print("reconstruction already exists: " + output_file)
        print("move on. ")
    else:
        # Load the input low-res DT image:
        input_file = os.path.join(gt_dir, subject, subpath, input_file_name)
        print('\n ... loading the test low-res image ...')
        dt_lowres = sr_utility.read_dt_volume(input_file, no_channels=no_channels)

        if not (dataset_type == 'life' or dataset_type == 'hcp_abnormal' or dataset_type == 'hcp_abnormal_map' or dataset_type == 'hcp1' or dataset_type == 'hcp2' or dataset_type == 'monkey' or dataset_type == 'hcp1_map' or dataset_type == 'hcp2_map'):
            dt_lowres = sr_utility.resize_DTI(dt_lowres, opt['upsampling_rate'])
        else:
            print('HCP dataset: no need to resample.')

        # Reconstruct:
        start_time = timeit.default_timer()
        if opt['decompose']:
            dt_hr, dt_var_model, dt_var_random = super_resolve_decompose(dt_lowres, opt)
        else:
            dt_hr, dt_std = super_resolve(dt_lowres, opt)

        end_time = timeit.default_timer()
        print('\nIt took %f secs. \n' % (end_time - start_time))

        # Post-processing:
        if opt["postprocess"]:
            # Clipping:
            print('... post-processing the output image')
            dt_hr[..., -no_channels:] = clip_image(dt_hr[..., -no_channels:],
                                                   bkgv=opt["background_value"],
                                                   tail_perc=0.01, head_perc=99.99)

    # ------------------- Saving stuff ------------------------
    print("\n ... saving stuff")
    if opt["not_save"]:
        print("Selected not to save the outputs")
    elif os.path.exists(output_file):
        print("reconstruction already exists: " + output_file)
    else:
        print('saving MC-estimated high-res volume as %s' % output_file)
        if not (os.path.exists(os.path.join(recon_dir, subject, nn_dir))):
            os.makedirs(os.path.join(recon_dir, subject, nn_dir))

        # Save predicted high-res brain volume:
        np.save(output_file, dt_hr)
        print('\nsave each super-resolved channel separately as a nii file ...')
        __, recon_file = os.path.split(output_file)
        sr_utility.save_as_nifti(recon_file, os.path.join(recon_dir,subject,nn_dir),
                                 os.path.join(gt_dir,subject,subpath),
                                 no_channels=no_channels,
                                 save_as_ijk=opt['save_as_ijk'],
                                 gt_header=opt['gt_header'])

        # Save uncertainty for probabilistic models:
        if opt['hetero'] or opt['vardrop']:

            if opt['decompose']:
                uncertainty_random_file = os.path.join(recon_dir, subject, nn_dir, opt['output_var_random_file_name'])
                uncertainty_model_file = os.path.join(recon_dir, subject, nn_dir, opt['output_var_model_file_name'])
                print('... saving random uncertainty as %s' % uncertainty_random_file)
                print('... saving model uncertainty as %s' % uncertainty_model_file)
                np.save(uncertainty_random_file, dt_var_random)
                np.save(uncertainty_model_file, dt_var_model)
                __, var_random_file = os.path.split(uncertainty_random_file)
                __, var_model_file = os.path.split(uncertainty_model_file)
                sr_utility.save_as_nifti(var_random_file,
                                         os.path.join(recon_dir, subject,nn_dir),
                                         os.path.join(gt_dir, subject, subpath),
                                         save_as_ijk=opt['save_as_ijk'],
                                         no_channels=no_channels,
                                         gt_header=gt_header)

                sr_utility.save_as_nifti(var_model_file,
                                         os.path.join(recon_dir, subject,nn_dir),
                                         os.path.join(gt_dir, subject, subpath),
                                         save_as_ijk=opt['save_as_ijk'],
                                         no_channels=no_channels,
                                         gt_header=gt_header)
            else:
                uncertainty_file = os.path.join(recon_dir, subject, nn_dir, opt['output_std_file_name'])
                print('... saving its uncertainty as %s' % uncertainty_file)
                np.save(uncertainty_file, dt_std)
                __, std_file = os.path.split(uncertainty_file)
                print('\nsave the uncertainty separately for respective channels as a nii file ...')
                sr_utility.save_as_nifti(std_file,
                                         os.path.join(recon_dir, subject, nn_dir),
                                         os.path.join(gt_dir, subject, subpath),
                                         no_channels=no_channels,
                                         save_as_ijk=opt['save_as_ijk'],
                                         gt_header=opt['gt_header'])

    # ----------------- Compute stats ---------------------------
    print('\n ... compute the evaluation statistics ...')

    if opt['gt_available']:
        # load the ground truth image and mask:
        dt_gt = sr_utility.read_dt_volume(nameroot=os.path.join(gt_dir, subject, subpath, gt_header), no_channels=no_channels)
        if os.path.exists(output_file): dt_hr = np.load(output_file)
        mask_file = "mask_us={:d}_rec={:d}.nii".format(opt["upsampling_rate"], 5)
        mask_dir_local = os.path.join(opt["mask_dir"], subject, opt["mask_subpath"], "masks")

        if os.path.exists(os.path.join(mask_dir_local, mask_file)):
            img = nib.load(os.path.join(mask_dir_local, mask_file))
            mask_interior = img.get_data() == 0
            mask = dt_hr[:, :, :, 0] == 0
            mask_edge = mask * (1 - mask_interior)

            m, m2, p, s = compare_images_and_get_stats(dt_gt[..., 2:], dt_hr[..., 2:], mask, "whole")
            m_int, m2_int, p_int, s_int = compare_images_and_get_stats(dt_gt[..., 2:], dt_hr[..., 2:], mask_interior, "interior")
            m_ed, m2_ed, p_ed, s_ed = compare_images_and_get_stats(dt_gt[..., 2:], dt_hr[..., 2:], mask_edge, "edge")

            csv_file = os.path.join(save_stats_dir, 'stats.csv')
            headers = ['subject',
                       'RMSE(interior)', 'RMSE(edge)', 'RMSE(whole)',
                       'Median(interior)', 'Median(edge)', 'Median(whole)',
                       'PSNR(interior)', 'PSNR(edge)', 'PSNR(whole)',
                       'MSSIM(interior)', 'MSSIM(edge)', 'MSSIM(whole)']
            stats = [m_int, m_ed, m, m2_int, m2_ed, m2, p_int, p_ed, p, s_int,
                     s_ed, s]
        else:
            print("Mask for the interior region NOT FOUND")
            mask = dt_hr[:, :, :, 0] == 0
            m, m2, p, s = compare_images_and_get_stats(dt_gt[..., 2:], dt_hr[..., 2:], mask, "whole")
            csv_file = os.path.join(save_stats_dir, 'stats_brain.csv')
            headers = ['subject', 'RMSE(whole)', 'Median(whole)', 'PSNR(whole)', 'MSSIM(whole)']
            stats = [m, m2, p, s]

        # Save the stats to a CSV file:
        save_stats(csv_file, opt['subject'], headers, stats)

        if opt["not_save"]:
            print('Difference maps are not saved.')
        else:
            # Compute difference maps and save:
            compute_differencemaps(dt_gt[..., 2:], dt_hr[..., 2:],
                                   mask, output_file, no_channels,
                                   save_as_ijk=opt['save_as_ijk'],
                                   gt_dir=os.path.join(gt_dir, subject, subpath),
                                   gt_header=opt['gt_header'])

    else:
        print(" Ground truth data not available. Finished.")















