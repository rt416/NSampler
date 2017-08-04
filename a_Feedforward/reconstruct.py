""" Reconstruction file """
import timeit
import cPickle as pkl
import numpy as np
import tensorflow as tf
import nibabel as nib
import os, sys
from train import get_output_radius
import common.sr_utility as sr_utility
from common.sr_utility import forward_periodic_shuffle
from common.utils import name_network, name_patchlib, set_network_config, define_checkpoint, mc_inference, dt_trim, dt_pad, clip_image, save_stats
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
    if not (os.path.exists(opt["recon_dir"])):
        os.makedirs(opt["stats_dir"])

    # ------------------------- Perform synthesis -----------------------------
    print('\n ... reconstructing high-res dti with network: \n%s.' % nn_dir)
    if os.path.exists(output_file):
        print("Reconstruction already exists: " + output_file)
        print("Move on. ")
    else:
        # Load the input low-res DT image:
        print('... loading the test low-res image ...')
        input_file = os.path.join(gt_dir,subject,subpath,input_file_name)
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
        print('\nReconstruct high-res dti with the network: \n%s.' % nn_dir)
        dt_hr = super_resolve(dt_lowres, opt)
        end_time = timeit.default_timer()
        print('\nIt took %f secs. \n' % (end_time - start_time))

        # Post-processing:
        if opt["postprocess"]:
            # Clipping:
            print('... post-processing the output image')
            dt_hr[..., -no_channels:] = clip_image(dt_hr[..., -no_channels:],
                                                   bkgv=opt["background_value"],
                                                   tail_perc=0.01, head_perc=99.99)

    # ---------------------------- Save stuff ---------------------------
    if opt["not_save"]:
        print("Selected not to save the outputs")
    elif os.path.exists(output_file):
        print("Reconstruction already exists: " + output_file)
    else:
        print('... saving as %s' % output_file)
        if not (os.path.exists(os.path.join(recon_dir, subject))):
            os.mkdir(os.path.join(recon_dir, subject))
        if not(os.path.exists(os.path.join(recon_dir, subject, nn_dir))):
            os.mkdir(os.path.join(recon_dir, subject, nn_dir))
        np.save(output_file, dt_hr)
        end_time = timeit.default_timer()
        print('\nIt took %f secs. \n' % (end_time - start_time))

        # Save each estimated dti separately as a nifti file for visualisation:
        __, recon_file = os.path.split(output_file)
        print('\nSave each estimated dti separately as a nii file ...')
        sr_utility.save_as_nifti(recon_file,
                                 os.path.join(recon_dir,subject,nn_dir),
                                 os.path.join(gt_dir,subject,subpath),
                                 no_channels=no_channels,
                                 gt_header=gt_header)

    # ------------- Compute stats (e.g. errors, difference maps, etc) ---------
    print('\nCompute the evaluation statistics ...')

    # load the ground truth image and mask:
    dt_gt = sr_utility.read_dt_volume(
        nameroot=os.path.join(gt_dir, subject, subpath, gt_header),
        no_channels=no_channels)
    if os.path.exists(output_file): dt_hr = np.load(output_file)
    mask_file = "mask_us={:d}_rec={:d}.nii".format(opt["upsampling_rate"], 5)
    mask_dir_local = os.path.join(opt["mask_dir"], subject, opt["mask_subpath"],
                                  "masks")

    if os.path.exists(os.path.join(mask_dir_local, mask_file)):
        img = nib.load(os.path.join(mask_dir_local, mask_file))
        mask_interior = img.get_data() == 0
        mask = dt_hr[:, :, :, 0] == 0
        mask_edge = mask * (1 - mask_interior)

        m, m2, p, s = compare_images_and_get_stats(dt_gt[..., 2:],
                                                   dt_hr[..., 2:], mask,
                                                   "whole")
        m_int, m2_int, p_int, s_int = compare_images_and_get_stats(
            dt_gt[..., 2:], dt_hr[..., 2:], mask_interior, "interior")
        m_ed, m2_ed, p_ed, s_ed = compare_images_and_get_stats(dt_gt[..., 2:],
                                                               dt_hr[..., 2:],
                                                               mask_edge,
                                                               "edge")

        csv_file = os.path.join(save_stats_dir, 'stats.csv')
        headers = ['subject',
                   'RMSE(interior)', 'RMSE(edge)', 'RMSE(whole)',
                   'Median(interior)', 'Median(edge)', 'Median(whole)',
                   'PSNR(interior)', 'PSNR(edge)', 'PSNR(whole)',
                   'MSSIM(interior)', 'MSSIM(edge)', 'MSSIM(whole)']
        stats = [m_int, m_ed, m, m2_int, m2_ed, m2, p_int, p_ed, p, s_int, s_ed,s]
    else:
        print("Mask for the interior region NOT FOUND")
        mask = dt_hr[:, :, :, 0] == 0
        m, m2, p, s = compare_images_and_get_stats(dt_gt[..., 2:],
                                                   dt_hr[..., 2:], mask,
                                                   "whole")
        csv_file = os.path.join(save_stats_dir, 'stats_brain.csv')
        headers = ['subject', 'RMSE(whole)', 'Median(whole)', 'PSNR(whole)', 'MSSIM(whole)']
        stats = [m, m2, p, s]

    # Save the stats to a CSV file:
    save_stats(csv_file, opt['subject'], headers, stats)

    # Compute difference maps and save:
    compute_differencemaps(dt_gt[..., 2:], dt_hr[..., 2:],
                           mask, output_file,no_channels,
                           save_as_ijk=opt['save_as_ijk'],
                           gt_dir=os.path.join(gt_dir, subject, subpath),
                           gt_header=gt_header)


# Reconstruct with shuffling:
def super_resolve(dt_lowres, opt):

    """Perform a patch-based super-resolution on a given low-res image.
    Args:
        dt_lowres (numpy array): a low-res diffusion tensor image volume
        opt (dict):
    Returns:
        the estimated high-res volume
    """

    # --------------------------- Define the model--------------------------:
    # Get the dir where the network is saved
    network_dir = define_checkpoint(opt)

    # Placeholders
    print('... defining the network model %s .' % opt['method'])
    side = 2*opt["input_radius"] + 1
    x = tf.placeholder(tf.float32,
                       shape=[1,side,side,side,opt['no_channels']],
                       name='input_x')
    phase_train = tf.placeholder(tf.bool, name='phase_train')
    keep_prob = tf.placeholder(tf.float32, name='dropout_rate')
    trade_off = tf.placeholder(tf.float32, name='trade_off')

    net = set_network_config(opt)
    transfile = os.path.join(opt['data_dir'], name_patchlib(opt), 'transforms.pkl')
    transform = pkl.load(open(transfile, 'rb'))
    y_pred = net.scaled_prediction(x, phase_train, transform)

    # Compute the output radius:
    opt['output_radius'] = get_output_radius(y_pred, opt['upsampling_rate'], opt['is_shuffle'])

    # Specify the network parameters to be restored:
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
        dt_hires[:, :, :, 1] = dt_lowres[:, :, :, 1]  # assign the same logS0

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

            ipatch = ipatch_tmp[np.newaxis, ...]

            # Predict high-res patch:
            fd = {x: ipatch,
                  keep_prob: 1.0-opt['dropout_rate'],
                  trade_off: 0.0,
                  phase_train: False}
            opatch = y_pred.eval(feed_dict=fd)

            if opt["is_shuffle"]:  # only apply shuffling if necessary
                opatch = forward_periodic_shuffle(opatch, opt['upsampling_rate'])

            dt_hires[opt['upsampling_rate'] * (i - opt['output_radius'] - 1):
                     opt['upsampling_rate'] * (i + opt['output_radius']),
                     opt['upsampling_rate'] * (j - opt['output_radius'] - 1):
                     opt['upsampling_rate'] * (j + opt['output_radius']),
                     opt['upsampling_rate'] * (k - opt['output_radius'] - 1):
                     opt['upsampling_rate'] * (k + opt['output_radius']),
                     2:] \
            = opatch

        # Trim unnecessary padding:
        dt_hires = dt_trim(dt_hires, padding)
        mask = dt_hires[:, :, :, 0] !=-1
        dt_hires[...,2:]=dt_hires[...,2:]*mask[..., np.newaxis]

        print("Size of dt_hires after trimming: %s", (dt_hires.shape,))
    return dt_hires