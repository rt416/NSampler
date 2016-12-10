""" Reconstruction file """

import os
import sys
import timeit

import cPickle as pkl
import h5py
import numpy as np
import tensorflow as tf

import ryu_preprocess as pp
import sr_utility
import models
from ryu_train import define_checkpoint, name_network
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
                          pd[2][0]:-pd[2][1],
                          :]
    return dt_volume


# Reconstruct using the specified NN:
def super_resolve(dt_lowres, opt):

    """Perform a patch-based super-resolution on a given low-res image.
    Args:
        dt_lowres (numpy array): a low-res diffusion tensor image volume
        n (int): the width of an input patch is 2*n + 1
        m (int): the width of an output patch is m
        us (int): the upsampling factord
    Returns:
        the estimated high-res volume
    """
    # ------------------ Load in the parameters ------------------:

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

    print('... defining the network model %s .' % method)
    x = tf.placeholder(tf.float32, [2*input_radius+1,
                                    2*input_radius+1,
                                    2*input_radius+1,
                                    no_channels],
                                    name='lo_res')
    y = tf.placeholder(tf.float32, [2*output_radius+1,
                                    2*output_radius+1,
                                    2*output_radius+1,
                                    no_channels*(upsampling_rate**3)],
                                    name='hi_res')
    lr = tf.placeholder(tf.float32, [], name='learning_rate')
    keep_prob = tf.placeholder(tf.float32)  # keep probability for dropout
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Load normalisation parameters and define prediction:
    transform = pkl.load(open(os.path.join(network_dir, 'transforms.pkl'), 'rb'))
    y_pred = models.scaled_prediction(method, x, transform, opt)

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
        dt_lowres, padding = dt_pad(dt_volume=dt_lowres, opt=opt)

        print("Size of dt_lowres after padding: %s", (dt_lowres.shape,))

        # Prepare high-res skeleton:
        dt_hires = np.zeros(dt_lowres.shape)
        dt_hires[:, :, :, 0] = dt_lowres[:, :, :, 0]  # same brain mask as input
        print("Size of dt_hires after padding: %s", (dt_hires.shape,))

        # Downsample:
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

            ipatch = dt_lowres[(i - input_radius - 1):(i + input_radius),
                               (j - input_radius - 1):(j + input_radius),
                               (k - input_radius - 1):(k + input_radius),
                               2:comp]

            # Predict high-res patch:
            fd = {x: ipatch, keep_prob: (1.0 - dropout_rate)}
            opatch_shuffled = y_pred.eval(feed_dict=fd)
            # print("shape of opatch before shuffling is :%s" % (opatch_shuffled.shape, ))
            opatch = forward_periodic_shuffle(opatch_shuffled, upsampling_rate)
            # print("shape of opatch is :%s" % (opatch.shape, ))
            # print("output_radius, %i" %output_radius)

            dt_hires[upsampling_rate * (i - output_radius - 1):
                     upsampling_rate * (i + output_radius),
                     upsampling_rate * (j - output_radius - 1):
                     upsampling_rate * (j + output_radius),
                     upsampling_rate * (k - output_radius - 1):
                     upsampling_rate * (k + output_radius),
                     2:] \
            = opatch

        # Trim unnecessary padding:
        dt_hires = dt_trim(dt_hires, padding)
        print("Size of dt_hires after trimming: %s", (dt_hires.shape,))
    return dt_hires


# Main reconstruction code:
def sr_reconstruct(opt):

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
    print('\nReconstruct high-res dti with the network: \n%s.' % nn_dir)
    dt_hr = super_resolve(dt_lowres, opt)

    # Save:
    output_file = os.path.join(recon_dir, subject, nn_dir, 'dt_recon_b1000.npy')
    print('... saving as %s' % output_file)
    if not(os.path.exists(os.path.join(recon_dir, subject, nn_dir))):
        os.mkdir(os.path.join(recon_dir, subject, nn_dir))
    np.save(output_file, dt_hr)
    end_time = timeit.default_timer()
    print('\nIt took %f secs. \n' % (end_time - start_time))

    # Compute the reconstruction error:
    __, recon_file = os.path.split(output_file)
    rmse, rmse_volume = sr_utility.compute_rmse(recon_file,
                                                os.path.join(recon_dir, subject, nn_dir),
                                                os.path.join(gt_dir, subject, subpath))

    print('\nReconsturction error (RMSE) is %f.' % rmse)

    # Save each estimated dti separately as a nifti file for visualisation:
    print('\nSave each estimated dti separately as a nii file ...')
    sr_utility.save_as_nifti(recon_file,
                             os.path.join(recon_dir, subject),
                             os.path.join(gt_dir, subject, subpath))
