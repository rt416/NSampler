""" Reconstruction file """

import os
import sys
import timeit

import cPickle as pkl
import h5py
import numpy as np
import tensorflow as tf

import sr_preprocess as pp
import sr_utility
import cgan.models as models
from cgan.train_v2 import define_checkpoint, name_network, name_patchlib
from sr_datageneration import forward_periodic_shuffle


def get_trasnform(opt):
    if opt['transform_opt']=='0':
        transform = dict()
        transform['input_mean'] = .0
        transform['input_std'] = 1.0
        transform['output_mean'] = .0
        transform['output_std'] = 1.0
    elif opt['transform_opt']=='standard':
        transform = dict()
        transform['input_mean']=np.array([0.0007143541,
                                          9.1065253e-09,
                                          1.0169547e-06,
                                          0.00074001489,
                                          -1.2684665e-05,
                                          0.00071994716])
        transform['input_std']=np.array([0.00047637321,
                                         8.8792462e-05,
                                         9.2553419e-05,
                                         0.00048288782,
                                         9.5012001e-05,
                                         0.00047756024])
        transform['output_mean']=np.array([0.00089535239,
                                           0.0008760544,
                                           0.00087555352])
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
        opt (dict):
    Returns:
        the estimated high-res volume
    """

    # --------------------------- Define the model--------------------------:
    # Input/Output details:
    upsampling_rate = opt['upsampling_rate']
    input_radius = opt['input_radius']
    output_radius = opt['output_radius']

    # get the dir where the network is saved
    network_dir = define_checkpoint(opt)

    print('... defining the network model %s .' % opt['method'])
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None,
                                        2 * input_radius + 1,
                                        2 * input_radius + 1,
                                        2 * input_radius + 1,
                                        opt['no_channels']],
                           name='lo_res')
        y = tf.placeholder(tf.float32, [None,
                                        2 * output_radius + 1,
                                        2 * output_radius + 1,
                                        2 * output_radius + 1,
                                        opt['no_channels'] * (upsampling_rate ** 3)],
                           name='hi_res')

    with tf.name_scope('learning_rate'):
        lr = tf.placeholder(tf.float32, [], name='learning_rate')

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)  # keep probability for dropout

    with tf.name_scope('tradeoff'):
        trade_off = tf.placeholder(tf.float32)  # keep probability for dropout

    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Load normalisation parameters and define prediction:
    transfile = opt['data_dir'] + name_patchlib(opt) + '/transforms.pkl'
    transform = pkl.load(open(transfile, 'rb'))
    y_pred, y_pred_std = models.scaled_prediction(opt['method'], x, y, keep_prob, transform, opt, trade_off)

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

            ipatch_tmp = dt_lowres[(i - input_radius - 1):(i + input_radius),
                                   (j - input_radius - 1):(j + input_radius),
                                   (k - input_radius - 1):(k + input_radius),
                                    2:comp]

            ipatch = ipatch_tmp[np.newaxis, ...]

            # Predict high-res patch:
            fd = {x: ipatch, keep_prob: 1.0, trade_off: 0.0}
            opatch_shuffled = y_pred.eval(feed_dict=fd)

            opatch = forward_periodic_shuffle(opatch_shuffled, upsampling_rate)

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
        mask = dt_hires[:, :, :, 0] !=-1
        dt_hires[...,2:]=dt_hires[...,2:]*mask[..., np.newaxis]

        print("Size of dt_hires after trimming: %s", (dt_hires.shape,))
    return dt_hires


# Main reconstruction code:
def sr_reconstruct(opt):
    print('\nStart reconstruction! \n')
    # load parameters:
    recon_dir = opt['recon_dir']
    gt_dir = opt['gt_dir']
    subpath = opt['subpath']
    subject = opt['subject']
    no_channels = opt['no_channels']

    input_file_name, _ = opt['input_file_name'].split('{')
    gt_header, _ = opt['gt_header'].split('{')
    if not('output_file_name' in opt):
        opt['output_file_name'] = 'dt_recon_b1000.npy'

    # Load the input low-res DT image:
    input_file = gt_dir + subject + subpath + input_file_name
    print('... loading the test low-res image ...')
    dt_lowres = sr_utility.read_dt_volume(input_file, no_channels=no_channels)

    # clear the graph (is it necessary?)
    tf.reset_default_graph()

    # Reconstruct:
    start_time = timeit.default_timer()
    nn_dir = name_network(opt)
    print('\nReconstruct high-res dti with the network: \n%s.' % nn_dir)
    dt_hr = super_resolve(dt_lowres, opt)

    # Save:
    if no_channels > 6:
        output_file = os.path.join(recon_dir, subject, nn_dir, opt['output_file_name'])
    else:
        output_file = os.path.join(recon_dir, subject, nn_dir, 'dt_recon_b1000.npy')

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
                             recon_dir+subject+'/'+nn_dir,
                             gt_dir+subject+subpath,
                             no_channels=no_channels,
                             gt_header=gt_header)

    # Compute the reconstruction error:
    mask_file = 'mask_us=' + str(opt['upsampling_rate']) + \
                '_rec=' + str(5) + '.nii'
    # mask_file = 'mask_us=' + str(opt['upsampling_rate']) + \
    #             '_rec=' + str(2*opt['receptive_field_radius']+1) +'.nii'
    mask_dir = opt['mask_dir']
    rmse, rmse_whole, rmse_volume \
        = sr_utility.compute_rmse(recon_file=recon_file,
                                  recon_dir=recon_dir+subject+'/'+nn_dir,
                                  gt_dir=gt_dir+subject+subpath,
                                  mask_choose=True,
                                  mask_dir=mask_dir+subject+'/masks',
                                  mask_file=mask_file,
                                  no_channels=no_channels,
                                  gt_header=gt_header)

    print('\nRMSE (no edge) is %f.' % rmse)
    print('\nRMSE (whole) is %f.' % rmse_whole)

    # Save the RMSE on the chosen test subject:
    print('Save it to settings.skl')
    network_dir = define_checkpoint(opt)
    model_details = pkl.load(open(os.path.join(network_dir, 'settings.pkl'), 'rb'))
    if not('subject_rmse' in model_details):
        model_details['subject_rmse'] = {opt['subject']:rmse}
    else:
        model_details['subject_rmse'].update({opt['subject']:rmse})

    with open(os.path.join(network_dir, 'settings.pkl'), 'wb') as fp:
        pkl.dump(model_details, fp, protocol=pkl.HIGHEST_PROTOCOL)

    return rmse, rmse_whole

