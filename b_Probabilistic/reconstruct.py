""" Reconstruction file """
import timeit
import cPickle as pkl
import numpy as np
import tensorflow as tf

from train import get_output_radius
import common.sr_utility as sr_utility
from common.sr_utility import forward_periodic_shuffle
from common.utils import *

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
    if not('output_file_name' in opt):
        opt['output_file_name'] = 'dt_recon_b1000.npy'

    # Load the input low-res DT image:
    input_file = os.path.join(gt_dir,subject,subpath,input_file_name)
    print('... loading the test low-res image ...')
    dt_lowres = sr_utility.read_dt_volume(input_file, no_channels=no_channels)

    # Seems to aggravate performance, so currently ignored.
    # Clip the input DTI:
    # if opt["is_clip"]:
    #     print('... clipping the input image')
    #     dt_lowres[...,-no_channels:]=clip_image(dt_lowres[...,-no_channels:],
    #                                             bkgv=opt["background_value"])

    # clear the graph (is it necessary?)
    tf.reset_default_graph()

    # Reconstruct:
    start_time = timeit.default_timer()
    nn_dir = name_network(opt)
    print('\nReconstruct high-res dti with the network: \n%s.' % nn_dir)
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

    # Save stuff:
    if opt["not_save"]:
        rmse, rmse_whole = 10**10, 10**10
    else:
        output_file = os.path.join(recon_dir, subject, nn_dir, opt['output_file_name'])
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
                                 no_channels=no_channels,
                                 gt_header=gt_header)

        # Save uncertainty for probabilistic models:
        if opt['hetero'] or opt['vardrop']:
            uncertainty_file = os.path.join(recon_dir, subject, nn_dir, opt['output_std_file_name'])
            print('... saving its uncertainty as %s' % uncertainty_file)
            np.save(uncertainty_file, dt_std)
            __, std_file = os.path.split(uncertainty_file)
            print(
            '\nSave the uncertainty separately for respective channels as a nii file ...')
            sr_utility.save_as_nifti(std_file,
                                     os.path.join(recon_dir, subject, nn_dir),
                                     os.path.join(gt_dir, subject, subpath),
                                     no_channels=no_channels,
                                     gt_header=gt_header)

    # Compute the reconstruction error:
    print('\nCompute the evaluation statistics ...')
    mask_file = "mask_us={:d}_rec={:d}.nii".format(opt["upsampling_rate"],5)
    mask_dir_local = os.path.join(opt["mask_dir"], subject, opt["mask_subpath"],"masks")
    rmse, rmse_whole, rmse_volume \
        = sr_utility.compute_rmse(recon_file=recon_file,
                                  recon_dir=os.path.join(recon_dir,subject,nn_dir),
                                  gt_dir=os.path.join(gt_dir,subject,subpath),
                                  mask_choose=True,
                                  mask_dir=mask_dir_local,
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


# Monte-Carlo inference:
def mc_inference(fn, fn_std, fd, opt, sess):
    """ Compute the mean and std of samples drawn from stochastic function"""
    no_samples = opt['mc_no_samples']
    if opt['hetero']:
        if opt['cov_on']:
            sum_out = 0.0
            sum_out2 = 0.0
            sum_var = 0.0
            for i in xrange(no_samples):
                current, current_std = sess.run([fn, fn_std], feed_dict=fd)
                sum_out += current
                sum_out2 += current ** 2
                sum_var += current_std ** 2
            mean = sum_out / (1. * no_samples)
            std = np.sqrt((np.abs(sum_out2 - 2 * mean * sum_out + no_samples * mean ** 2) + sum_var) / no_samples)
        else:
            sum_out = 0.0
            sum_out2 = 0.0
            for i in xrange(no_samples):
                current = 1. * fn.eval(feed_dict=fd)
                sum_out += current
                sum_out2 += current ** 2
            mean = sum_out / (1. * no_samples)
            std = np.sqrt(np.abs(sum_out2 - 2 * mean * sum_out + no_samples * mean ** 2) / no_samples)
            std += 1. * fn_std.eval(feed_dict=fd)
    else:
        if opt['vardrop']:
            sum_out = 0.0
            sum_out2 = 0.0
            for i in xrange(no_samples):
                current = 1. * fn.eval(feed_dict=fd)
                sum_out += current
                sum_out2 += current ** 2

            mean = sum_out / (1.*no_samples)
            std = np.sqrt(np.abs(sum_out2 - 2*mean*sum_out + no_samples*mean**2)/no_samples)
        else:
            # raise Exception('The specified method does not support MC inference.')
            mean = fn.eval(feed_dict=fd)
            std = 0.0*mean  # zero in every entry
    return mean, std

# Pad the volumes:
def dt_pad(dt_volume, upsampling_rate, input_radius):
    """ Pad a volume with zeros before reconstruction """
    # --------------------- Pad ---------------:
    # Pad with zeros so all brain-voxel-centred pathces are extractable and
    # each dimension is divisible by upsampling rate.
    dim_x_highres, dim_y_highres, dim_z_highres, dim_channels = dt_volume.shape
    pad_min = (input_radius + 1) * upsampling_rate

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


# Clip images:
def clip_image(img, bkgv=0.0, tail_perc=0.1, head_perc=99.9):
    """ Truncate 3d volume by the specified percentile
    Assumptions:
        img is 4d numpy array with last dim being channels e.g. DTI components
        back ground value is consistently given by bkgv.

    Args:
    """
    assert img.ndim == 4
    brain_mask = img[..., 0] != bkgv # get the foreground voxels
    for ch_idx in range(img.shape[-1]):
        v_ch=img[...,ch_idx][brain_mask]
        inp_perc_tail = np.percentile(v_ch, tail_perc)
        inp_perc_head = np.percentile(v_ch, head_perc)
        img[...,ch_idx][brain_mask]=np.clip(v_ch, inp_perc_tail, inp_perc_head)
    return img


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

    # input_file_name, = opt['input_file_name'].split('{')
    # gt_header, _ = opt['gt_header'].split('{')
    if not('output_file_name' in opt):
        opt['output_file_name'] = 'dt_recon_b1000.npy'

    # Load the input low-res DT image:
    input_file = os.path.join(gt_dir,subject,subpath,input_file_name)
    print('... loading the test low-res image ...')
    dt_lowres = sr_utility.read_dt_volume(input_file, no_channels=no_channels)

    if not (dataset_type == 'life' or dataset_type == 'hcp_abnormal' or dataset_type == 'hcp1' or dataset_type == 'hcp2' or dataset_type == 'monkey'):
        dt_lowres = sr_utility.resize_DTI(dt_lowres, opt['upsampling_rate'])
    else:
        print('HCP dataset: no need to resample.')

    # Reconstruct:
    tf.reset_default_graph()
    start_time = timeit.default_timer()
    print('\nReconstructing high-res dti \n')
    dt_hr, dt_std = super_resolve(dt_lowres, opt)
    nn_dir = name_network(opt)
    end_time = timeit.default_timer()
    print('\nIt took %f secs. \n' % (end_time - start_time))

    # Post-processing:
    if opt["postprocess"]:
        # Clipping:
        print('... post-processing the output image')
        dt_hr[..., -no_channels:] = clip_image(dt_hr[..., -no_channels:],
                                               bkgv=opt["background_value"],
                                               tail_perc=0.01, head_perc=99.99)

    print('... saving MC-estimated high-res volume as %s' % output_file)
    if not (os.path.exists(os.path.join(recon_dir, subject, nn_dir))):
        os.makedirs(os.path.join(recon_dir, subject, nn_dir))

    # Save predicted high-res brain volume:
    output_file = os.path.join(recon_dir, subject, nn_dir, opt['output_file_name'])
    np.save(output_file, dt_hr)
    print('\nSave each super-resolved channel separately as a nii file ...')
    __, recon_file = os.path.split(output_file)
    sr_utility.save_as_nifti(recon_file,
                             os.path.join(recon_dir,subject,nn_dir),
                             os.path.join(gt_dir,subject,subpath),
                             no_channels=no_channels,
                             gt_header=gt_header)

    # Save uncertainty for probabilistic models:
    if opt['hetero'] or opt['vardrop']:
        uncertainty_file = os.path.join(recon_dir, subject, nn_dir, opt['output_std_file_name'])
        print('... saving its uncertainty as %s' % uncertainty_file)
        np.save(uncertainty_file, dt_std)
        __, std_file = os.path.split(uncertainty_file)
        print('\nSave the uncertainty separately for respective channels as a nii file ...')
        sr_utility.save_as_nifti(std_file,
                                 os.path.join(recon_dir, subject, nn_dir),
                                 os.path.join(gt_dir, subject, subpath),
                                 no_channels=no_channels,
                                 gt_header=gt_header)

    # todo: add a proper evaluation function here and save it to a common cfv file.
    # # Compute the reconstruction error:
    # print('\nCompute the evaluation statistics ...')
    # mask_file = "mask_us={:d}_rec={:d}.nii".format(opt["upsampling_rate"],5)
    # mask_dir_local = os.path.join(opt["mask_dir"], subject, opt["mask_subpath"],"masks")
    # rmse, rmse_whole, rmse_volume \
    #     = sr_utility.compute_rmse(recon_file=recon_file,
    #                               recon_dir=os.path.join(recon_dir,subject,nn_dir),
    #                               gt_dir=os.path.join(gt_dir,subject,subpath),
    #                               mask_choose=True,
    #                               mask_dir=mask_dir_local,
    #                               mask_file=mask_file,
    #                               no_channels=no_channels,
    #                               gt_header=gt_header)
    #
    # print('\nRMSE (no edge) is %f.' % rmse)
    # print('\nRMSE (whole) is %f.' % rmse_whole)
