""" Reconstruction file """

import os
import sys
import timeit

import cPickle
import h5py
import numpy as np
import tensorflow as tf

import preprocess as pp
import sr_utility
import models

# Reconstruct using the specified NN:
def super_resolve(dt_lowres,
                  method='mlp_h=1',
                  n_h1=500, n_h2=200, n_h3=100,
                  n=2, m=2, us=2, dropout_rate=0.0,
                  network_dir='/Users/ryutarotanno/DeepLearning/nsampler/models/linear'):

    """Perform a patch-based super-resolution on a given low-res image.
    Args:
        dt_lowres (numpy array): a low-res diffusion tensor image volume
        n (int): the width of an input patch is 2*n + 1
        m (int): the width of an output patch is m
        us (int): the upsampling factord
    Returns:
        the estimated high-res volume
    """

    # Specify the network:
    print('... defining the network model %s .' % method)
    n_in, n_out = 6 * (2 * n + 1) ** 3, 6 * m ** 3  # dimensions of input and output
    x_scaled = tf.placeholder(tf.float32, shape=[None, n_in])
    y_scaled = tf.placeholder(tf.float32, shape=[None, n_out])
    keep_prob = tf.placeholder(tf.float32)  # keep probability for dropout
    y_pred_scaled, L2_sqr, L1 = models.inference(method, x_scaled, keep_prob, n_in, n_out,
                                                 n_h1=n_h1, n_h2=n_h2, n_h3=n_h3)

    # load the transforms used for normalisation of the training data:
    transform_file = os.path.join(network_dir, 'transforms.pkl')
    transform = cPickle.load(open(transform_file, 'rb'))
    train_set_x_mean = transform['input_mean'].reshape((1, n_in))  # row vector representing the mean
    train_set_x_std = transform['input_std'].reshape((1, n_in))
    train_set_y_mean = transform['output_mean'].reshape((1, n_out))
    train_set_y_std = transform['output_std'].reshape((1, n_out))
    del transform

    # load the weights with the best performance:
    settings_file = os.path.join(network_dir, 'settings.pkl')
    details = cPickle.load(open(settings_file, 'rb'))
    best_step = details['best step']

    # Restore all the variables and perform reconstruction:
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, os.path.join(network_dir, "model-" + str(best_step)))
        print("Model restored.")

        # reconstruct
        dt_lowres = dt_lowres[0::us, 0::us, 0::us, :]  # take every us th entry to reduce it to the original resolution.
        (xsize, ysize, zsize, comp) = dt_lowres.shape
        dt_hires = np.zeros((xsize * us, ysize * us, zsize * us, comp)) # the base array for the output high-res volume.
        dt_hires[:, :, :, 0] = -1  # initialise all the voxels as 'background'.

        for k in np.arange(n + 1, zsize - n + 1):
            print('Slice %i of %i.' % (k, zsize))
            for j in np.arange(n + 1, ysize - n + 1):
                for i in np.arange(n + 1, xsize - n + 1):
                    ipatch = dt_lowres[(i - n - 1):(i + n), (j - n - 1):(j + n), (k - n - 1):(k + n), 2:comp] # input patch

                    # Process only if the whole patch is foreground
                    if np.min(dt_lowres[(i - n - 1):(i + n), (j - n - 1):(j + n), (k - n - 1):(k + n), 0]) >= 0:

                        # Vectorise input patch (following 'Fortran' reshape ordering) and normalise:
                        ipatch_row = ipatch.reshape((1, ipatch.size), order='F')
                        ipatch_row_scaled = (ipatch_row - train_set_x_mean)/train_set_x_std

                        # Predict the corresponding high-res output patch in the normalised space:
                        opatch_row_scaled = y_pred_scaled.eval(feed_dict={x_scaled: ipatch_row_scaled,
                                                                          keep_prob: (1.0 - dropout_rate)})

                        # Send back into the original space and reshape into a cubic patch:
                        opatch_row = train_set_y_std*opatch_row_scaled + train_set_y_mean
                        opatch = opatch_row.reshape((m, m, m, comp - 2), order='F')

                        # Select the correct location of the output patch in the brain and store:
                        x_temp_1, x_temp_2 = (us * (i - 1) + 1 - (m - us) / 2) - 1, (us * i + (m - us) / 2)
                        y_temp_1, y_temp_2 = (us * (j - 1) + 1 - (m - us) / 2) - 1, (us * j + (m - us) / 2)
                        z_temp_1, z_temp_2 = (us * (k - 1) + 1 - (m - us) / 2) - 1, (us * k + (m - us) / 2)

                        dt_hires[x_temp_1:x_temp_2, y_temp_1:y_temp_2, z_temp_1:z_temp_2, 2:comp] \
                            = dt_hires[x_temp_1:x_temp_2, y_temp_1:y_temp_2, z_temp_1:z_temp_2, 2:comp] + opatch

                        # Label only reconstructed voxels as foreground.
                        dt_hires[x_temp_1:x_temp_2, y_temp_1:y_temp_2, z_temp_1:z_temp_2, 0] = 0
    return dt_hires

# Main reconstruction code:
def sr_reconstruct(method='linear', n_h1=500, n_h2=200,  n_h3=100, us=2, n=2, m=2,
                   optimisation_method='adam', dropout_rate=0.0, cohort='Diverse', no_subjects=8, sample_rate=32,
                   model_dir='/Users/ryutarotanno/DeepLearning/nsampler/models',
                   recon_dir='/Users/ryutarotanno/DeepLearning/nsampler/recon',
                   gt_dir='/Users/ryutarotanno/DeepLearning/Test_1/data'):

    start_time = timeit.default_timer()

    # Load the input low-res DT image:
    print('... loading the test low-res image ...')
    dt_lowres = sr_utility.read_dt_volume(nameroot=os.path.join(gt_dir, 'dt_b1000_lowres_2_'))

    # clear the graph (is it necessary?)
    tf.reset_default_graph()

    # Reconstruct:
    nn_file = sr_utility.name_network(method=method, n_h1=n_h1, n_h2=n_h2, n_h3=n_h3, cohort=cohort, no_subjects=no_subjects,
                                      sample_rate=sample_rate, us=us, n=n, m=m,
                                      optimisation_method=optimisation_method, dropout_rate=dropout_rate)
    network_dir = os.path.join(model_dir, nn_file)  # full path to the model you want to restore in testing
    print('\nReconstruct high-res dti with the network: \n%s.' % network_dir)
    dt_hr = super_resolve(dt_lowres, method=method, n_h1=n_h1, n_h2=n_h2, n=n, m=m, us=us,
                          dropout_rate=0.0,  # Here set the drop-out rate to be zero.
                          network_dir=network_dir)
    end_time = timeit.default_timer()

    # Save:
    output_file = os.path.join(recon_dir, 'dt_' + nn_file + '.npy')
    print('... saving as %s' % output_file)
    np.save(output_file, dt_hr)
    print('\nIt took %f secs to reconsruct a whole brain volumne. \n' % (end_time - start_time))

    # Compute the reconstruction error:
    recon_dir, recon_file = os.path.split(output_file)
    rmse, rmse_volume = sr_utility.compute_rmse(recon_file=recon_file, recon_dir=recon_dir, gt_dir=gt_dir)
    print('\nReconsturction error (RMSE) is %f.' % rmse)

    # Save each estimated dti separately as a nifti file for visualisation:
    print('\nSave each estimated dti separately as a nifti file for visualisation ...')
    sr_utility.save_as_nifti(recon_file=recon_file, recon_dir=recon_dir, gt_dir=gt_dir)
