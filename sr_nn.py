""" Train and evaluates the super-resolution network:
 1. sr_train() - train a specified model with the chosen method.
 2. super_resolve() - function to perform super-resolution on a given low-res DT image
 3. sr_reconstruct() - the main code to run the super-resolution/compute errors/save files in required formats, etc.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cPickle

import numpy as np
import timeit
import tensorflow as tf

# Set temporarily the git dir on python path. In future, remove this and add the dir to search path.
import sys
sys.path.append('/Users/ryutarotanno/DeepLearning/nsampler/codes')   # dir name of the git repo
import sr_utility  # utility functions for loading/processing data
import models


def sr_train(method='linear', n_h1=500, n_h2=200,
             data_dir='/Users/ryutarotanno/DeepLearning/Test_1/data/',
             cohort='Diverse', no_subjects=8, sample_rate=32, us=2, n=2, m=2,
             optimisation_method='standard', dropout_rate=0.0, learning_rate=1e-4, L1_reg=0.00, L2_reg=1e-5,
             n_epochs=1000, batch_size=25,
             save_dir='/Users/ryutarotanno/DeepLearning/nsampler/models'):

    ##########################
    # Load the training data:
    ##########################
    # get the full path to the training set:
    dataset = data_dir + 'PatchLibs%sDS%02i_%ix%i_%ix%i_TS%i_SRi%03i_0001.mat' \
                         % (cohort, us, 2 * n + 1, 2 * n + 1, m, m, no_subjects, sample_rate)
    data_dir, data_file = os.path.split(dataset)

    # load
    print('... loading the training dataset %s' % data_file)
    patchlib = sr_utility.load_patchlib(patchlib=dataset)
    train_set_x, valid_set_x, train_set_y, valid_set_y = patchlib  # load the original patch libs

    # normalise the data and keep the transforms:
    (train_set_x_scaled, train_set_x_mean, train_set_x_std, train_set_y_scaled, train_set_y_mean, train_set_y_std)\
        = sr_utility.standardise_data(train_set_x, train_set_y, option='default')  # normalise the data

    # normalise the validation sets into the same space as training sets:
    valid_set_x_scaled = (valid_set_x - train_set_x_mean) / train_set_x_std
    valid_set_y_scaled = (valid_set_y - train_set_y_mean) / train_set_y_std
    del train_set_x, valid_set_x, train_set_y, valid_set_y, patchlib  # clear original data as you don't need them.


    ####################
    # Define the model:
    ####################
    print('... defining the model')

    # clear the graph
    tf.reset_default_graph()

    # define input and output:
    n_in, n_out = 6 * (2 * n + 1) ** 3, 6 * m ** 3  # dimensions of input and output
    x_scaled = tf.placeholder(tf.float32, shape=[None, n_in])  # normalised input low-res patch
    y_scaled = tf.placeholder(tf.float32, shape=[None, n_out])  # normalised output high-res patch
    keep_prob = tf.placeholder(tf.float32)  # keep probability for dropout

    y_pred_scaled, L2_sqr, L1 = models.inference(method, x_scaled, keep_prob, n_in, n_out, n_h1, n_h2)
    cost = models.cost(y_scaled, y_pred_scaled, L2_sqr, L1, L2_reg, L1_reg)
    train_step = models.training(cost, learning_rate, option=optimisation_method)
    mse = tf.reduce_mean(tf.square(train_set_y_std * (y_scaled - y_pred_scaled)))


    #######################
    # Start training:
    #######################
    # Add the variable initializer Op.
    init = tf.initialize_all_variables()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    print('... training')

    with tf.Session() as sess:
        # Run the Op to initialize the variables.
        sess.run(init)

        # Compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x_scaled.shape[0] // batch_size
        n_valid_batches = valid_set_x_scaled.shape[0] // batch_size

        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
        # found
        improvement_threshold = 0.995  # a relative improvement of this much is
        # considered significant
        validation_frequency = min(n_train_batches, patience // 2)
        # go through this many
        # minibatche before checking the network
        # on the validation set; in this case we
        # check every epoch

        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0.

        start_time = timeit.default_timer()

        epoch = 0
        done_looping = False

        while (epoch < n_epochs) and (not done_looping):
            epoch += 1
            for minibatch_index in range(n_train_batches):

                # perform gradient descent:
                train_step.run(
                    feed_dict={x_scaled: train_set_x_scaled[minibatch_index * batch_size:(minibatch_index + 1) * batch_size, :],
                               y_scaled: train_set_y_scaled[minibatch_index * batch_size:(minibatch_index + 1) * batch_size, :],
                               keep_prob: (1.0 - dropout_rate)})

                # iteration number
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set

                    validation_losses = [mse.eval(
                        feed_dict={x_scaled: valid_set_x_scaled[index * batch_size:(index + 1) * batch_size, :],
                                   y_scaled: valid_set_y_scaled[index * batch_size:(index + 1) * batch_size, :],
                                   keep_prob: (1.0 - dropout_rate)}
                        )
                        for index in range(n_valid_batches)]

                    this_validation_loss = np.mean(validation_losses)

                    training_losses = [mse.eval(
                        feed_dict={x_scaled: train_set_x_scaled[index * batch_size:(index + 1) * batch_size, :],
                                   y_scaled: train_set_y_scaled[index * batch_size:(index + 1) * batch_size, :],
                                   keep_prob: (1.0 - dropout_rate)}
                        )
                        for index in range(n_valid_batches)]

                    this_training_loss = np.mean(training_losses)

                    print(
                        '\nEpoch %i, minibatch %i/%i:\n'
                        '     training error (rmse) %f times 1E-5\n'
                        '     validation error (rmse) %f times 1E-5' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            np.sqrt(this_training_loss * 10 ** 10),
                            np.sqrt(this_validation_loss * 10 ** 10)
                        )
                    )
                    print('     number of minibatches = %i and patience = %i' % (iter, patience))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        # improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *improvement_threshold :

                            patience = max(patience, iter * patience_increase)
                            print('     reduces the previous error by more than %f %%'
                                  % ((1 - improvement_threshold) * 100.))

                        best_validation_loss = this_validation_loss
                        best_iter = iter

                if patience <= iter:
                    done_looping = True
                    break

        end_time = timeit.default_timer()
        time_train = end_time - start_time

        print(('\nOptimization complete. Best validation score of %f  '
               'obtained at iteration %i') %
              (np.sqrt(best_validation_loss * 10**10), best_iter + 1))

        print('Training done!!! It took %f secs.' % time_train)

        # Save the model:
        nn_file = sr_utility.name_network(method=method, n_h1=n_h1, n_h2=n_h2, cohort=cohort, no_subjects=no_subjects,
                                          sample_rate=sample_rate, us=us, n=n, m=m,
                                          optimisation_method=optimisation_method, dropout_rate=dropout_rate)

        save_subdir = os.path.join(save_dir, nn_file)

        if not os.path.exists(save_subdir):  # create a subdirectory to save the model.
            os.makedirs(save_subdir)

        save_path = saver.save(sess, os.path.join(save_subdir, "model.ckpt"))
        print("Model saved in file: %s" % save_path)

        # Save the model details:
        print('... saving the model details')
        model_details = {'method': method, 'cohort': cohort,
                         'no of subjects': no_subjects, 'sample rate': sample_rate, 'upsampling factor': us, 'n': n,
                         'm': m, 'optimisation': optimisation_method, 'dropout rate': dropout_rate,
                         'learning rate': learning_rate,
                         'L1 coefficient': L1_reg, 'L2 coefficient': L2_reg, 'max no of epochs': n_epochs,
                         'batch size': batch_size, 'training length': time_train,
                         'best validation rmse': np.sqrt(best_validation_loss)}
        cPickle.dump(model_details, file(os.path.join(save_subdir, 'settings.pkl'), 'wb'),
                     protocol=cPickle.HIGHEST_PROTOCOL)

        print('... saving the transforms used for data normalisation for the test time')
        transform = {'input_mean': train_set_x_mean, 'input_std': train_set_x_std,
                     'output_mean': train_set_y_mean, 'output_std': train_set_y_std}
        f = file(os.path.join(save_subdir, 'transforms.pkl'), 'wb')
        cPickle.dump(transform, f, protocol=cPickle.HIGHEST_PROTOCOL)

    # clear the graph
    tf.reset_default_graph()


# Reconstruct using the specified NN:
def super_resolve(dt_lowres, method='mlp_h=1', n_h1=500, n_h2=200, n=2, m=2, us=2, dropout_rate=0.0,
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
    y_pred_scaled, L2_sqr, L1 = models.inference(method, x_scaled, keep_prob, n_in, n_out, n_h1, n_h2)

    # load the transforms used for normalisation of the training data:
    transform_file = os.path.join(network_dir, 'transforms.pkl')
    transform = cPickle.load(open(transform_file, 'rb'))
    train_set_x_mean = transform['input_mean'].reshape((1, n_in))  # row vector representing the mean
    train_set_x_std = transform['input_std'].reshape((1, n_in))
    train_set_y_mean = transform['output_mean'].reshape((1, n_out))
    train_set_y_std = transform['output_std'].reshape((1, n_out))
    del transform

    # Restore all the variables and perform reconstruction:
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, os.path.join(network_dir, "model.ckpt"))
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
                                                                          keep_prob: (1.0-dropout_rate)})

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


def sr_reconstruct(method='linear', n_h1=500, n_h2=200, us=2, n=2, m=2,
                   optimisation_method='standard', dropout_rate=0.0, cohort='Diverse', no_subjects=8, sample_rate=32,
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
    nn_file = sr_utility.name_network(method=method, n_h1=n_h1, n_h2=n_h2, cohort=cohort, no_subjects=no_subjects,
                                      sample_rate=sample_rate, us=us, n=n, m=m,
                                      optimisation_method=optimisation_method, dropout_rate=dropout_rate)
    network_dir = os.path.join(model_dir, nn_file)  # full path to the model you want to restore in testing
    print('\nReconstruct high-res dti with the network: \n%s.' % network_dir)
    dt_hr = super_resolve(dt_lowres, method=method, n_h1=n_h1, n_h2=n_h2, n=n, m=m, us=us, network_dir=network_dir)
    end_time = timeit.default_timer()

    # Save:
    output_file = os.path.join(recon_dir, 'dt_' + nn_file + '.npy')
    print('... saving as %s' % output_file)
    np.save(output_file, dt_hr)
    print('\nIt took %f secs to reconsruct a whole brain volumne. \n' % (end_time - start_time))

    # Compute the reconstruction error:
    recon_dir, recon_file = os.path.split(output_file)
    rmse = sr_utility.compute_rmse(recon_file=recon_file, recon_dir=recon_dir, gt_dir=gt_dir)
    print('\nReconsturction error (RMSE) is %f.' % rmse)

    # Save each estimated dti separately as a nifti file for visualisation:
    print('\nSave each estimated dti separately as a nifti file for visualisation ...')
    sr_utility.save_as_nifti(recon_file=recon_file, recon_dir=recon_dir, gt_dir=gt_dir)


if __name__ == "__main__":
    sr_train()
    sr_reconstruct()