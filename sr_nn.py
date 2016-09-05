""" Train and test """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Set temporarily the git dir on python path. In future, remove this and add the dir to search path.
import sys
sys.path.append('/Users/ryutarotanno/DeepLearning/nsampler/codes')   # dir name of the git repo
import sr_utility  # utility functions for loading/processing data

import os
import cPickle

import numpy as np
import timeit
import tensorflow as tf

#####################
# Train your network:
#####################

def sr_train(method='linear', n_h1=500, n_h2=200,
             save_dir='/Users/ryutarotanno/DeepLearning/nsampler/models',
             n=2, m=2, us=2, learning_rate=0.01, L1_reg=0.00, L2_reg=1e-4,
             n_epochs=1000, batch_size=25):

    ##########################
    # Load the training data:
    ##########################

    dataset = '/Users/ryutarotanno/DeepLearning/Test_1/data/' \
              + 'PatchLibsDiverseDS02_5x5_2x2_TS8_SRi032_0001.mat'
    data_dir, data_file = os.path.split(dataset)

    print('... loading the training dataset %s' % data_file)
    patchlib = sr_utility.load_patchlib(patchlib=dataset)
    train_set_x, valid_set_x, train_set_y, valid_set_y = patchlib  # load the original patch libs

    # normalise the data and keep the transforms:
    (train_set_x_scaled, train_set_x_mean, train_set_x_std, train_set_y_scaled, train_set_y_mean, train_set_y_std)\
        = sr_utility.standardise_data(train_set_x, train_set_y, option='default')  # normalise the data

    valid_set_x_scaled = (valid_set_x - train_set_x_mean) / train_set_x_std
    valid_set_y_scaled = (valid_set_y - train_set_y_mean) / train_set_y_std

    del train_set_x, valid_set_x, train_set_y, valid_set_y  # clear original data as you don't need them.

    print('... saving the transforms used for data normalisation for the test time')
    save_subdir = os.path.join(save_dir, method)
    if not os.path.exists(save_subdir):
        os.makedirs(save_subdir)
    transform = {'input_mean': train_set_x_mean, 'input_std': train_set_x_std,
                 'output_mean': train_set_y_mean, 'output_std': train_set_y_std}
    f = file(os.path.join(save_subdir,'transforms.pkl'), 'wb')
    cPickle.dump(transform, f, protocol=cPickle.HIGHEST_PROTOCOL)

    ####################
    # Define the model:
    ####################

    print('... defining the model')

    # define input and output:
    n_in, n_out = 6 * (2 * n + 1) ** 3, 6 * m ** 3  # dimensions of input and output
    x_scaled = tf.placeholder(tf.float32, shape=[None, n_in])  # normalised input low-res patch
    y_scaled = tf.placeholder(tf.float32, shape=[None, n_out])  # normalised output high-res patch

    # Build the selected model: followed http://cs231n.github.io/neural-networks-2/ for initialisation.
    if method == 'linear':
        # Standard linear regression:
        W1 = tf.Variable(
            tf.random_normal([n_in, n_out], stddev=np.sqrt(2.0/n_in)),
            name='W1')
        b1 = tf.Variable(tf.constant(1e-2, shape=[n_out]), name='b1')

        y_pred_scaled = tf.matmul(x_scaled, W1) + b1  # predicted high-res patch in the normalised space
        # y_pred = train_set_y_std*y_pred_scaled + train_set_y_mean  # predicted high-res patch in the original space

        # Predictive metric and regularisers:
        mse_scaled = tf.reduce_mean((y_scaled - y_pred_scaled) ** 2)  # mse in the normalised space
        mse = tf.reduce_mean(tf.square(train_set_y_std * (y_scaled - y_pred_scaled)))  # mse in the original space
        L2_sqr = tf.reduce_sum(W1 ** 2)
        L1 = tf.reduce_sum(tf.abs(W1))

        # Objective function:
        cost = mse_scaled + L2_reg * L2_sqr + L1_reg * L1

    elif method == 'mlp_h=1':
        # MLP with one hidden layer:
        W1 = tf.Variable(
            tf.random_normal([n_in, n_h1], stddev=np.sqrt(2.0/n_in)),
            name='W1')
        b1 = tf.Variable(tf.constant(1e-2, shape=[n_h1]), name='b1')

        hidden1 = tf.nn.relu(tf.matmul(x_scaled, W1) + b1)

        W2 = tf.Variable(
            tf.random_normal([n_h1, n_out], stddev=np.sqrt(2.0/n_h1)),
            name='W2')
        b2 = tf.Variable(tf.constant(1e-2, shape=[n_out]), name='b2')

        y_pred_scaled = tf.matmul(hidden1, W2) + b2

        # Predictive metric and regularisers:
        mse_scaled = tf.reduce_mean((y_scaled - y_pred_scaled) ** 2)
        mse = tf.reduce_mean(tf.square(train_set_y_std * (y_scaled - y_pred_scaled)))
        L2_sqr = tf.reduce_sum(W1 ** 2) + tf.reduce_sum(W2 ** 2)
        L1 = tf.reduce_sum(tf.abs(W1)) + tf.reduce_sum(tf.abs(W2))

        # Objective function:
        cost = mse_scaled + L2_reg * L2_sqr + L1_reg * L1

    elif method == 'mlp_h=2':
        # MLP with two hidden layers:
        W1 = tf.Variable(
            tf.random_normal([n_in, n_h1], stddev=np.sqrt(2.0/n_in)),
            name='W1')
        b1 = tf.Variable(tf.constant(1e-2, shape=[n_h1]), name='b1')

        hidden1 = tf.nn.relu(tf.matmul(x_scaled, W1) + b1)

        W2 = tf.Variable(
            tf.random_normal([n_h1, n_h2], stddev=np.sqrt(2.0/n_h1)),
            name='W2')
        b2 = tf.Variable(tf.constant(1e-2, shape=[n_h2]), name='b2')

        hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + b2)

        W3 = tf.Variable(
            tf.random_normal([n_h2, n_out], stddev=np.sqrt(2.0/n_h2)),
            name='W3')
        b3 = tf.Variable(tf.constant(1e-2, shape=[n_out]), name='b3')

        y_pred_scaled = tf.matmul(hidden2, W3) + b3

        # Predictive metric and regularisers:
        mse_scaled = tf.reduce_mean((y_scaled - y_pred_scaled) ** 2)
        mse = tf.reduce_mean(tf.square(train_set_y_std * (y_scaled - y_pred_scaled)))
        L2_sqr = tf.reduce_sum(W1 ** 2) + tf.reduce_sum(W2 ** 2) + tf.reduce_sum(W3 ** 2)
        L1 = tf.reduce_sum(tf.abs(W1)) + tf.reduce_sum(tf.abs(W2)) + tf.reduce_sum(tf.abs(W3))

        # Objective function:
        cost = mse_scaled + L2_reg * L2_sqr + L1_reg * L1

    else:
        raise ValueError('The chosen method not available ...')

    # Define the optimisation method:
    # train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


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
            epoch = epoch + 1
            for minibatch_index in range(n_train_batches):

                # perform gradient descent:
                train_step.run(
                    feed_dict={x_scaled: train_set_x_scaled[minibatch_index * batch_size:(minibatch_index + 1) * batch_size, :],
                               y_scaled: train_set_y_scaled[minibatch_index * batch_size:(minibatch_index + 1) * batch_size, :]})

                # iteration number
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set

                    validation_losses = [mse.eval(
                        feed_dict={x_scaled: valid_set_x_scaled[index * batch_size:(index + 1) * batch_size, :],
                                   y_scaled: valid_set_y_scaled[index * batch_size:(index + 1) * batch_size, :]}
                        )
                        for index in range(n_valid_batches)]

                    this_validation_loss = np.mean(validation_losses)

                    training_losses = [mse.eval(
                        feed_dict={x_scaled: train_set_x_scaled[index * batch_size:(index + 1) * batch_size, :],
                                   y_scaled: train_set_y_scaled[index * batch_size:(index + 1) * batch_size, :]}
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
        print(('\nOptimization complete. Best validation score of %f  '
               'obtained at iteration %i') %
              (np.sqrt(best_validation_loss * 10**10), best_iter + 1))

        print('Training done!!! It took %f secs.' % (end_time - start_time))

        save_subdir = os.path.join(save_dir, method)
        if not os.path.exists(save_subdir):
            os.makedirs(save_subdir)

        save_path = saver.save(sess, os.path.join(save_subdir, "model.ckpt"))
        print("Model saved in file: %s" % save_path)

        # # clear the whole graph:
        # if reset:
        #     print("Clear all tensors and ops")
        #     tf.reset_default_graph()


# Reconstruct using the specified NN:
def super_resolve(dt_lowres, method='mlp_h=1', n_h1=500, n_h2=200, n=2, m=2, us=2,
                    network_dir =  '/Users/ryutarotanno/DeepLearning/nsampler/models'):
    """Perform a patch-based super-resolution on a given low-res image.
    Args:
        dt_lowres (numpy array): a low-res diffusion tensor image volume
        n (int): the width of an input patch is 2*n + 1
        m (int): the width of an output patch is m
        us (int): the upsampling factord
    Returns:
        the estimated high-res volume
    """

    ######################
    # Specify the network:
    ######################

    print('... defining the network model %s .' % method)
    n_in, n_out = 6 * (2 * n + 1) ** 3, 6 * m ** 3  # dimensions of input and output
    x = tf.placeholder(tf.float32, shape=[None, n_in])
    y = tf.placeholder(tf.float32, shape=[None, n_out])

    # load the transforms used for normalisation of the training data:
    transform_file = os.path.join(network_dir, method, 'transforms.pkl')
    transform = cPickle.load(open(transform_file, 'rb'))
    train_set_x_mean = transform['input_mean'].reshape((1, n_in))  # row vector representing the mean
    train_set_x_std = transform['input_std'].reshape((1, n_in))
    train_set_y_mean = transform['output_mean'].reshape((1, n_out))
    train_set_y_std = transform['output_std'].reshape((1, n_out))
    del transform

    if method == 'linear':
        # Redefine the graph:
        pretrained = False
        W1 = tf.Variable(
            tf.random_uniform([n_in, n_out], minval=-np.sqrt(6. / (n_in + n_out)), maxval=np.sqrt(6. / (n_in + n_out))),
            name='W1')
        b1 = tf.Variable(tf.constant(0.0, shape=[n_out]), name='b1')

        y_pred = tf.matmul(x, W1) + b1

    elif method == 'mlp_h=1':
        # MLP with one hidden layer:
        pretrained = False
        W1 = tf.Variable(
            tf.random_uniform([n_in, n_h1], minval=-np.sqrt(6. / (n_in + n_out)), maxval=np.sqrt(6. / (n_in + n_out))),
            name='W1')
        b1 = tf.Variable(tf.constant(0.0, shape=[n_h1]), name='b1')

        hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)

        W2 = tf.Variable(
            tf.random_uniform([n_h1, n_out], minval=-np.sqrt(6. / (n_in + n_out)), maxval=np.sqrt(6. / (n_in + n_out))),
            name='W2')
        b2 = tf.Variable(tf.constant(0.0, shape=[n_out]), name='b2')

        y_pred = tf.matmul(hidden1, W2) + b2

    elif method == 'mlp_h=2':
        # MLP with two hidden layers:
        pretrained = False
        W1 = tf.Variable(
            tf.random_uniform([n_in, n_h1], minval=-np.sqrt(6. / (n_in + n_h1)), maxval=np.sqrt(6. / (n_in + n_h1))),
            name='W1')
        b1 = tf.Variable(tf.constant(0.0, shape=[n_h1]), name='b1')

        hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)

        W2 = tf.Variable(
            tf.random_uniform([n_h1, n_h2], minval=-np.sqrt(6. / (n_h1 + n_h2)), maxval=np.sqrt(6. / (n_h1 + n_h2))),
            name='W2')
        b2 = tf.Variable(tf.constant(0.0, shape=[n_h2]), name='b2')

        hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + b2)

        W3 = tf.Variable(
            tf.random_uniform([n_h2, n_out], minval=-np.sqrt(6. / (n_h2 + n_out)), maxval=np.sqrt(6. / (n_h2 + n_out))),
            name='W3')
        b3 = tf.Variable(tf.constant(0.0, shape=[n_out]), name='b3')

        y_pred = tf.matmul(hidden2, W3) + b3

    elif method == 'linear_test':
        pretrained = True
        print('use the weights of the manually pre-trained linear regression !')
        weights_file = '/Users/ryutarotanno/DeepLearning/Test_1/models/linear_test.npy'
        W_temp = np.load(weights_file)

        n_in, n_out = 6 * (2 * n + 1) ** 3, 6 * m ** 3  # dimensions of input and output
        x = tf.placeholder(tf.float32, shape=[None, n_in])
        y = tf.placeholder(tf.float32, shape=[None, n_out])

        W1 = tf.Variable(W_temp.T[:-1, :].astype('float32'), name='W1')
        b1 = tf.Variable(1E-3*W_temp.T[-1, :].astype('float32'), name='b1')

        y_pred = tf.matmul(x, W1) + b1

    else:
        raise ValueError('No correct model specified')


    ##############
    # Reconstruct!
    ##############

    # Restore all the variables and perform reconstruction:
    saver = tf.train.Saver()
    init_op = tf.initialize_all_variables()

    with tf.Session() as sess:
        if pretrained:
            sess.run(init_op)
        else:
            # Restore variables from disk.
            saver.restore(sess, os.path.join(network_dir, method, "model.ckpt"))
            print("Model restored.")

        # reconstruct
        dt_lowres = dt_lowres[0::us, 0::us, 0::us, :]  # take every us th entry to reduce it to the original resolution.
        (xsize, ysize, zsize, comp) = dt_lowres.shape
        dt_hires = np.zeros((xsize * us, ysize * us, zsize * us, comp)) # the base array for the output high-res volume.
        dt_hires[:, :, :, 0] = -1 # initialise all the voxels as 'background'.

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
                        opatch_row_scaled = y_pred.eval(feed_dict={x: ipatch_row_scaled})

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


def sr_reconstruct(method='mlp_h=1', n=2, m=2, us=2, n_h1=500, n_h2=200,
                           recon_dir='/Users/ryutarotanno/DeepLearning/nsampler/recon',
                           gt_dir='/Users/ryutarotanno/DeepLearning/Test_1/data'):

    start_time = timeit.default_timer()

    # Load the input low-res DT image:
    print('... loading the test low-res image ...')
    dt_lowres = sr_utility.read_dt_volume(nameroot=os.path.join(gt_dir, 'dt_b1000_lowres_2_'))

    # Reconstruct and save:
    print('\nReconstruct high-res dti with method = %s ...' % method)
    dt_hr = super_resolve(dt_lowres, method=method, n_h1=n_h1, n_h2=n_h2, n=n, m=m, us=us)
    end_time = timeit.default_timer()

    output_file = os.path.join(recon_dir, method + '_highres_dti.npy')
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