"""sr_bayes is a collection of Bayesian NN scripts"""

import os
import sys
import time

import numpy as np
import tensorflow as tf

import models
import sr_utility
import timeit
import cPickle

def sr_train(method='mlp_h=1_kingma', n_h1=500, n_h2=200,
             data_dir='./data/',
             cohort='Diverse', no_subjects=8, sample_rate=32, us=2, n=2, m=2,
             optimisation_method='adam', dropout_rate=0.5, learning_rate=1e-4, L1_reg=0.00, L2_reg=1e-5,
             n_epochs=1000, batch_size=25,
             save_dir='./models'):

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
    del train_set_x, valid_set_x, train_set_y, valid_set_y  # clear original data as you don't need them.


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

    y_pred_scaled, L2_sqr, L1, reg = models.inference(method, x_scaled, keep_prob, n_in, n_out, n_h1, n_h2)
    cost = models.cost(y_scaled, y_pred_scaled, L2_sqr, L1, L2_reg, L1_reg)
    train_step = models.training(cost, learning_rate, option=optimisation_method)
    mse = tf.reduce_mean(tf.square(train_set_y_std * (y_scaled - y_pred_scaled)))
    cost += tf.add_n(reg)/3.

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
        print(('\nOptimization complete. Best validation score of %f  '
               'obtained at iteration %i') %
              (np.sqrt(best_validation_loss * 10**10), best_iter + 1))

        print('Training done!!! It took %f secs.' % (end_time - start_time))

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
                         'batch size': batch_size}
        cPickle.dump(model_details, file(os.path.join(save_subdir, 'settings.pkl'), 'wb'),
                     protocol=cPickle.HIGHEST_PROTOCOL)

        print('... saving the transforms used for data normalisation for the test time')
        transform = {'input_mean': train_set_x_mean, 'input_std': train_set_x_std,
                     'output_mean': train_set_y_mean, 'output_std': train_set_y_std}
        f = file(os.path.join(save_subdir, 'transforms.pkl'), 'wb')
        cPickle.dump(transform, f, protocol=cPickle.HIGHEST_PROTOCOL)

    # clear the graph
    tf.reset_default_graph()


if __name__ == '__main__':
    sr_train()