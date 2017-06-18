"""Training file (Ryu) """

import os
import glob
import shutil
import sys
import timeit
sys.path.append("../common")

import cPickle as pkl
import numpy as np
import tensorflow as tf

import models

from data_generator import prepare_data
from ops import get_tensor_shape
from utils import *


def update_best_loss(this_loss, bests, iter_, current_step):
    bests['counter'] += 1
    if this_loss < bests['val_loss']:
        bests['counter'] = 0
        bests['val_loss'] = this_loss
        bests['iter_'] = iter_
        bests['step'] = current_step + 1
    return bests


def update_best_loss_epoch(this_loss, bests, current_step):
    if this_loss < bests['val_loss_save']:
        bests['val_loss_save'] = this_loss
        bests['step_save'] = current_step + 1
    return bests


def initialise_model(sess, saver, phase_train, checkpoint_dir, bests, opt):
    # Initialization:
    if opt['continue']:
        # Specify the network parameters to be restored:
        if os.path.exists(os.path.join(checkpoint_dir, 'settings.pkl')):
            print('continue from the previous training ...')
            model_details = pkl.load(
                open(os.path.join(opt['checkpoint_dir'], 'settings.pkl'), 'rb'))
            nn_file = os.path.join(opt['checkpoint_dir'],
                                   "model-" + str(model_details['step_save']))
            # Initialise the best parameters dict with the previous training:
            bests.update(model_details)

            # Set the initial epoch:
            epoch_init = model_details['last_epoch']

            # Initialise and then restore the previous model parameters:
            # init_global = tf.global_variables_initializer()
            # init_local = tf.local_variables_initializer()
            # sess.run([init_global,init_local], feed_dict={phase_train: True})
            saver.restore(sess, nn_file)
        else:
            print('no trace of previous training!')
            print('intialise and start training from scratch.')
            # init = tf.initialize_all_variables()
            init_global = tf.global_variables_initializer()
            init_local = tf.local_variables_initializer()
            epoch_init = 0
            sess.run([init_global,init_local], feed_dict={phase_train: True})
    else:
        if os.path.exists(os.path.join(checkpoint_dir, 'settings.pkl')):
            if opt['overwrite']:
                print('Overwriting the previous trained model ...')
            else:
                print(
                    'You have selected not to overwrite. Stop training ...')
                return
        else:
            print('Start brand new training ...')

        # init = tf.initialize_all_variables()
        init_global = tf.global_variables_initializer()
        init_local = tf.local_variables_initializer()
        sess.run([init_global, init_local], feed_dict={phase_train: True})
        epoch_init = 0

    return epoch_init


def get_output_radius(y_pred, upsampling_rate, is_shuffle):
    """ Compute the output radius """
    if is_shuffle:
        # output radius in low-resolution:
        output_radius = get_tensor_shape(y_pred)[1] // 2
    else:
        output_radius = (get_tensor_shape(y_pred)[1]/upsampling_rate) // 2
        assert get_tensor_shape(y_pred)[1] % upsampling_rate == 0
    return output_radius


def get_optimizer(optimizer, lr):
    if optimizer=='adam':
        optim = tf.train.AdamOptimizer(learning_rate=lr)
    else:
        raise Exception('Specified optimizer not available.')
    return optim


def set_network_config(opt):
    """ Define the model type"""
    if opt["method"] == "espcn":
        assert opt["is_shuffle"]
        net = models.espcn(upsampling_rate=opt['upsampling_rate'],
                           out_channels=opt['no_channels'],
                           filters_num=opt['no_filters'],
                           layers=opt['no_layers'],
                           bn=opt['is_BN'])

    elif opt["method"] == "espcn_deconv" :
        assert not(opt["is_shuffle"])
        net = models.espcn_deconv(upsampling_rate=opt['upsampling_rate'],
                                  out_channels=opt['no_channels'],
                                  filters_num=opt['no_filters'],
                                  layers=opt['no_layers'],
                                  bn=opt['is_BN'])
    elif opt["method"] == "segnet":
        net = models.unet(upsampling_rate=opt['upsampling_rate'],
                          out_channels=opt['no_channels'],
                          filters_num=opt['no_filters'],
                          layers=opt['no_layers'],
                          conv_num=2,
                          bn=opt['is_BN'],
                          is_concat=False)

    elif opt["method"] == "unet":
        net = models.unet(upsampling_rate=opt['upsampling_rate'],
                          out_channels=opt['no_channels'],
                          filters_num=opt['no_filters'],
                          layers=opt['no_layers'],
                          conv_num=2,
                          bn=opt['is_BN'],
                          is_concat=True)
    else:
        raise ValueError("The specified network type %s not available" %
                        (opt["method"],))
    return net


def train_cnn(opt):
    # ----------------------- DEFINE THE MODEL ---------------------------------
    # Currently, the size of the output radius is only computed after defining
    # the model.

    # define place holders and network:
    # todo: need to define separately the number of input/output channels
    # todo: allow for input of even numbered size

    x = tf.placeholder(tf.float32,
                       [opt["batch_size"],
                       2*opt['input_radius']+1,
                       2*opt['input_radius']+1,
                       2*opt['input_radius']+1,
                       opt['no_channels']],
                       name='input_x')
    phase_train = tf.placeholder(tf.bool, name='phase_train')
    net = set_network_config(opt)
    y_pred = net.forwardpass(x, phase_train)
    y = tf.placeholder(tf.float32, get_tensor_shape(y_pred), name='input_y')

    # other place holders:
    keep_prob = tf.placeholder(tf.float32, name='dropout_rate')
    trade_off = tf.placeholder(tf.float32, name='trade_off')
    global_step = tf.Variable(0, name="global_step", trainable=False)
    transform = tf.placeholder(tf.float32, name='norm_transform')

    # define loss and evaluation criteria:
    cost = net.cost(y, y_pred)
    mse = tf.reduce_mean(tf.square(transform * (y - y_pred)))
    tf.summary.scalar('mse', mse)

    # define training op
    lr = tf.placeholder(tf.float32, [], name='learning_rate')
    optim = get_optimizer(opt["optimizer"], lr)
    train_step = optim.minimize(cost, global_step=global_step)

    # ----------------------- Directory settings -------------------------------
    # compute the output radius (needed for defining the network name):
    opt['output_radius'] = get_output_radius(y_pred, opt['upsampling_rate'],
                                             opt['is_shuffle'])

    # Create the root model directory:
    if not (os.path.exists(opt['save_dir'] + name_network(opt))):
        os.makedirs(opt['save_dir'] + name_network(opt))

    # Save displayed output to a text file:
    if opt['disp']:
        f = open(opt['save_dir'] + name_network(opt) + '/output.txt', 'w')
        # Redirect all the outputs to the text file:
        print("Redirecting the output to: "
              + opt['save_dir'] + name_network(opt) + "/output.txt")
        sys.stdout = f

    # Set the directory for saving checkpoints:
    checkpoint_dir = define_checkpoint(opt)
    log_dir = define_logdir(opt)
    opt["checkpoint_dir"] = checkpoint_dir
    with open(checkpoint_dir + '/config.txt', 'w') as fp:
        for p in sorted(opt.iteritems(), key=lambda (k, v): (v, k)):
            fp.write("%s:%s\n" % p)

    # Exit if network is already trained unless specified otherwise:
    if os.path.exists(os.path.join(checkpoint_dir, 'settings.pkl')):
        if not (opt['continue']) and not (opt['overwrite']):
            print('Network already trained. Move on to next.')
            return
        elif opt['overwrite']:
            print('Overwriting: delete the previous results')
            files = glob.glob(
                opt['save_dir'] + '/' + name_network(opt) + '/model*')
            files.extend(glob.glob(
                opt['save_dir'] + '/' + name_network(opt) + '/checkpoint*'))
            for f in files: os.remove(f)
            shutil.rmtree(opt['log_dir'] + '/' + name_network(opt))

    # -------------------- SET UP THE DATA LOADER ------------------------------
    filename_patchlib = name_patchlib(opt)

    dataset, train_folder = prepare_data(size=opt['train_size'],
                                         eval_frac=opt['validation_fraction'],
                                         inpN=opt['input_radius'],
                                         outM=opt['output_radius'],
                                         no_channels=opt['no_channels'],
                                         patchlib_name=filename_patchlib,
                                         method=opt['patch_sampling_opt'],
                                         whiten=opt['transform_opt'],
                                         inp_header=opt['input_file_name'],
                                         out_header=opt['gt_header'],
                                         train_index=opt['train_subjects'],
                                         bgval=opt['background_value'],
                                         is_reset=opt['is_reset'],
                                         clip=opt['is_clip'],
                                         shuffle=opt['is_shuffle'],
                                         pad_size=opt['pad_size'],
                                         us_rate=opt['upsampling_rate'],
                                         data_dir_root=opt['gt_dir'],
                                         save_dir_root=opt['data_dir'],
                                         subpath=opt['subpath'])

    print(opt['input_radius'], opt['output_radius'], opt['is_shuffle'], dataset._shuffle)

    opt['train_noexamples'] = dataset.size
    opt['valid_noexamples'] = dataset.size_valid
    print('Patch-lib size:', opt['train_noexamples'] + opt['valid_noexamples'],
          'Train size:', opt['train_noexamples'],
          'Valid size:', opt['valid_noexamples'])


    # ######################### START TRAINING ###################
    saver = tf.train.Saver()
    print('\nStart training!\n')
    with tf.Session() as sess:

        # Merge all the summaries and write them out to ../network_name/log
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
        valid_writer = tf.summary.FileWriter(log_dir + '/valid')

        # Compute number of minibatches for training, validation and testing
        n_train_batches = opt['train_noexamples'] // opt['batch_size']
        n_valid_batches = opt['valid_noexamples'] // opt['batch_size']

        # Compute the trade-off values:
        tradeoff_list = models.get_tradeoff_values_v2(opt['method'], opt['no_epochs'])

        # Define the normalisation tranform:
        norm_std = dataset._transform['output_std']

        # Define some counters
        start_time = timeit.default_timer()
        epoch = 0
        done_looping = False
        iter_valid = 0
        total_val_mse_epoch = 0
        total_tr_mse_epoch = 0
        total_tr_cost_epoch = 0
        total_val_cost_epoch = 0

        # Define dictionary to save some results:
        bests = {}
        bests['val_loss'] = np.inf  # best valid loss itr wise
        bests['val_loss_save'] = np.inf  # best valid loss in saved checkpoints
        bests['iter_'] = 0
        bests['step'] = 0
        bests['step_save'] = 0  # global step for the best saved model
        bests['counter'] = 0
        bests['counter_thresh'] = 10
        bests['last_epoch'] = 0
        bests['epoch_tr_mse'] = []
        bests['epoch_val_mse'] = []
        bests['epoch_tr_cost'] = []
        bests['epoch_val_cost'] = []
        validation_frequency = n_train_batches  # save every epoch basically.
        save_frequency = 1

        # Initialise:
        epoch_init=initialise_model(sess, saver, phase_train, checkpoint_dir, bests, opt)

        # Start training!
        while (epoch < opt['no_epochs']) and (not done_looping):

            start_time_epoch = timeit.default_timer()
            lr_ = opt['learning_rate']

            # gradually reduce learning rate every 50 epochs:
            if (epoch+1) % 50 == 0:
                lr_=lr_/ 10.

            for mi in xrange(n_train_batches):
                # Select minibatches using a slice object---consider
                # multi-threading for speed if this is too slow

                xt, yt = dataset.next_batch(opt['batch_size'])
                xv, yv = dataset.next_val_batch(opt['batch_size'])

                # train op and loss
                current_step = tf.train.global_step(sess, global_step)
                fd_t={x: xt, y: yt, lr: lr_,
                      keep_prob: 1.-opt['dropout_rate'],
                      trade_off:tradeoff_list[epoch],
                      phase_train:True,
                      transform: norm_std}

                __, tr_mse, tr_cost = sess.run([train_step, mse, cost],feed_dict=fd_t)
                total_tr_mse_epoch += tr_mse
                total_tr_cost_epoch += tr_cost

                # valid loss
                fd_v = {x: xv, y: yv,
                        keep_prob: 1.-opt['dropout_rate'],
                        trade_off:tradeoff_list[epoch],
                        phase_train:False,
                        transform: norm_std}

                va_mse, va_cost = sess.run([mse,cost], feed_dict=fd_v)
                total_val_mse_epoch += va_mse
                total_val_cost_epoch += va_cost

                # iteration number
                iter_ = epoch * n_train_batches + mi
                iter_valid += 1

                # Print out current progress
                if (iter_ + 1) % (max(validation_frequency/10,1)) == 0:
                    summary_t = sess.run(merged, feed_dict=fd_t)
                    summary_v = sess.run(merged, feed_dict=fd_v)
                    train_writer.add_summary(summary_t, iter_+1)
                    valid_writer.add_summary(summary_v, iter_+1)

                    vl = np.sqrt(va_mse*10**10)
                    vc = va_cost

                    sys.stdout.flush()
                    sys.stdout.write('\tvalid mse: %.2f,  valid cost: %.3f \r' % (vl,vc))

                if (iter_ + 1) % validation_frequency == 0:
                    # Print out the errors for each epoch:
                    this_val_mse = total_val_mse_epoch/iter_valid
                    this_tr_mse = total_tr_mse_epoch/iter_valid
                    this_val_cost = total_val_cost_epoch/iter_valid
                    this_tr_cost = total_tr_cost_epoch/iter_valid

                    end_time_epoch = timeit.default_timer()

                    bests['epoch_val_mse'].append(this_val_mse)
                    bests['epoch_tr_mse'].append(this_tr_mse)
                    bests['epoch_val_cost'].append(this_val_cost)
                    bests['epoch_tr_cost'].append(this_tr_cost)

                    epoch = dataset.epochs_completed

                    print('\nEpoch %i, minibatch %i/%i:\n' \
                          '\ttraining error (rmse) : %f times 1E-5\n' \
                          '\tvalidation error (rmse) : %f times 1E-5\n' \
                          '\ttraining cost : %.2f \n'\
                          '\tvalidation cost : %.2f \n'\
                          '\ttook %f secs'
                          % (epoch + 1 + epoch_init,
                             mi + 1, n_train_batches,
                             np.sqrt(this_tr_mse*10**10),
                             np.sqrt(this_val_mse*10**10),
                             np.sqrt(this_tr_cost*10**10),
                             np.sqrt(this_val_cost*10**10),
                             end_time_epoch - start_time_epoch))

                    if opt['valid']:
                        this_val_loss = this_val_cost
                    else:
                        this_val_loss = this_val_mse

                    bests = update_best_loss(this_val_loss,
                                             bests,
                                             iter_,
                                             current_step)

                    # Start counting again:
                    total_val_mse_epoch = 0
                    total_tr_mse_epoch = 0
                    total_val_cost_epoch = 0
                    total_tr_cost_epoch = 0

                    iter_valid = 0
                    start_time_epoch = timeit.default_timer()

            if (epoch+1) % save_frequency == 0:
                if opt['valid']:
                    this_val_loss=this_val_cost
                else:
                    this_val_loss=this_val_mse

                if this_val_loss < bests['val_loss_save']:
                    bests['last_epoch'] = epoch + 1
                    bests = update_best_loss_epoch(this_val_loss, bests, current_step)
                    save_model(opt, sess, saver, global_step, bests)

        # close the summary writers:
        train_writer.close()
        valid_writer.close()

        # Display the best results:
        print(('\nOptimization complete. Best validation score of %f  '
               'obtained at iteration %i') %
              (np.sqrt(bests['val_loss']*10**10), bests['step']))

        # Display the best results for saved models:
        print(('\nOptimization complete. Best validation score of %f  '
               'obtained at iteration %i') %
              (np.sqrt(bests['val_loss_save'] * 10 ** 10), bests['step_save']))

        end_time = timeit.default_timer()
        time_train = end_time - start_time
        print('Training done!!! It took %f secs.' % time_train)
