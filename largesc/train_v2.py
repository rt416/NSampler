"""Training file (Ryu) """

import os
import sys
import timeit

import cPickle as pkl
import h5py
import numpy as np
import tensorflow as tf
import largesc.data_generator as data_generator
import sr_preprocess as pp
import sr_utility
import models


def define_checkpoint(opt):
    nn_file = name_network(opt)
    checkpoint_dir = os.path.join(opt['save_dir'], nn_file)
    if not os.path.exists(checkpoint_dir):
        print(checkpoint_dir)
        os.makedirs(checkpoint_dir)
    return checkpoint_dir


def define_logdir(opt):
    nn_file = name_network(opt)
    checkpoint_dir = os.path.join(opt['log_dir'], nn_file)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    return checkpoint_dir


def name_network(opt):
    """given inputs, return the model name."""
    optim = opt['optimizer'].__name__

    nn_header = opt['method'] if opt['dropout_rate']==0 \
    else opt['method'] + str(opt['dropout_rate'])

    # problem definition:
    nn_var = (opt['upsampling_rate'],
              2*opt['input_radius']+1,
              2*opt['receptive_field_radius']+1,
             (2*opt['output_radius']+1)*opt['upsampling_rate'])
    nn_str = 'us=%i_in=%i_rec=%i_out=%i_'

    nn_var += (opt['no_subjects'],
               opt['train_size'],
               opt['transform_opt'],
               opt['patch_sampling_opt'],
               opt['patchlib_idx'])
    nn_str += 'ts=%d_pl=%d_nrm=%s_smpl=%s_%03i'

    # nn_var += (optim, str(opt['dropout_rate']), opt['transform_opt'])
    # nn_str +='opt=%s_drop=%s_prep=%s_'

    nn_body = nn_str % nn_var

    if opt['valid']:
        # Validate on the cost:
        nn_header += '_valid_cost'

    return nn_header + '_' + nn_body

def name_patchlib(opt):
    """given inputs, return the patchlib name """

    header = 'patchlib_'

    # problem definition:
    nn_var = (opt['upsampling_rate'],
              2 * opt['input_radius'] + 1,
              2 * opt['receptive_field_radius'] + 1,
              (2 * opt['output_radius'] + 1) * opt['upsampling_rate'])
    nn_str = 'us=%i_in=%i_rec=%i_out=%i_'

    nn_var += (opt['no_subjects'],
               opt['train_size'],
               opt['transform_opt'],
               opt['patch_sampling_opt'],
               opt['patchlib_idx'])
    nn_str += 'ts=%d_pl=%d_nrm=%s_smpl=%s_%03i'

    # nn_var += (optim, str(opt['dropout_rate']), opt['transform_opt'])
    # nn_str +='opt=%s_drop=%s_prep=%s_'

    nn_body = nn_str % nn_var
    return header+nn_body


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


def save_model(opt, sess, saver, global_step, model_details):
    checkpoint_dir = opt['checkpoint_dir']
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    save_path = saver.save(sess, checkpoint_prefix, global_step=global_step)
    print("Model saved in file: %s" % save_path)
    with open(os.path.join(checkpoint_dir, 'settings.pkl'), 'wb') as fp:
        pkl.dump(model_details, fp, protocol=pkl.HIGHEST_PROTOCOL)
    print('Model details saved')


def train_cnn(opt):
    # ------------------ Load inputs from op ------------------------:
    # Network details:
    dropout_rate = opt['dropout_rate']
    L1_reg = opt['L1_reg']
    L2_reg = opt['L2_reg']
    n_epochs = opt['n_epochs']
    batch_size = opt['batch_size']
    validation_fraction = opt['validation_fraction']
    train_fraction = int(1.0 - validation_fraction)
    shuffle = opt['shuffle']

    method = opt['method']
    n_h1 = opt['n_h1']
    n_h2 = opt['n_h2']
    n_h3 = opt['n_h3']
    cohort = opt['cohort']

    # Input/Output details:
    upsampling_rate = opt['upsampling_rate']
    no_channels = opt['no_channels']
    input_radius = opt['input_radius']
    receptive_field_radius = opt['receptive_field_radius']
    output_radius = ((2*input_radius - 2*receptive_field_radius + 1) // 2)
    opt['output_radius'] = output_radius
    transform_opt = opt['transform_opt']


    # Set the directory for saving checkpoints:
    checkpoint_dir = define_checkpoint(opt)
    log_dir = define_logdir(opt)
    opt["checkpoint_dir"] = checkpoint_dir

    # exit if the network has already been trained:
    if os.path.exists(os.path.join(checkpoint_dir, 'settings.pkl')) \
       and not(opt['continue']) and not(opt['overwrite']):
        print('Network already trained. Move on to next.')
        return

    # -------------------------load data---------------------------------------:
    filename_patchlib = name_patchlib(opt)
    dataset, train_folder = data_generator.prepare_data(opt['train_size'],
                                                        opt['validation_fraction'],
                                                        input_radius,
                                                        output_radius,
                                                        patchlib_name=filename_patchlib,
                                                        whiten=opt['transform_opt'],
                                                        train_index=opt['train_subjects'],
                                                        bgval=opt['background_value'],
                                                        is_reset=False,
                                                        us_rate=opt['upsampling_rate'],
                                                        data_dir_root=opt['gt_dir'],
                                                        save_dir_root=opt['data_dir'],
                                                        subpath=opt['subpath'],
                                                        )

    opt['train_noexamples'] = dataset.size
    opt['valid_noexamples'] = dataset.size_valid
    print ('\nPatch-lib size:', opt['train_noexamples']+opt['valid_noexamples'],
           'Train size:', opt['train_noexamples'],
           'Valid size:', opt['valid_noexamples'])

    # --------------------------- Define the model--------------------------:
    #  define input and output:
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None,
                                        2*input_radius+1,
                                        2*input_radius+1,
                                        2*input_radius+1,
                                        no_channels],
                                        name='lo_res')

        y = tf.placeholder(tf.float32, [None,
                                        2*output_radius+1,
                                        2*output_radius+1,
                                        2*output_radius+1,
                                        no_channels*(upsampling_rate**3)],
                                        name='hi_res')

    with tf.name_scope('learning_rate'):
        lr = tf.placeholder(tf.float32, [], name='learning_rate')

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)  # keep probability for dropout

    with tf.name_scope('tradeoff'):
        trade_off = tf.placeholder(tf.float32)  # keep probability for dropout

    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Build model and loss function
    y_pred, y_std, cost = models.inference(method, x, y, keep_prob, opt, trade_off)

    # Define gradient descent op
    with tf.name_scope('train'):
        optim = opt['optimizer'](learning_rate=lr)
        train_step = optim.minimize(cost, global_step=global_step)

    with tf.name_scope('accuracy'):
        # todo: introduce proper scaling
        transform = dataset._transform
        print(transform)
        mse = tf.reduce_mean(tf.square(transform['output_std']*(y - y_pred)))
        # mse = tf.reduce_mean(tf.square(y - y_pred))
        tf.summary.scalar('mse', mse)

    # -------------------------- Start training -----------------------------:
    saver = tf.train.Saver()
    print('\nStart training!\n')
    with tf.Session() as sess:

        # Merge all the summaries and write them out to ../network_name/log
        merged = tf.summary.merge_all()
        train_writer = tf.train.SummaryWriter(log_dir + '/train', sess.graph)
        valid_writer = tf.train.SummaryWriter(log_dir + '/valid')

        # Compute number of minibatches for training, validation and testing
        n_train_batches = opt['train_noexamples'] // opt['batch_size']
        n_valid_batches = opt['valid_noexamples'] // opt['batch_size']

        # Compute the trade-off values:
        tradeoff_list = models.get_tradeoff_values(opt)

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
        validation_frequency = n_train_batches  # save every epoch basically.
        save_frequency = 1

        model_details = opt.copy()
        model_details.update(bests)
        model_details['last_epoch'] = 0
        model_details['epoch_tr_mse'] = []
        model_details['epoch_val_mse'] = []
        model_details['epoch_tr_cost'] = []
        model_details['epoch_val_cost'] = []

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

                # Restore the previous model parameters:
                saver.restore(sess, nn_file)
            else:
                print('no trace of previous training!')
                print('intialise and start training from scratch.')
                #init = tf.initialize_all_variables()
                init = tf.global_variables_initializer()
                epoch_init = 0
                sess.run(init)
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

            #init = tf.initialize_all_variables()
            init = tf.global_variables_initializer()
            epoch_init = 0
            sess.run(init)

        # Start training!
        while (epoch < n_epochs) and (not done_looping):

            start_time_epoch = timeit.default_timer()
            lr_ = opt['learning_rate']

            # gradually reduce learning rate every 50 epochs:
            # if (epoch+1) % 50 == 0:
            #     lr_ = lr_ / 10.

            for mi in xrange(n_train_batches):
                # Select minibatches using a slice object---consider
                # multi-threading for speed if this is too slow

                xt, yt = dataset.next_batch(opt['batch_size'])
                xv, yv = dataset.next_val_batch(opt['batch_size'])

                # xt = pp.dict_whiten(data, 'in', 'train', idx)
                # yt = pp.dict_whiten(data, 'out', 'train', idx)
                # xv = pp.dict_whiten(data, 'in', 'valid', idx)
                # yv = pp.dict_whiten(data, 'out', 'valid', idx)
                current_step = tf.train.global_step(sess, global_step)

                # train op and loss
                fd_t={x: xt, y: yt, lr: lr_,
                      keep_prob: 1.-dropout_rate, trade_off:tradeoff_list[epoch]}

                __, tr_mse, tr_cost = sess.run([train_step, mse, cost],feed_dict=fd_t)
                total_tr_mse_epoch += tr_mse
                total_tr_cost_epoch += tr_cost

                # valid loss
                fd_v = {x: xv, y: yv,
                        keep_prob: 1.-dropout_rate, trade_off:tradeoff_list[epoch]}
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

                    model_details['epoch_val_mse'].append(this_val_mse)
                    model_details['epoch_tr_mse'].append(this_tr_mse)

                    model_details['epoch_val_cost'].append(this_val_cost)
                    model_details['epoch_tr_cost'].append(this_tr_cost)

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
                    model_details['last_epoch'] = epoch + 1
                    bests = update_best_loss_epoch(this_val_loss, bests, current_step)
                    model_details.update(bests)
                    save_model(opt, sess, saver, global_step, model_details)

            # Update iteration counters:
            # epoch_auro = dataset.epochs_completed
            # epoch += 1
            # print('epoch=%d \n'
            #       'epoch_auro=%d' % (epoch, epoch_auro))

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
