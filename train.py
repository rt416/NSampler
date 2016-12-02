'''Training file'''

import os
import sys
import timeit

import cPickle as pkl
import h5py
import numpy as np
import tensorflow as tf

import preprocess as pp
import sr_utility 
import models


def load_data(opt):
	cohort = opt['cohort']
	no_subjects =opt['no_subjects'] 
	sample_rate = opt['sample_rate'] 
	us = opt['us'] 
	n = opt['n'] // 2
	m = opt['m']
	data_dir = opt['data_dir']

	dataset = data_dir + 'PatchLibs%sDS%02i_%ix%i_%ix%i_TS%i_SRi%03i_0001.mat' \
			% (cohort, us, 2 * n + 1, 2 * n + 1, m, m, no_subjects, sample_rate)
	data_dir, data_file = os.path.split(dataset)

	print('... loading the training dataset %s' % data_file)
	patchlib = sr_utility.load_patchlib(patchlib=dataset)
	# load the original patch libs
	train_x, valid_x, train_y, valid_y = patchlib  
	
	# normalise the data and keep the transforms:
	data = sr_utility.standardise_data(train_x, train_y, option='default')
	data = list(data)
	train_x_scaled = data[0]
	train_x_mean = data[1]
	train_x_std = data[2]
	train_y_scaled = data[3]
	train_y_mean = data[4]
	train_y_std = data[5]
	valid_x_scaled = (valid_x - train_x_mean) / train_x_std
	valid_y_scaled = (valid_y - train_y_mean) / train_y_std
	data.append(valid_x_scaled)
	data.append(valid_y_scaled)
	return data


def define_checkpoint(opt):
	nn_file = sr_utility.name_network(opt)
	checkpoint_dir = os.path.join(opt['save_dir'], nn_file)
	if not os.path.exists(checkpoint_dir):  
		os.makedirs(checkpoint_dir)
	return checkpoint_dir

def update_best_loss(this_loss, bests, iter_, current_step):	
	bests['counter'] += 1
	if this_loss < bests['val_loss']:
		bests['counter'] = 0
		bests['val_loss'] = this_loss
		bests['iter_'] = iter_
		bests['step'] = current_step + 1
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
	# Load opt into namespace
	optimisation_method = opt['optimizer'].__name__
	globals().update(opt)
	train_fraction = int(1.-opt['validation_fraction'])
	
	# Set the directory for saving checkpoints:
	checkpoint_dir = define_checkpoint(opt)
	opt['checkpoint_dir'] = checkpoint_dir
	
	# ---------------------------load data--------------------------:
	data = pp.load_hdf5(opt)
	in_shape = data['in']['train'].shape[1:]
	out_shape = data['out']['train'].shape[1:]
	
	# --------------------------- Define the model--------------------------:
	# define input and output:
	x = tf.placeholder(tf.float32, [None,n,n,n,6], name='lo_res') 
	y = tf.placeholder(tf.float32, [None,1,1,1,out_shape[-1]], name='hi_res') 
	lr = tf.placeholder(tf.float32, [], name='learning_rate')
	keep_prob = tf.placeholder(tf.float32)  # keep probability for dropout
	global_step = tf.Variable(0, name="global_step", trainable=False)
	
	# Build model and loss function
	y_pred = models.inference(method, x, opt)
	cost = tf.reduce_mean(tf.square(y - y_pred))
	
	# Define gradient descent op
	optim = opt['optimizer'](learning_rate=lr)
	train_step = optim.minimize(cost, global_step=global_step)
	mse = tf.reduce_mean(tf.square(data['out']['std'] * (y - y_pred)))
	
	# -------------------------- Start training -----------------------------:
	saver = tf.train.Saver()
	print('... training')
	with tf.Session() as sess:
		# Run the Op to initialize the variables.
		init = tf.initialize_all_variables()
		sess.run(init)
	
		# Compute number of minibatches for training, validation and testing
		n_train_batches = data['in']['train'].shape[0] // batch_size
		n_valid_batches = data['in']['valid'].shape[0] // batch_size
		
		# Define some counters
		test_score = 0
		start_time = timeit.default_timer()
		epoch = 0
		done_looping = False
		iter_valid = 0
		total_val_loss_epoch = 0
		total_tr_loss_epoch = 0
		lr_ = opt['learning_rate']
		
		bests = {}
		bests['val_loss'] = np.inf
		bests['iter_'] = 0
		bests['step'] = 0
		bests['counter'] = 0
		bests['counter_thresh'] = 10
		validation_frequency = n_train_batches
		save_frequency = 10
		
		model_details = opt.copy()
		model_details.update(bests)
		
		while (epoch < n_epochs) and (not done_looping):
			epoch += 1
			start_time_epoch = timeit.default_timer()
			if epoch % 50 == 0:
				lr_ = lr_ / 10.
			for mi in xrange(n_train_batches):
				# Select minibatches using a slice object
				idx = np.s_[mi*batch_size:(mi+1)*batch_size,...]
				xt = pp.dict_whiten(data, 'in', 'train', idx)
				yt = pp.dict_whiten(data, 'out', 'train', idx)
				xv = pp.dict_whiten(data, 'in', 'valid', idx)
				yv = pp.dict_whiten(data, 'out', 'valid', idx)
				current_step = tf.train.global_step(sess, global_step)
				
				# train op and loss
				fd={x: xt, y: yt, lr: lr_, keep_prob: 1.-dropout_rate}
				__, tr_loss = sess.run([train_step, mse], feed_dict=fd)
				total_tr_loss_epoch += tr_loss
				# valid loss
				fd = {x: xv, y: yv, keep_prob: 1.-dropout_rate}
				va_loss = sess.run(mse, feed_dict=fd)
				total_val_loss_epoch += va_loss
	
				# iteration number
				iter_ = (epoch - 1) * n_train_batches + mi
				iter_valid += 1
	
				if (iter_ + 1) % validation_frequency == 0:
					# Print out the errors for each epoch:
					this_val_loss = total_val_loss_epoch/iter_valid
					this_tr_loss = total_tr_loss_epoch/iter_valid
					end_time_epoch = timeit.default_timer()
					
					print('\nEpoch %i, minibatch %i/%i:\n' \
						  '\ttraining error (rmse) %f times 1E-5\n' \
						  '\tvalidation error (rmse) %f times 1E-5\n' \
						  '\ttook %f secs' % (epoch, mi + 1, n_train_batches,
							np.sqrt(this_tr_loss*10**10), 
							np.sqrt(this_val_loss*10**10), 
							end_time_epoch - start_time_epoch))
					bests = update_best_loss(this_val_loss, bests, iter_, current_step)
	
					# Start counting again:
					total_val_loss_epoch = 0
					total_tr_loss_epoch = 0
					iter_valid = 0
					start_time_epoch = timeit.default_timer()
					
			if epoch % save_frequency == 0:
				model_details.update(bests)
				save_model(opt, sess, saver, global_step, model_details)
			
		# Display the best results:
		print(('\nOptimization complete. Best validation score of %f  '
			   'obtained at iteration %i') %
			  (np.sqrt(bests['val_loss']*10**10), bests['step']))
	
		end_time = timeit.default_timer()
		time_train = end_time - start_time
		print('Training done!!! It took %f secs.' % time_train)
	
