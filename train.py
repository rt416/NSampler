'''Training file'''

import os
import sys
import timeit

import cPickle
import numpy as np
import tensorflow as tf

import sr_utility 
import models


def load_data(opt):
	cohort = opt['cohort']
	no_subjects =opt['no_subjects'] 
	sample_rate = opt['sample_rate'] 
	us = opt['us'] 
	n = opt['n']
	m = opt['m']
	data_dir = opt['data_dir']

	dataset = data_dir + 'PatchLibs%sDS%02i_%ix%i_%ix%i_TS%i_SRi%03i_0001.mat' \
			% (cohort, us, 2 * n + 1, 2 * n + 1, m, m, no_subjects, sample_rate)
	data_dir, data_file = os.path.split(dataset)

	print('... loading the training dataset %s' % data_file)
	patchlib = sr_utility.load_patchlib(patchlib=dataset)
	# load the original patch libs
	train_set_x, valid_set_x, train_set_y, valid_set_y = patchlib  
	
	# normalise the data and keep the transforms:
	data = sr_utility.standardise_data(train_set_x, train_set_y, option='default')
	data = list(data)
	train_set_x_scaled = data[0]
	train_set_x_mean = data[1]
	train_set_x_std = data[2]
	train_set_y_scaled = data[3]
	train_set_y_mean = data[4]
	train_set_y_std = data[5]
	
	# normalise the validation sets into the same space as training sets:
	valid_set_x_scaled = (valid_set_x - train_set_x_mean) / train_set_x_std
	valid_set_y_scaled = (valid_set_y - train_set_y_mean) / train_set_y_std
	data.append(valid_set_x_scaled)
	data.append(valid_set_y_scaled)
	
	return data

def train_cnn(opt):
	# Load opt
	optimisation_method = opt['optimisation_method'] 
	dropout_rate = opt['dropout_rate'] 
	learning_rate = opt['learning_rate']
	L1_reg =opt['L1_reg']
	L2_reg = opt['L2_reg'] 
	n_epochs = opt['n_epochs'] 
	batch_size = opt['batch_size'] 
	method = opt['method']
	n_h1 = opt['n_h1']
	n_h2 = opt['n_h2']
	n_h3 = opt['n_h3'] 
	cohort = opt['cohort']
	no_subjects =opt['no_subjects'] 
	sample_rate = opt['sample_rate'] 
	us = opt['us'] 
	n = opt['n']
	m = opt['m']
	data_dir = opt['data_dir']
	save_dir = opt['save_dir']
	
	# ---------------------------load data--------------------------:
	data = load_data(opt)
	train_set_x_scaled = data[0]
	train_set_x_mean = data[1]
	train_set_x_std = data[2]
	train_set_y_scaled = data[3]
	train_set_y_mean = data[4]
	train_set_y_std = data[5]
	valid_set_x_scaled = data[6]
	valid_set_y_scaled = data[7]
	
	# --------------------------- Define the model--------------------------:
	# define input and output:
	n_in = 6 * (2 * n + 1) ** 3
	n_out = 6 * m ** 3  
	x_scaled = tf.placeholder(tf.float32, shape=[None, n_in]) # low res
	y_scaled = tf.placeholder(tf.float32, shape=[None, n_out])  # high res
	keep_prob = tf.placeholder(tf.float32)  # keep probability for dropout
	global_step = tf.Variable(0, name="global_step", trainable=False)
	
	y_pred_scaled, L2_sqr, L1 = models.inference(method, x_scaled, keep_prob, n_in, n_out,
												 n_h1=n_h1, n_h2=n_h2, n_h3=n_h3)
	cost = models.cost(y_scaled, y_pred_scaled, L2_sqr, L1, L2_reg, L1_reg)
	train_step = models.training(cost, learning_rate, global_step=global_step, option=optimisation_method)
	mse = tf.reduce_mean(tf.square(train_set_y_std * (y_scaled - y_pred_scaled)))
	
	# -------------------------- Start training -----------------------------:
	saver = tf.train.Saver()
	# Set the directory for saving checkpoints:
	nn_file = sr_utility.name_network(method=method, n_h1=n_h1, n_h2=n_h2, n_h3=n_h3, cohort=cohort, no_subjects=no_subjects,
									  sample_rate=sample_rate, us=us, n=n, m=m,
									  optimisation_method=optimisation_method, dropout_rate=dropout_rate)
	
	checkpoint_dir = os.path.join(save_dir, nn_file)
	
	if not os.path.exists(checkpoint_dir):  # create a subdirectory to save the model.
		os.makedirs(checkpoint_dir)
	
	# Save the transforms used for data normalisation:
	print('... saving the transforms used for data normalisation for the test time')
	transform = {'input_mean': train_set_x_mean, 'input_std': train_set_x_std,
				 'output_mean': train_set_y_mean, 'output_std': train_set_y_std}
	f = file(os.path.join(checkpoint_dir, 'transforms.pkl'), 'wb')
	cPickle.dump(transform, f, protocol=cPickle.HIGHEST_PROTOCOL)
	
	
	# Create a session for running Ops on the Graph.
	print('... training')
	
	with tf.Session() as sess:
		# Run the Op to initialize the variables.
		init = tf.initialize_all_variables()
		sess.run(init)
	
		# Compute number of minibatches for training, validation and testing
		n_train_batches = train_set_x_scaled.shape[0] // batch_size
		n_valid_batches = valid_set_x_scaled.shape[0] // batch_size
	
		# early-stopping parameters
		patience = 10000  # look as this many examples regardless
		patience_increase = 2  # wait this much longer when a new best is found
		improvement_threshold = 0.995  # a relative improvement of this much is considered significant
		validation_frequency = min(n_train_batches, patience // 2)
		# go through this many minibatches before checking the network on the validation set;
		# in this case we check every epoch
	
		best_validation_loss = np.inf
		best_iter = 0
		test_score = 0
	
		start_time = timeit.default_timer()
	
		epoch = 0
		done_looping = False
	
		iter_valid = 0
		total_validation_loss_epoch = 0
		total_training_loss_epoch = 0
	
		while (epoch < n_epochs) and (not done_looping):
			epoch += 1
			start_time_epoch = timeit.default_timer()
	
			for minibatch_index in range(n_train_batches):
	
				# Select batches:
				x_batch_train = train_set_x_scaled[minibatch_index * batch_size:(minibatch_index + 1) * batch_size, :]
				y_batch_train = train_set_y_scaled[minibatch_index * batch_size:(minibatch_index + 1) * batch_size, :]
				x_batch_valid = valid_set_x_scaled[minibatch_index * batch_size:(minibatch_index + 1) * batch_size, :]
				y_batch_valid = valid_set_y_scaled[minibatch_index * batch_size:(minibatch_index + 1) * batch_size, :]
	
				# track the number of steps
				current_step = tf.train.global_step(sess, global_step)
	
				# perform gradient descent:
				train_step.run(
					feed_dict={x_scaled: x_batch_train,
							   y_scaled: y_batch_train,
							   keep_prob: (1.0 - dropout_rate)})
	
				# Accumulate validation/training errors for each epoch:
				total_validation_loss_epoch += mse.eval(
					feed_dict={x_scaled: x_batch_valid,
							   y_scaled: y_batch_valid,
							   keep_prob: (1.0 - dropout_rate)})
	
				total_training_loss_epoch += mse.eval(
					feed_dict={x_scaled: x_batch_train,
							   y_scaled: y_batch_train,
							   keep_prob: (1.0 - dropout_rate)})
	
				# iteration number
				iter = (epoch - 1) * n_train_batches + minibatch_index
				iter_valid += 1
	
				if (iter + 1) % validation_frequency == 0:
					# Print out the errors for each epoch:
	
					this_validation_loss = total_validation_loss_epoch/iter_valid
					this_training_loss = total_training_loss_epoch/iter_valid
					end_time_epoch = timeit.default_timer()
	
					print(
						'\nEpoch %i, minibatch %i/%i:\n'
						'     training error (rmse) %f times 1E-5\n'
						'     validation error (rmse) %f times 1E-5\n'
						'     took %f secs' %
						(
							epoch,
							minibatch_index + 1,
							n_train_batches,
							np.sqrt(this_training_loss * 10 ** 10),
							np.sqrt(this_validation_loss * 10 ** 10),
							end_time_epoch - start_time_epoch
						)
					)
					print('     number of minibatches = %i and patience = %i' % (iter + 1, patience))
					print('     validation frequency = %i, iter_valid = %i' % (validation_frequency, iter_valid))
					# if we got the best validation score until now
					if this_validation_loss < best_validation_loss:
	
						# improve patience if loss improvement is good enough
						if this_validation_loss < best_validation_loss *improvement_threshold :
	
							patience = max(patience, iter * patience_increase)
							print('     reduces the previous error by more than %f %%'
								  % ((1 - improvement_threshold) * 100.))
	
						best_validation_loss = this_validation_loss
						best_training_loss = this_training_loss
						best_iter = iter
						best_step = current_step + 1
	
					# Save the model:
					checkpoint_prefix = os.path.join(checkpoint_dir, "model")
					save_path = saver.save(sess, checkpoint_prefix, global_step=global_step)
					print("Model saved in file: %s" % save_path)
	
					# Save the model details:
					print('... saving the model details')
					model_details = {'method': method, 'cohort': cohort,
									 'no of subjects': no_subjects, 'sample rate': sample_rate, 'upsampling factor': us,
									 'n': n,
									 'm': m, 'optimisation': optimisation_method, 'dropout rate': dropout_rate,
									 'learning rate': learning_rate,
									 'L1 coefficient': L1_reg, 'L2 coefficient': L2_reg, 'max no of epochs': n_epochs,
									 'batch size': batch_size, 'training length': end_time_epoch - start_time,
									 'best validation rmse': np.sqrt(best_validation_loss),
									 'best training rmse': np.sqrt(best_training_loss),
									 'best step': best_step}
					cPickle.dump(model_details, file(os.path.join(checkpoint_dir, 'settings.pkl'), 'wb'),
								 protocol=cPickle.HIGHEST_PROTOCOL)
	
					# Terminate training when the validation loss starts decreasing.
					if this_validation_loss > best_validation_loss:
						patience = 0
						print('Validation error increases - terminate training ...')
						break
	
					# Start counting again:
					total_validation_loss_epoch = 0
					total_training_loss_epoch = 0
					iter_valid = 0
					start_time_epoch = timeit.default_timer()
	
	
				if patience <= iter:
					done_looping = True
					break
	
		# Display the best results:
		print(('\nOptimization complete. Best validation score of %f  '
			   'obtained at iteration %i') %
			  (np.sqrt(best_validation_loss * 10**10), best_step))
	
		end_time = timeit.default_timer()
		time_train = end_time - start_time
		print('Training done!!! It took %f secs.' % time_train)
	
