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
	train_x = np.reshape(train_x, (-1,5,5,5,6), order='F')
	valid_x = np.reshape(valid_x, (-1,5,5,5,6), order='F')
	train_y = np.reshape(train_y, (-1,1,1,1,6*8), order='F')
	valid_y = np.reshape(valid_y, (-1,1,1,1,6*8), order='F')
	
	data = {}
	data['in'] = {}
	data['out'] = {}
	data['in']['train'] = train_x
	data['in']['valid'] = valid_x
	data['in']['mean'], data['in']['std'] = pp.moments(opt, train_x)
	
	data['out']['train'] = train_y
	data['out']['valid'] = valid_y
	data['out']['mean'], data['out']['std']  = pp.moments(opt, train_y)
	
	return data

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
	#data = load_data(opt)
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
			if shuffle:
				indices = np.random.permutation(data['in']['train'].shape[0])
			else:
				indices = np.arange(data['in']['train'].shape[0])
			for mi in xrange(n_train_batches):
				# Select minibatches using a slice object---consider
				# multi-threading for speed if this is too slow
				idx = np.s_[indices[mi*batch_size:(mi+1)*batch_size],...]
				
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
				
				# Print out current progress
				if (iter_ + 1) % (validation_frequency/100) == 0:
					vl = np.sqrt(va_loss*10**10)
					sys.stdout.flush()
					sys.stdout.write('\tvalidation error: %.2f\r' % (vl,))
	
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
					bests = update_best_loss(this_val_loss, bests, iter_,
											 current_step)
	
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
	
# Reconstruct using the specified NN:
def super_resolve(opt):
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
def reconstruct(opt):
	"""Run the network on the test data"""
	# Load the input low-res DT image:
	print('... loading the test low-res image ...')
	dt_lowres = sr_utility.read_dt_volume(nameroot=os.path.join(gt_dir, 'dt_b1000_lowres_2_'))
	
	#nn_file = sr_utility.name_network(opt)
	#network_dir = os.path.join(model_dir, nn_file) 
	#print('\nReconstruct high-res dti with the network: \n%s.' % network_dir)
	dt_hr = super_resolve(dt_lowres, opt)
	
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