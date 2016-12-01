'''Load and preprocessing script'''

import os
import sys

import cPickle as pkl
import h5py
import numpy as np
import tensorflow as tf

def load_hdf5(opt):
	cohort = opt['cohort']
	no_subjects =opt['no_subjects'] 
	sample_rate = opt['sample_rate'] 
	us = opt['us'] 
	n = opt['n'] // 2
	m = opt['m']
	data_dir = opt['data_dir']
	fstr = 'PatchLibs_%s_Upsample%02i_Input%02i_Recep%02i_TS%i_SRi%03i_001.h5'
	filename = data_dir + fstr \
					% (cohort, us, 2*n+1, 2*n+1, no_subjects, sample_rate)
	
	# {'in': {'X': <raw_data>, 'mean': <mean>, 'std': <std>}, 'out' : {...}}
	f = h5py.File(filename, 'r')
	data = {}
	
	print("Loading %s" % (filename,))
	for i in ['in','out']:
		print("\tLoading input and stats")
		data[i] = {}
		data[i]['X'] = f[i+"put_lib"]
		data[i]['mean'], data[i]['std'] = rescale(opt, data[i]['X'])

	# Save the transforms used for data normalisation:
	print('\tSaving transforms for data normalization for test time')
	transform = {'input_mean': data['in']['mean'],
				 'input_std': data['in']['std'],
				 'output_mean': data['out']['mean'],
				 'output_std': data['out']['std']}
	with open(os.path.join(opt['checkpoint_dir'], 'transforms.pkl'), 'w') as fp:
		pkl.dump(transform, fp, protocol=pkl.HIGHEST_PROTOCOL)
	return data
	

def rescale(opt, x):
	"""Per-element whitenin on the training set"""
	xsh = x.shape[0]
	num_train = int((1.-opt['validation_fraction'])*xsh)
	
	mean = np.mean(x[:num_train,...], axis=0, keepdims=True)
	std = np.std(x[num_train:,...], axis=0, keepdims=True)
	return mean, std