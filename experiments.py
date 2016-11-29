"""Main experiments script"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from train import train_cnn

# Options
opt = {}
opt['optimizer'] = tf.train.AdamOptimizer
opt['dropout_rate'] = 0.0
opt['learning_rate'] = 1e-3
opt['L1_reg'] = 0.00
opt['L2_reg'] = 1e-5
opt['n_epochs'] = 1000
opt['batch_size'] = 25

opt['method'] = 'cnn_simple'
opt['n_h1'] = 50
opt['n_h2'] = 20
opt['n_h3'] = 10
opt['cohort'] ='Diverse'
opt['no_subjects'] = 8
opt['sample_rate'] = 4
opt['us'] = 2
opt['n'] = 5
opt['m'] = 2

opt['data_dir'] ='../data/'
opt['save_dir'] = '../models'

train_cnn(opt)


