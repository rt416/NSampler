"""Main experiments script"""

import numpy as np
import matplotlib.pyplot as plt

from train import train_cnn

# ------------- Drop-out experiments:

from sr_nn import *

# Model:
drop_list = [0.75, 0.5, 0.25]
methods_list = ['mlp_h=3']
n_h1 = 500
n_h2 = 200
n_h3 = 100

# Training data details:
sr_list =[32, 16, 8, 4]
us, n, m = 2, 2, 2

# Options
opt = {}
opt['optimisation_method'] = 'adam'
opt['dropout_rate'] = 0.0
opt['learning_rate'] = 1e-4
opt['L1_reg'] = 0.00
opt['L2_reg'] = 1e-5
opt['n_epochs'] = 1000
opt['batch_size'] = 25

opt['method'] = 'linear'
opt['n_h1'] = 500
opt['n_h2'] = 200
opt['n_h3'] = 100
opt['cohort'] ='Diverse'
opt['no_subjects'] = 8
opt['sample_rate'] = 32
opt['us'] = 2
opt['n'] = 2
opt['m'] = 2

opt['data_dir'] ='../data/'
opt['save_dir'] = '../models'

train_cnn(opt)


