import numpy as np
import matplotlib.pyplot as plt

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

# Training method details:
optimisation_method = 'adam'
dropout_rate = 0.0
learning_rate = 1e-4
L1_reg = 0.00
L2_reg = 1e-5
n_epochs = 1000
batch_size = 25

for dropout_rate in drop_list:
    for method in methods_list:
        for sample_rate in sr_list:
            # train model:
            sr_train(method=method, n_h1=n_h1, n_h2=n_h2, n_h3=n_h3,
                     cohort='Diverse', sample_rate=sample_rate, us=us, n=n, m=m,
                     optimisation_method=optimisation_method, dropout_rate=dropout_rate, learning_rate=learning_rate,
                     L1_reg=L1_reg, L2_reg=L2_reg,
                     n_epochs=n_epochs, batch_size=batch_size)

            # clear the graph:
            tf.reset_default_graph()


# --------------------- Reconstruct for differnet dropout rates:
drop_list = [0.75, 0.5, 0.25, 0.0]
methods_list = ['mlp_h=3']
n_h1 = 500
n_h2 = 200
n_h3 = 100

# Training data details:
sr_list =[32]
us, n, m = 2, 2, 2

# Training method details:
optimisation_method = 'adam'
dropout_rate = 0.0
learning_rate = 1e-4
L1_reg = 0.00
L2_reg = 1e-5
n_epochs = 1000
batch_size = 25


for dropout_rate in drop_list:
    for method in methods_list:
        for sample_rate in sr_list:
            # train model:
            sr_reconstruct(method=method, n_h1=n_h1, n_h2=n_h2, n_h3=n_h3, us=us, n=n, m=m,
                           optimisation_method=optimisation_method, dropout_rate=dropout_rate,
                           cohort='Diverse', no_subjects=8,
                           sample_rate=sample_rate,
                           model_dir='/Users/ryutarotanno/DeepLearning/nsampler/models',
                           recon_dir='/Users/ryutarotanno/DeepLearning/nsampler/recon',
                           gt_dir='/Users/ryutarotanno/DeepLearning/Test_1/data')

            # clear the graph:
            tf.reset_default_graph()





# ----------------- Plot the errors:
error_classic = np.array([])
error_dl = np.array([[1.18814500e-04, 7.95399921e-05],
                    [7.33397994e-05, 7.34538416e-05],
                    [7.15772966e-05, 7.06104271e-05],
                    [7.08437470e-05, 6.81309372e-05]])
