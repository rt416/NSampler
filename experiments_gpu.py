# Set temporarily the git dir on python path. In future, remove this and add the dir to search path.
from sr_nn import *
import sr_bayes

# --------------- Perform training ---------------------------------
from sr_nn import *
base_dir = '/home/rtanno/Codes/SuperRes'
base_data_dir = '/home/rtanno/Shared/HDD/SuperRes'

# Model:
methods_list = ['mlp_h=1', 'mlp_h=2']
n_h1 = 500
n_h2 = 200
save_dir = os.path.join(base_dir, 'models')

# Training data details:
data_dir = os.path.join(base_data_dir, 'Training/')
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

for method in methods_list:
    for sample_rate in sr_list:
        # train model:
        sr_train(method=method, n_h1=n_h1, n_h2=n_h2,
                 data_dir=data_dir,
                 cohort='Diverse', sample_rate=sample_rate, us=us, n=n, m=m,
                 optimisation_method=optimisation_method, dropout_rate=dropout_rate, learning_rate=learning_rate,
                 L1_reg=L1_reg, L2_reg=L2_reg,
                 n_epochs=n_epochs, batch_size=batch_size,
                 save_dir=save_dir)

        # clear the graph:
        tf.reset_default_graph()


# ---------------- Try variational drop-out ----------------------------------
base_dir = '/home/rtanno/Codes/SuperRes'
base_data_dir = '/home/rtanno/Shared/HDD/SuperRes'

# Model:
methods_list = ['mlp_h=1_kingma']
n_h1=500
n_h2=200
save_dir = os.path.join(base_dir, 'models')

# Training data details:
data_dir = os.path.join(base_data_dir, 'Training/')
sr_list=[32]
us,n,m= 2,2,2

# Training method details:
optimisation_method = 'adam'
dropout_rate = 0.5
learning_rate = 1e-4
L1_reg = 0.00
L2_reg = 1e-5
n_epochs = 1000
batch_size = 25

for method in methods_list:
    for sample_rate in sr_list:
        # train model:
        sr_bayes.sr_train(method=method, n_h1=n_h1, n_h2=n_h2,
                          data_dir=data_dir,
                          cohort='Diverse', sample_rate=sample_rate, us=us, n=n, m=m,
                          optimisation_method=optimisation_method, dropout_rate=dropout_rate,
                          learning_rate=learning_rate, L1_reg=L1_reg, L2_reg=L2_reg,
                          n_epochs=n_epochs, batch_size=batch_size,
                          save_dir=save_dir)

        # clear the graph:
        tf.reset_default_graph()


# ---------------- Perform reconstruction on 8 test subjects -----------------
base_dir = '/home/rtanno/Codes/SuperRes'
base_data_dir = '/home/rtanno/Shared/HDD/SuperRes'
methods_list = ['linear', 'mlp_h=1', 'mlp_h=2']
sr_list = [32, 16]
subjects_list = ['904044', '165840', '889579', '713239', '899885', '117324', '214423', '857263']

for subject in subjects_list:
    gt_dir = base_data_dir + '/HCP/' + subject + '/T1w/Diffusion'
    recon_dir = gt_dir + '/DL'
    model_dir = base_dir + '/models'

    if not os.path.exists(recon_dir):  # create a subdirectory to save the model.
            os.makedirs(recon_dir)

    for sample_rate in sr_list:
        for method in methods_list:
            print('Subject %s, sample rate =%i, method = %s' % (subject, sample_rate, method))
            tf.reset_default_graph()
            sr_reconstruct(method=method, optimisation_method='adam', sample_rate=sample_rate,
                           model_dir=model_dir, gt_dir=gt_dir, recon_dir=recon_dir)

# Compute the average errors:
error_array = np.zeros((len(sr_list), len(methods_list)))
for subject in subjects_list:
    gt_dir = base_data_dir + '/HCP/' + subject + '/T1w/Diffusion'
    recon_dir = gt_dir + '/DL'
    for idx1, sr_rate in enumerate(sr_list):
        for idx2, method in enumerate(methods_list):
            recon_file = 'dt_'\
            + sr_utility.name_network(method=method, optimisation_method='adam', sample_rate=sr_rate) + '.npy'
            error_array[idx1,idx2] += sr_utility.compute_rmse(recon_file=recon_file, recon_dir=recon_dir, gt_dir=gt_dir)

error_array /= len(subjects_list)  # compute the average rmse across test subjects
