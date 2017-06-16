""" Reconstruct the average MAP based weighted by the uncertainty."""

import nibabel as nib
import numpy as np
import sr_utility

recon_dir = '/SAN/vision/hcp/Ryu/non-HCP/Prisma/Diffusion_2.5mm/MAP'
# nn_dir_header = '/cnn_heteroscedastic_variational_channelwise_hybrid_control_us=2_in=11_rec=5_out=14_opt=AdamOptimizer_drop=0.0_prep=standard_Diverse_TS8_Subsample343_'
# save_dir = '/SAN/vision/hcp/Ryu/non-HCP/Prisma/Diffusion_2.5mm/MAP/cnn_heteroscedastic_variational_channelwise_hybrid_control_weight_mean'
# weight=True

nn_dir_header = '/cnn_heteroscedastic_variational_channelwise_hybrid_control_us=2_in=11_rec=5_out=14_opt=AdamOptimizer_drop=0.0_prep=standard_Diverse_TS8_Subsample343_'
save_dir = '/SAN/vision/hcp/Ryu/non-HCP/Prisma/Diffusion_2.5mm/MAP/cnn_heteroscedastic_variational_channelwise_hybrid_control_mean'
weight=False

nn_dir_header = '/cnn_heteroscedastic_variational_channelwise_hybrid_control_us=2_in=11_rec=5_out=14_opt=AdamOptimizer_drop=0.0_prep=standard_Diverse_TS8_Subsample343_'
save_dir = '/SAN/vision/hcp/Ryu/non-HCP/Prisma/Diffusion_2.5mm/MAP/cnn_heteroscedastic_variational_channelwise_hybrid_control_weight_mean'
weight=False

# For each component, compute the weighted average:
for slice_idx in range(2,25):
    print('MAP idx: %i/24' % (slice_idx,))
    if weight:
        current_pred = 0
        norm_const = 0
        print("compute the weighted average of MAP components.")
        for nn_idx in range(1,9):
            print('     model: %i/8' % (nn_idx,))
            nn_dir = nn_dir_header + '%03i' % (nn_idx,)

            map_file = recon_dir + nn_dir + '/dt_recon_%i.nii' % (slice_idx,)
            std_data_file = recon_dir + nn_dir + '/dt_std_data_%i.nii' % (slice_idx,)
            std_model_file = recon_dir + nn_dir + '/dt_std_model_%i.nii' % (slice_idx,)

            map_pred = nib.load(map_file).get_data()
            map_uncertainty = np.sqrt(np.square(nib.load(std_data_file).get_data()) + \
                                      np.square(nib.load(std_model_file).get_data()))

            current_pred += map_pred/map_uncertainty
            norm_const += 1/map_uncertainty

        final_pred = current_pred/norm_const
        save_file = save_dir + '/dt_recon_%i.nii' % (slice_idx,)
        sr_utility.ndarray_to_nifti(final_pred, save_file, ref_file=None)
        print('Done.')
    else:
        print("compute the average of MAP components.")
        current_pred = 0
        for nn_idx in range(1, 9):
            print('     model: %i/8' % (nn_idx,))
            nn_dir = nn_dir_header + '%03i' % (nn_idx,)
            map_file = recon_dir + nn_dir + '/dt_recon_%i.nii' % (slice_idx,)
            current_pred += nib.load(map_file).get_data()

        final_pred = current_pred/8.0
        save_file = save_dir + '/dt_recon_%i.nii' % (slice_idx,)
        sr_utility.ndarray_to_nifti(final_pred, save_file, ref_file=None)
        print('Done.')




