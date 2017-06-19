data=/media/daniel/HDD/superres_data/HCP/
mask=/media/daniel/HDD/superres_data/HCP/
mask_sp=T1w/Diffusion
python run_me.py --base_dir . --gt_dir  $data --mask_dir $mask --mask_subpath $mask_sp --is_shuffle --is_dt_all
