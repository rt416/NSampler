#!/usr/bin/env bash
python reconstruct_nonHCP.py --experiment 11July_dcespcn \
                             --dataset tumour \
                             --method dcespcn --is_shuffle --hetero -ts 32 \
                             --no_layers 9 -ir 12 --no_filters 12 --is_BN --pad_size 0 --is_clip \
                             --mc_no_samples 1

python reconstruct_nonHCP.py --experiment 11July_dcespcn \
                             --dataset ms \
                             --method dcespcn --is_shuffle --hetero -ts 32 \
                             --no_layers 9 -ir 12 --no_filters 12 --is_BN --pad_size 0 --is_clip \
                             --mc_no_samples 1


: <<'END'
# testing locally
bid=/Users/ryutarotanno/DeepLearning/nsampler/data
brd=/Users/ryutarotanno/DeepLearning/nsampler/recon/miccai2017
dataset=tumour
base_dir=/Users/ryutarotanno/tmp/iqt_DL/auro
exp=11July_dcespcn


python reconstruct_nonHCP.py --base_dir $base_dir --experiment $exp \
                             --base_input_dir $bid --base_recon_dir $brd \
                             --dataset $dataset \
                             --method dcespcn --is_shuffle --hetero -ts 32 \
                             --no_layers 9 -ir 12 --no_filters 12 --is_BN --pad_size 0 --is_clip \
                             --mc_no_samples 1

END

