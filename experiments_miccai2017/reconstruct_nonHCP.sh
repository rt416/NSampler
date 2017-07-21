#!/usr/bin/env bash

bid = '/Users/ryutarotanno/DeepLearning/nsampler/data'
brd = '/Users/ryutarotanno/DeepLearning/nsampler/recon/miccai2017'
python reconstruct_nonHCP.py --base_input_dir $bid --base_recon_dir $brd --dataset tumour
