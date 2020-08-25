#!/usr/bin/env bash

gt_dir=/Users/ryutarotanno/Datasets/hcp/
base_dir=/Users/ryutarotanno/DeepLearning/nsampler/recon/journal2020/

python run_me.py --experiment 2020-08-07_test --method espcn --is_shuffle --gt_dir $gt_dir --base_dir $base_dir -ts 2 -pl 100 --batch_size 6 --no_epochs 2
