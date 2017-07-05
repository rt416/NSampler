#!/usr/bin/env bash

gt_dir=/Users/ryutarotanno/tmp/MAP/HCP_map_v2/
base_dir=/Users/ryutarotanno/tmp/iqt_DL/auro/

python run_me.py --gt_dir $gt_dir --base_dir $base_dir -ts 1 --is_map --method espcn --is_shuffle --is_BN -pl 100
#python run_me_v2.py --gt_dir $gt_dir --base_dir $base_dir -ts 2 --is_map 

