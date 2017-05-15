#!/usr/bin/bash

# subject number vs rmse
e = largesc_v1
c = 1
for i in 4 8 16 32; do
    tsp CUDA_VISIBLE_DEVICES=$c tf-python run_me_v2.py --experiment $e  -ts $i
done

