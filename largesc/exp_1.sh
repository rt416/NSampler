#!/usr/bin/bash

# subject number vs rmse

for i in 4 8 16 32; do
    e = largesc_v1
    c = 1
    tsp tf-python run_me_v2.py --experiment $e -ts $i --disp
done

