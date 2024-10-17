#!/bin/bash

python main.py \
    --beta_schedule "cosine" \
    --loss_type "MSE" \
    --weight_type "min_snr_5"

python main.py \
    --beta_schedule "optim" \
    --loss_type "MAPPED_MSE" \
    --weight_type "constant"

