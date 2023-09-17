#! /bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py \
    --lr 1e-4 \
    --gamma 0.9 \
    --epochs 40 \
    --weight_decay 1e-4 \
    --momentum 0.9 \
    --accum_iter 1 \
    --log_info Log \
    --save_path checkpoints/model.pkl \
    --val_time 5 \
    --model p2t_tiny \
    --dataset ImageNet \
    --data_root data \
    --batch_size 64 \
    --shuffle False \
    --num_workers 32
