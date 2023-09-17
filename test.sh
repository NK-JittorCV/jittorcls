#! /bin/bash

CUDA_VISIBLE_DEVICES=1 python test.py \
    --batch_size 64 \
    --model p2t_tiny \
    --dataset ImageNet \
    --data_root data \
    --batch_size 64 \
    --shuffle False \
    --num_workers 32 \
    --data_root /path/imagenet
