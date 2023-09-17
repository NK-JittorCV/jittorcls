CUDA_VISIBLE_DEVICES="0,1,2,3" mpirun -np 4 python -m test \
    --batch_size 64 \
    --model p2t_tiny \
    --dataset ImageNet \
    --data_root data \
    --batch_size 64 \
    --shuffle False \
    --num_workers 32
