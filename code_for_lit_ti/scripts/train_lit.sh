#!/usr/bin/env bash
GPUS=$1

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=3124 \
    --use_env main.py --config config/lit-ti.json