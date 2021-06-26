#!/usr/bin/env bash

GPUS=$1
CHECKPOINT=$2

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=1234 \
    --use_env main.py --config config/lit-ti.json --resume $CHECKPOINT --eval