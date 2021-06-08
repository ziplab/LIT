#!/bin/bash

GPUS=$1

python -m torch.distributed.launch --nproc_per_node $GPUS --master_port 13335  main.py \
--cfg configs/lit-base.yaml --use-checkpoint
