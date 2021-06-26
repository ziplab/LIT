#!/bin/bash

GPUS=$1
CHECKPOINT=$2

python -m torch.distributed.launch --nproc_per_node $GPUS --master_port 13335  main.py \
--cfg configs/lit-small.yaml --resume $CHECKPOINT --eval
