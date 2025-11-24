#!/bin/bash

# custom config
TYPE="Classification"
TRAINER=ADD
DATA=DATASET

CUDA_VISIBLE_DEVICES=1

for SEED in 1 2 3 4 5
# for SEED in 1 
do
    DIR=output/train/${TYPE}/${TRAINER}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Resuming..."
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python tools/train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --config-file configs/${TYPE}/${TRAINER}.yaml \
        --output-dir ${DIR} 

    else
        echo "Run this job and save the output to ${DIR}"
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python tools/train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --config-file configs/${TYPE}/${TRAINER}.yaml \
        --output-dir ${DIR} 
    fi

done
