#!/bin/bash

# custom config
TYPE="Classification"
TRAINER=UMEML_GAN
DATA=DATASET

export CUDA_LAUNCH_BLOCKING=1


# for SEED in 1 2 3 4 5
for SEED in 1 2 3 4 5
do
    DIR=output/train/${TYPE}/${TRAINER}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Resuming..."
        python tools/train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer MBTRAIN \
        --config-file configs/${TYPE}/${TRAINER}.yaml \
        --output-dir ${DIR} 

    else
        echo "Run this job and save the output to ${DIR}"
        python tools/train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer MBTRAIN \
        --config-file configs/${TYPE}/${TRAINER}.yaml \
        --output-dir ${DIR} 
    fi
done

