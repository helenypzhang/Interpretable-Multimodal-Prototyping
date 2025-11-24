#!/bin/bash


CUDA_VISIBLE_DEVICES=2  python tools/clip_feature.py --root  DATASET/224/mnt/disk10T_2/fuyibing/wxf_data/TCGA/brain/extract_224
CUDA_VISIBLE_DEVICES=2  python tools/clip_feature.py --root  DATASET/224/supple_224
CUDA_VISIBLE_DEVICES=2  python tools/clip_feature.py --root  DATASET/224/extract_224