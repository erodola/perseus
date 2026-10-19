#!/bin/bash
# ViT-B/16 at 224, encrypted, 4 EuroSAT images: the table number is SUMMARY s/image (median of the image walls).
#   GPU=0 NODE=1 MEMS=1,2 bash scripts/repro/vit_base.sh
source "$(dirname "$0")/common.sh"
bind $PY scripts/repro/vit_npy.py --images scripts/repro/data/vit_eurosat --n 4 \
    --plan bootstrap_placements/vit_b16_224 --device "$GPU" "$@"
