#!/usr/bin/env

export TMPDIR=/shared/nitishg/tmp
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export CUDA=-1

TRAIN_SEARCH_JSON=./resources/data/nlvr/processed/agenda_PL_ML11/train_grouped.json

TRAIN_ERM_JSON=./resources/data/nlvr/processed/paired_data/train_v13_P1M1NT1.json

DEV_JSON=./resources/data/nlvr/processed/dev_grouped.json

CKPT_ROOT=./resources/checkpoints/nlvr/pairedv13_P1M1NT1_LR0005

# 21 42 1337 5 14
for SEED in 23 41 57 64 89
do
  python scripts/train/iterative_train.py \
    --train_search_json ${TRAIN_SEARCH_JSON} \
    --dev_json ${DEV_JSON} \
    --train_json ${TRAIN_ERM_JSON} \
    --ckpt_root ${CKPT_ROOT} \
    --seed ${SEED} &
done


