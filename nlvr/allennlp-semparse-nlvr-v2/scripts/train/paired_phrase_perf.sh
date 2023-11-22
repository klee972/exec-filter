#!/usr/bin/env

export TMPDIR=/shared/nitishg/tmp
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

SPLIT=test

BASIC_ERM=./resources/checkpoints/nlvr/pruned/basicerm/SEED_1337/ERM/Iter5_MDS22/predictions/${SPLIT}-predictions.jsonl

PAIRED_V1=./resources/checkpoints/nlvr/pruned/pairedv1_T07_F1_P1M1/SEED_1337/ERM/Iter5_MDS22/predictions/${SPLIT}-predictions.jsonl

PAIRED_V7=./resources/checkpoints/nlvr/pruned/pairedv7_T07_F1_P1M2/SEED_1337/ERM/Iter5_MDS22/predictions/${SPLIT}-predictions.jsonl

PAIRED_V8=./resources/checkpoints/nlvr/pruned/pairedv8_T07_F1_P1M2/SEED_1337/ERM/Iter5_MDS22/predictions/${SPLIT}-predictions.jsonl

PAIRED_V2=./resources/checkpoints/nlvr/pruned/pairedv2_T07_F1_P1M1/SEED_21/ERM/Iter5_MDS22/predictions/${SPLIT}-predictions.jsonl

PAIRED_V9=./resources/checkpoints/nlvr/pruned/pairedv9_T07_F1_P1M2/SEED_21/ERM/Iter5_MDS22/predictions/${SPLIT}-predictions.jsonl

PAIRED_V10=./resources/checkpoints/nlvr/pruned/pairedv10_T07_F1_P1M2/SEED_21/ERM/Iter5_MDS22/predictions/${SPLIT}-predictions.jsonl

PAIRED_V11=./resources/checkpoints/nlvr/pruned/pairedv11_T07_F1_P1M2/SEED_42/ERM/Iter5_MDS22/predictions/${SPLIT}-predictions.jsonl

PAIRED_V11NT=./resources/checkpoints/nlvr/pruned/pairedv11_T07_F1_P1M1NT1/SEED_21/ERM/Iter5_MDS22/predictions/${SPLIT}-predictions.jsonl

PAIRED_V13NT=./resources/checkpoints/nlvr/pruned/pairedv13_T07_F1_P1M1NT1/SEED_21/ERM/Iter5_MDS22/predictions/${SPLIT}-predictions.jsonl


VERSION=v13
PHRASES=./scripts/nlvr_v2/data/paired_phrases_${VERSION}.json


python scripts/nlvr_v2/analysis/phrase_based_performance.py ${PHRASES} ${BASIC_ERM} ${PAIRED_V13NT}

