#!/usr/bin/env

export TMPDIR=/shared/nitishg/tmp
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export CUDA=-1

INCLUDE_PACKAGE=allennlp_semparse
CONFIGFILE=training_config/nlvr_paired_parser.jsonnet

# DATA PATH
export TRAIN_DATA=./resources/data/nlvr/processed/paired_data/train_v2_P1M1NT1.json
export DEV_DATA=./resources/data/nlvr/processed/dev_grouped.json

# HYPER-PARAMETERS
export MDS=14

export EPOCHS=50
export SEED=1337

export MML_MODEL_TAR=./resources/checkpoints/nlvr/pruned/pairedv2_T07_P1M2/SEED_21/MML/Iter3_MDS18/model.tar.gz

# SERIALIZATION PATH
CHECKPOINT_ROOT=./resources/checkpoints
MODEL_DIR=nlvr_paired_parser/${DATADIR}
PARAMETERS=MDS_${MDS}/S_${SEED}
SERIALIZATION_DIR=${CHECKPOINT_ROOT}/${MODEL_DIR}/${PARAMETERS}-ActionShare

SERIALIZATION_DIR=./resources/checkpoints/test; rm -rf ./resources/checkpoints/test

allennlp train --include-package allennlp_semparse  -s ${SERIALIZATION_DIR} ${CONFIGFILE}
