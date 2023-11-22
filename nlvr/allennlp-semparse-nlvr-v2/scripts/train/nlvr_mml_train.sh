#!/usr/bin/env

export TMPDIR=/shared/nitishg/tmp
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export OPENMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

export CUDA=-1

INCLUDE_PACKAGE=allennlp_semparse
CONFIGFILE=training_config/nlvr_mml_parser.jsonnet

# DATA PATH
export DATADIR=agendav6_SORT_ML10
export TRAIN_DATA=./resources/data/nlvr/processed/${DATADIR}/train_grouped.json
export DEV_DATA=./resources/data/nlvr/processed/dev_grouped.json

# HYPER-PARAMETERS
export MDS=12
export EPOCHS=50
export SEED=42

# SERIALIZATION PATH
CHECKPOINT_ROOT=./resources/checkpoints
MODEL_DIR=nlvr/mml_parser/${DATADIR}
PARAMETERS=MDS_${MDS}/S_${SEED}
SERIALIZATION_DIR=${CHECKPOINT_ROOT}/${MODEL_DIR}/${PARAMETERS}

# SERIALIZATION_DIR=${CHECKPOINT_ROOT}/test

allennlp train --include-package allennlp_semparse  -s ${SERIALIZATION_DIR} ${CONFIGFILE}
