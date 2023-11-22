#!/usr/bin/env

export TMPDIR=/shared/nitishg/tmp
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export OPENMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

export CUDA=-1

INCLUDE_PACKAGE=allennlp_semparse
CONFIGFILE=training_config/nlvr_erm_parser.jsonnet

# DATA PATH
# export DATADIR=agenda_v6_ML11
export TRAIN_DATA=./resources/data/nlvr/processed/train_grouped.json
export DEV_DATA=./resources/data/nlvr/processed/dev_grouped.json

# HYPER-PARAMETERS
export MDS=14

export MML_MODEL_TAR=./resources/checkpoints/mml_parser/nlvr/agenda_v6_ML11/MDS_18/S_42/model.tar.gz

export SEED=1337

# SERIALIZATION PATH
CHECKPOINT_ROOT=./resources/checkpoints
MODEL_DIR=erm_parser/nlvr
PARAMETERS=MDS_${MDS}/S_${SEED}
SERIALIZATION_DIR=${CHECKPOINT_ROOT}/${MODEL_DIR}/${PARAMETERS}

# SERIALIZATION_DIR=${CHECKPOINT_ROOT}/test; rm -r ${SERIALIZATION_DIR}


allennlp train --include-package allennlp_semparse  -s ${SERIALIZATION_DIR} ${CONFIGFILE}
