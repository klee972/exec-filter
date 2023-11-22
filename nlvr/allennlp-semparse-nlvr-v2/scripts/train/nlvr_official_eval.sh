#!/usr/bin/env

export TMPDIR=/shared/nitishg/tmp
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export OPENMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

export CUDA=-1

INCLUDE_PACKAGE=allennlp_semparse

# DATA PATH
export DEV_DATA=./resources/data/nlvr/raw/dev.json

# MODEL PATH
SERIALIZATION_DIR=./resources/checkpoints/mml_parser/nlvr/agenda_v6_ML11/PI_S_21-MDS_20-Iter4
MODEL_TAR_GZ=${SERIALIZATION_DIR}/model.tar.gz

mkdir ${SERIALIZATION_DIR}/predictions
OUTPUT_PATH=${SERIALIZATION_DIR}/predictions/dev-ungrouped-preds.jsonl

CSV_PREDICTOR=nlvr-parser

allennlp predict --output-file ${OUTPUT_PATH} \
                 --batch-size 4 --silent \
                 --cuda-device ${CUDA} \
                 --predictor ${CSV_PREDICTOR} \
                 --include-package allennlp_semparse \
                 ${MODEL_TAR_GZ} ${DEV_DATA}


echo -e "Predictions written to: ${OUTPUT_PATH}"

python ~/code/nlvr-official/nlvr/metrics_structured_rep.py ${OUTPUT_PATH} ${DEV_DATA}

