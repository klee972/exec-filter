#!/usr/bin/env

# export TMPDIR=/shared/nitishg/tmp
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export CUDA=-1

INCLUDE_PACKAGE=allennlp_semparse

# DATA PATH
export DATASET_JSON=$1
export SERDIR=$2
export CSV_OUTPUT_NAME=$3
export METRICS_OUTPUT_NAME=$4

CSV_PREDICTOR=nlvr-parser

for SEED in 5 14 21 23 41 42 57 64 89 1337
do
  export MODEL_TAR_GZ=${SERDIR}/model_${SEED}.tar.gz
  export CSV_OUTPUT=${SERDIR}/${SEED}_${CSV_OUTPUT_NAME}.csv
  export METRICS_OUTPUT=${SERDIR}/${SEED}_${METRICS_OUTPUT_NAME}.txt

  allennlp predict --output-file ${CSV_OUTPUT} \
                 --batch-size 4 --silent \
                 --cuda-device ${CUDA} \
                 --predictor ${CSV_PREDICTOR} \
                 --include-package allennlp_semparse \
                 ${MODEL_TAR_GZ} ${DATASET_JSON}

  echo -e "Predictions written to: ${CSV_OUTPUT}"
  python scripts/nlvr_eval/metrics_structured_rep.py ${CSV_OUTPUT} ${DATASET_JSON} > ${METRICS_OUTPUT}
  echo -e "Metrics written to: ${METRICS_OUTPUT}"
done

#### Example command ####
# bash ./scripts/nlvr_eval/run_all_eval.sh \
#   ./resources/data/test-p.json \
#   ./resources/checkpoints/nlvr/final_ckpts/all_modelA \
#   test-p \
#   test-p-metrics