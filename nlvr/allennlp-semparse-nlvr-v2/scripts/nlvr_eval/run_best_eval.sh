#!/usr/bin/env

# export TMPDIR=/shared/nitishg/tmp
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export CUDA=1

INCLUDE_PACKAGE=allennlp_semparse

# DATA PATH
export DATASET_JSON=$1
export MODEL_TAR_GZ=$2
export CSV_OUTPUT=$3
export METRICS_OUTPUT=$4

CSV_PREDICTOR=nlvr-parser

allennlp predict --output-file ${CSV_OUTPUT} \
                 --batch-size 4 --silent \
                 --cuda-device ${CUDA} \
                 --predictor ${CSV_PREDICTOR} \
                 --include-package allennlp_semparse \
                 ${MODEL_TAR_GZ} ${DATASET_JSON}


echo -e "Predictions written to: ${CSV_OUTPUT}"
python scripts/nlvr_eval/metrics_structured_rep.py ${CSV_OUTPUT} ${DATASET_JSON} > ${METRICS_OUTPUT}
echo -e "Metrics written to: ${METRICS_OUTPUT}"

#### Example command ####
# bash ./scripts/nlvr_eval/run_best_eval.sh \
#   ./resources/data/test-p.json \
#   ./resources/checkpoints/nlvr/final_ckpts/modelB_best.tar.gz \
#   ./resources/checkpoints/nlvr/final_ckpts/modelB-test-p.csv \
#   ./resources/checkpoints/nlvr/final_ckpts/modelB-test-p-metrics.txt