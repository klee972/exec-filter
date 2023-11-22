#!/usr/bin/env

export TMPDIR=/shared/nitishg/tmp
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export CUDA=-1

INCLUDE_PACKAGE=allennlp_semparse

SPLIT=test
export DEV_DATA=./resources/data/nlvr/processed/${SPLIT}_grouped.json
# IID
# ./resources/data/nlvr/processed/${SPLIT}_grouped.json
# Comp-gem
# ./resources/data/nlvr/comp_gen/absstr_v1/${SPLIT}.json

# MODEL PATH
SERIALIZATION_DIR=./resources/checkpoints/nlvr/pruned/pairedv14_T07_F1_P1M1/SEED_1337/ERM/Iter5_MDS22
MODEL_TAR_GZ=${SERIALIZATION_DIR}/model.tar.gz

mkdir ${SERIALIZATION_DIR}/predictions
OUTPUT_VISUALIZE_PATH=${SERIALIZATION_DIR}/predictions/${SPLIT}-visualize.jsonl
OUTPUT_PREDICTION_PATH=${SERIALIZATION_DIR}/predictions/${SPLIT}-predictions.jsonl
OUTPUT_METRICS_PATH=${SERIALIZATION_DIR}/predictions/${SPLIT}-metrics.json

VISUALIZE_PREDICTOR=nlvr-parser-visualize
PREDICTION_PREDICTOR=nlvr-parser-predictions


allennlp predict --output-file ${OUTPUT_VISUALIZE_PATH} \
                 --batch-size 4 --silent \
                 --cuda-device ${CUDA} \
                 --predictor ${VISUALIZE_PREDICTOR} \
                 --include-package allennlp_semparse \
                 ${MODEL_TAR_GZ} ${DEV_DATA} &


allennlp predict --output-file ${OUTPUT_PREDICTION_PATH} \
                 --batch-size 4 --silent \
                 --cuda-device ${CUDA} \
                 --predictor ${PREDICTION_PREDICTOR} \
                 --include-package allennlp_semparse \
                 ${MODEL_TAR_GZ} ${DEV_DATA}  &


allennlp evaluate --output-file ${OUTPUT_METRICS_PATH} \
                  --cuda-device ${CUDA} \
                  --include-package allennlp_semparse \
                  --overrides "{"model": {"beam_size": 10}}" \
                  ${MODEL_TAR_GZ} ${DEV_DATA}

# --overrides "{"model": {"max_decoding_steps": 14}}" \
#--overrides "{"model": {"beam_size": 10}}" \

echo -e "Metrics written to: ${OUTPUT_METRICS_PATH}"
echo -e "Predictions written to: ${OUTPUT_PREDICTION_PATH}"
echo -e "Visualizations written to: ${OUTPUT_VISUALIZE_PATH}"