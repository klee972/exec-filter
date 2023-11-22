#!/usr/bin/env

export TMPDIR=/shared/nitishg/tmp
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export CUDA=-1

INCLUDE_PACKAGE=allennlp_semparse

SPLIT=1
export DEV_DATA=./resources/data/WikiTableQuestions/data/random-split-${SPLIT}-dev.examples
# IID
# ./resources/data/nlvr/processed/${SPLIT}_grouped.json
# Comp-gem
# ./resources/data/nlvr/comp_gen/absstr_v1/${SPLIT}.json

# MODEL PATH
SERIALIZATION_DIR=./resources/checkpoints/wtq/basicerm/split_1/ERM/Iter3_MDS16
MODEL_TAR_GZ=${SERIALIZATION_DIR}/model.tar.gz

mkdir ${SERIALIZATION_DIR}/predictions
OUTPUT_VISUALIZE_PATH=${SERIALIZATION_DIR}/predictions/split${SPLIT}-dev-visualize.jsonl
#OUTPUT_PREDICTION_PATH=${SERIALIZATION_DIR}/predictions/${SPLIT}-predictions.jsonl
#OUTPUT_METRICS_PATH=${SERIALIZATION_DIR}/predictions/${SPLIT}-metrics.json

VISUALIZE_PREDICTOR=wtq-parser-visualize
# PREDICTION_PREDICTOR=nlvr-parser-predictions


allennlp predict --output-file ${OUTPUT_VISUALIZE_PATH} \
                 --batch-size 10 --silent \
                 --cuda-device ${CUDA} \
                 --predictor ${VISUALIZE_PREDICTOR} \
                 --include-package allennlp_semparse \
                 --use-dataset-reader \
                 ${MODEL_TAR_GZ} ${DEV_DATA}


#allennlp predict --output-file ${OUTPUT_PREDICTION_PATH} \
#                 --batch-size 4 --silent \
#                 --cuda-device ${CUDA} \
#                 --predictor ${PREDICTION_PREDICTOR} \
#                 --include-package allennlp_semparse \
#                 ${MODEL_TAR_GZ} ${DEV_DATA}  &


#allennlp evaluate --output-file ${OUTPUT_METRICS_PATH} \
#                  --cuda-device ${CUDA} \
#                  --include-package allennlp_semparse \
#                  --overrides "{"model": {"beam_size": 10}}" \
#                  ${MODEL_TAR_GZ} ${DEV_DATA}

# --overrides "{"model": {"max_decoding_steps": 14}}" \
#--overrides "{"model": {"beam_size": 10}}" \

#echo -e "Metrics written to: ${OUTPUT_METRICS_PATH}"
#echo -e "Predictions written to: ${OUTPUT_PREDICTION_PATH}"
echo -e "Visualizations written to: ${OUTPUT_VISUALIZE_PATH}"