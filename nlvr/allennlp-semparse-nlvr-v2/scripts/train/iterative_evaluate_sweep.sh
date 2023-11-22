#!/usr/bin/env

export TMPDIR=/shared/nitishg/tmp
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export CUDA=-1

SPLIT=test

EVAL_JSON=./resources/data/nlvr/comp_gen/absstr_S21/${SPLIT}.json
# ./resources/data/nlvr/comp_gen/absstr_S21/${SPLIT}.json
# ./resources/data/nlvr/comp_gen/attrpair/${SPLIT}.json
# ./resources/data/nlvr/processed/${SPLIT}_grouped.json

CKPT_ROOT=./resources/checkpoints/nlvr/comp-gen/absstr_S21/pairedv13_P1M1NT1

# 21 42 1337 5 14
# 23 41 57 64 89
for SEED in 21 42 1337 5 14 23 41 57 64 89
do
  SERIALIZATION_DIR=${CKPT_ROOT}/SEED_${SEED}/ERM/Iter5_MDS22
  MODEL_TAR_GZ=${SERIALIZATION_DIR}//model.tar.gz
  mkdir ${SERIALIZATION_DIR}/predictions
  OUTPUT_METRICS_PATH=${SERIALIZATION_DIR}/predictions/${SPLIT}-metrics.json

  allennlp evaluate --output-file ${OUTPUT_METRICS_PATH} \
                  --cuda-device ${CUDA} \
                  --include-package allennlp_semparse \
                  --overrides "{"model": {"beam_size": 10}}" \
                  ${MODEL_TAR_GZ} ${EVAL_JSON} &

done


