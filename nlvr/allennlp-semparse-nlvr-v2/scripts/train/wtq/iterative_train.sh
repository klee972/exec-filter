#!/usr/bin/env

export TMPDIR=/shared/nitishg/tmp
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export CUDA=-1

# time python scripts/wikitables/search_for_logical_forms.py \
#   ./resources/data/WikiTableQuestions \
#   ./resources/data/WikiTableQuestions/data/training.examples \
#   ./resources/data/wtq/search_non_conservative \
#   --max-path-length 10 \
#   --use-agenda \
#   --output-separate-files \
#   --num-splits 20

export TABLES_DIR=./resources/data/WikiTableQuestions
export TRAIN_SEARCH_DIR=./resources/data/wtq/search_non_conservative

for SPLIT in 1 2 3 4 5
do
  export TRAIN_DATA=./resources/data/WikiTableQuestions/data/random-split-${SPLIT}-train.examples
  export DEV_DATA=./resources/data/WikiTableQuestions/data/random-split-${SPLIT}-dev.examples

  python scripts/train/wtq/iterative_train.py \
    --train_examples ${TRAIN_DATA} \
    --dev_examples ${DEV_DATA} \
    --train_search_dir ${TRAIN_SEARCH_DIR} \
    --tables_dir ${TABLES_DIR} \
    --ckpt_root ./resources/checkpoints/wtq/basicerm \
    --split_num ${SPLIT} &

#  allennlp train --include-package allennlp_semparse \
#    -s ./resources/checkpoints/wtq/mml-split${SPLIT} \
#    training_config/wikitables_mml_parser.jsonnet &


done



