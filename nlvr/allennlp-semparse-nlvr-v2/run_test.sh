export MODE=ERM
export EXP=iwbs5_aww8_erm_soft
export ITER=5
export MDS=22

for SEED in 5 21 42 1337
do
  bash ./scripts/nlvr_eval/run_best_eval.sh \
    ./resources/data/nlvr/processed/test.json \
    ./resources/checkpoints/nlvr/${EXP}/SEED_${SEED}/${MODE}/Iter${ITER}_MDS${MDS}/model.tar.gz \
    ./resources/checkpoints/nlvr/${EXP}/SEED_${SEED}/${MODE}/Iter${ITER}_MDS${MDS}/test.csv \
    ./resources/checkpoints/nlvr/${EXP}/SEED_${SEED}/${MODE}/Iter${ITER}_MDS${MDS}/metrics.txt
done
