python scripts/train/iterative_train.py \
    --train_search_json ./resources/data/nlvr/processed/agendav6_SORT_ML11/train_grouped.json \
    --train_json  ./resources/data/nlvr/processed/train_grouped_proxy_utt_50.json \
    --erm_model worldcode \
    --cuda-device 3 \
    --ckpt_root ./resources/checkpoints/nlvr/iwbs5_aww8_erm_soft \
    --seed 1337

# SEED: 5 21 42 1337
