Our code is mostly based on https://github.com/nitishgupta/allennlp-semparse/tree/nlvr-v2/scripts/nlvr_v2

## Installing

`allennlp-semparse` is available on PyPI. You can install through `pip` with

```
pip install allennlp-semparse
```

## Usage


# Candidate programs via exhaustive search
Run `get_nlvr_logical_forms.py` to search for candidate programs. 
With function composition and currying langauge with one action removed, 
we can search for programs with length = 11. Takes about 8 hours.

```
time python scripts/nlvr_v2/get_nlvr_logical_forms.py \
    resources/data/nlvr/processed/train_grouped.json \
    resources/data/nlvr/processed/agendav6_SORT_ML11/train_grouped.json \
    --write-action-sequences \
    --max-path-length 11
```
Coverage: `54.6%`

# Run iterative training with execution-based filtering
```
bash run_generate_proxy_utt_data.sh
bash run_iterative_train.sh
```

# Test model
```
bash run_test.sh
```