from typing import List, Dict, Tuple
import sys
import os
import json
import copy
import argparse
import random

random.seed(42)

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
    ),
)

from scripts.nlvr_v2.data.nlvr_instance import NlvrInstance, read_nlvr_data, write_nlvr_data, print_dataset_stats


def write_search_cands(train_json, all_data_search, output_filename):
    print("Reading train instances")
    train_instances: List[NlvrInstance] = read_nlvr_data(train_json)
    print_dataset_stats(train_instances)

    print("Reading all-data w/ search instances")
    all_instances: List[NlvrInstance] = read_nlvr_data(all_data_search)
    id2searchinstance = {instance.identifier: instance for instance in all_instances}
    print_dataset_stats(all_instances)

    for instance in train_instances:
        instance.correct_candidate_sequences = id2searchinstance[instance.identifier].correct_candidate_sequences

    print_dataset_stats(train_instances)
    output_dir = os.path.split(train_json)[0]
    output_json = os.path.join(output_dir, output_filename)
    write_nlvr_data(train_instances, output_json)

    # write_nlvr_data(dev_instances, os.path.join(output_dir, "dev.json"))
    # write_nlvr_data(test_instances, os.path.join(output_dir, "test.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_jsonl", type=str, help="Input data file")
    parser.add_argument("all_data_search_jsonl", type=str, help="All data json w/ search candidates")
    parser.add_argument("output_filename", type=str, help="train_search.json filename in the same dir as train_jsonl")

    args = parser.parse_args()
    write_search_cands(train_json=args.train_jsonl, all_data_search=args.all_data_search_jsonl,
                       output_filename=args.output_filename)