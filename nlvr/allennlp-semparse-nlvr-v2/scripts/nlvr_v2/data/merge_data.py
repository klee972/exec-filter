#! /usr/bin/env python
from typing import List, Dict
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


def merge_data(input_jsonl1, input_jsonl2, output_jsonl):
    """Merge Nlvr data from two files; input_jsonl1 is preferred to break ties."""

    nlvr1: List[NlvrInstance] = read_nlvr_data(input_jsonl1)
    id2instance_1: Dict[str, NlvrInstance] = {instance.identifier: instance for instance in nlvr1}
    print("Dataset1")
    print_dataset_stats(nlvr1)

    nlvr2: List[NlvrInstance] = read_nlvr_data(input_jsonl2)
    id2instance_2: Dict[str, NlvrInstance] = {instance.identifier: instance for instance in nlvr2}
    print("Dataset2:")
    print_dataset_stats(nlvr2)

    extra_in_nlvr2 = set(id2instance_2.keys()).difference(set(id2instance_1.keys()))

    merged_instances = nlvr1 + [id2instance_2[identifier] for identifier in extra_in_nlvr2]
    write_nlvr_data(merged_instances, output_jsonl)
    print("Merged:")
    print_dataset_stats(merged_instances)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl1", type=str, help="Input preferred data file ")
    parser.add_argument("input_jsonl2", type=str, help="Input data file - 2")
    parser.add_argument("output_jsonl", type=str, help="Output data file")
    args = parser.parse_args()
    merge_data(args.input_jsonl1, args.input_jsonl2, args.output_jsonl)
