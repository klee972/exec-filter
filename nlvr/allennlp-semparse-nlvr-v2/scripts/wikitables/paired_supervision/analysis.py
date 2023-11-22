#! /usr/bin/env python
from typing import List, Dict, Tuple
import sys
import os
import json
import copy
import argparse
import random
from collections import defaultdict

random.seed(42)

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
    ),
)

from allennlp_semparse.dataset_readers.wikitables import parse_example_line


def get_ngrams(tokens, n) -> List:
    output = []
    for i in range(len(tokens) - n + 1):
        output.append(tokens[i : i + n])
    return output


def analyze_data(train_examples_file: str, ngram_len: int = 3):
    num_lines = 0
    num_instances = 0
    examples: List[Dict] = []
    with open(train_examples_file) as data_file:
        for line in data_file.readlines():
            line = line.strip("\n")
            if not line:
                continue
            num_lines += 1
            # "id", "question", "table_filename", "target_values"
            parsed_info = parse_example_line(line)
            examples.append(parsed_info)
            num_instances += 1

    print("Num examples: {}".format(num_instances))

    table2examples = defaultdict(list)
    for example in examples:
        table2examples[example["table_filename"]].append(example)

    avg_questions_per_table = float(num_instances)/len(table2examples)
    print("num_of_tables: {}".format(len(table2examples)))
    print("avg_questions_per_table: {}".format(avg_questions_per_table))

    for table, tab_examples in table2examples.items():
        print(table)
        for ex in tab_examples:
            question = ex["question"]
            print(question)
        print()

        # ngram2questions = defaultdict(list)
        # for ex in tab_examples:
        #     question = ex["question"]
        #     question = question.replace("?", " ?").lower()
        #     tokens = question.split(" ")
        #     ngrams = get_ngrams(tokens, ngram_len)
        #     for ngram in ngrams:
        #         ngram2questions[" ".join(ngram)].append(ex)
        #
        # for ngram, ngram_exs in ngram2questions.items():
        #     if len(ngram_exs) <= 1:
        #         continue
        #     print()
        #     print(ngram)
        #     print(table)
        #     for ex in ngram_exs:
        #         print(f"{ex['question']} {ex['target_values']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_examples_file", type=str, help="Input data file")


    args = parser.parse_args()
    analyze_data(train_examples_file=args.train_examples_file)


