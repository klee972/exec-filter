#! /usr/bin/env python
import json
import argparse
from typing import Tuple, List, Dict
import os
import sys
import random
random.seed(42)


def read_json_line(line: str) -> Tuple[str, str, List[List[str]]]:
    data = json.loads(line)
    instance_id = data.get("identifier", None)
    if instance_id is None:
        instance_id = data.get("id", None)
    sentence = data["sentence"]
    correct_sequences = data.get("correct_sequences", None)
    return instance_id, sentence, correct_sequences


def write_sample_for_program_annotation(
    input_file: str,
    output_file: str,
    num_samples: int
) -> None:
    """
    Write few training sentences for manual program annotations

    Output:
    -------
    output_file: A txt file where we write the logical-forms for human consumption
    """
    instances = []
    for line in open(input_file):
        identifier, sentence, correct_sequences = read_json_line(line)
        instances.append({"id": identifier, "sentence": sentence})

    random.shuffle(instances)
    output_dicts = []
    for i in range(num_samples):
        output_dicts.append({
            "identifier": instances[i]["id"],
            "sentence": instances[i]["sentence"],
            "logical_form": ""
        })

    print("Writing output to: {}".format(output_file))
    with open(output_file, "w") as outfile:
        json.dump(output_dicts, outfile, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="NLVR data file")
    parser.add_argument("output", type=str, help="Output json")
    parser.add_argument("num_samples", type=int, default=100, help="Num of samples to write")

    args = parser.parse_args()
    write_sample_for_program_annotation(
        args.input,
        args.output,
        args.num_samples
    )
