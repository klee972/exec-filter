#! /usr/bin/env python
import json
import argparse
from typing import Tuple, List, Dict
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
)


def read_json_line(line: str) -> Tuple[str, str, List[Dict], List[str]]:
    data = json.loads(line)
    instance_id = data["identifier"]
    sentence = data["sentence"]
    if "worlds" in data:
        structured_reps = data["worlds"]
        label_strings = [label_str.lower() for label_str in data["labels"]]
    else:
        # We're reading ungrouped data.
        structured_reps = [data["structured_rep"]]
        label_strings = [data["label"].lower()]
    return instance_id, sentence, structured_reps, label_strings


def write_tsv(
    input_file: str,
    output_file: str,
) -> None:
    """
    Reads an NLVR dataset and outputs a TSV containing utterances.
    """
    instance_ids = []
    utterances = []
    for line in open(input_file):
        instance_id, sentence, structured_reps, label_strings = read_json_line(line)
        instance_ids.append(instance_id)
        utterances.append(sentence)

    print("Total number of utterances: {}".format(len(utterances)))
    print("Writing {} instances to: {}".format(len(utterances), output_file))
    with open(output_file, "w") as outfile:
        outfile.write("Instance-Id\tUtterance\n")
        for i in range(len(instance_ids)):
            instance_id = instance_ids[i]
            sentence = utterances[i]
            outfile.write("{}\t{}\n".format(instance_id, sentence))
        outfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="NLVR data file")
    parser.add_argument("output", type=str, help="Output tsv")
    args = parser.parse_args()
    print(args)
    write_tsv(
        args.input,
        args.output,
    )
