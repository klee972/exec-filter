#! /usr/bin/env python
import json
import argparse
from typing import Tuple, List, Dict
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
    ),
)

from allennlp_semparse.domain_languages import NlvrLanguageFuncComposition


def read_json_line(line: str) -> Tuple[str, str, List[List[str]]]:
    data = json.loads(line)
    instance_id = data.get("identifier", None)
    if instance_id is None:
        instance_id = data.get("id", None)
    sentence = data["sentence"]
    correct_sequences = data.get("correct_sequences", None)
    return instance_id, sentence, correct_sequences


def write_candidate_programs(
    input_file: str,
    output_file: str,
) -> None:
    """
    Reads a grouped-NLVR dataset which contains "correct_sequences" (key); these are action-sequences for consistent
    programs for an instance. They can be generated from (1) exhaustive search or (2) beam-search on a semantic-parser

    Output:
    -------
    output_file: A txt file where we write the logical-forms for human consumption
    """
    language = NlvrLanguageFuncComposition({})
    output_dicts = []
    for line in open(input_file):
        instance_id, sentence, correct_sequences = read_json_line(line)
        if not correct_sequences:
            continue
        candidate_logical_forms = [
            language.action_sequence_to_logical_form(a) for a in correct_sequences
        ]
        output_dict = {
            "id": instance_id,
            "sentence": sentence,
            "candidate_logical_forms": candidate_logical_forms,
        }
        output_dicts.append(output_dict)

    print("Writing {} examples to: {}".format(len(output_dicts), output_file))
    with open(output_file, "w") as outfile:
            json.dump(output_dicts, outfile, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="NLVR data file")
    parser.add_argument("output", type=str, help="Output json")

    args = parser.parse_args()
    write_candidate_programs(
        args.input,
        args.output,
    )
