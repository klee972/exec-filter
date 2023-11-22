#! /usr/bin/env python
import json
import argparse
from typing import Tuple, List
import os
import sys
import random
random.seed(42)

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
)

from allennlp.common.util import JsonDict
from allennlp_semparse.domain_languages import NlvrLanguageFuncComposition
from allennlp_semparse.domain_languages.nlvr_language_v2 import Box
from allennlp_semparse import ActionSpaceWalker


def read_json_line(line: str) -> Tuple[str, str, List[JsonDict], List[str]]:
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


def process_data(
    input_file: str,
    output_file: str,
    max_path_length: int,
    max_num_logical_forms: int,
    ignore_agenda: bool,
    allow_partial: bool,
    write_sequences: bool,
    prune_data: bool,
) -> None:
    """
    Reads an NLVR dataset and returns a JSON representation containing sentences, labels, correct and
    incorrect logical forms. The output will contain at most `max_num_logical_forms` logical forms
    each in both correct and incorrect lists. The output format is:
        ``[{"id": str, "label": str, "sentence": str, "correct": List[str], "incorrect": List[str]}]``
    """
    processed_data: JsonDict = []
    # We can instantiate the ``ActionSpaceWalker`` with any world because the action space is the
    # same for all the ``NlvrLanguage`` objects. It is just the execution that differs.
    walker = ActionSpaceWalker(NlvrLanguageFuncComposition({}), max_path_length=max_path_length)
    num_examples = 0
    (
        examples_w_correct_logical_forms,
        max_num_correct_logical_forms,
        total_num_correct_logical_forms,
    ) = (0, 0, 0)
    for line in open(input_file):
        instance_id, sentence, structured_reps, label_strings = read_json_line(line)
        worlds = []
        for structured_representation in structured_reps:
            boxes = {
                Box(object_list, box_id)
                for box_id, object_list in enumerate(structured_representation)
            }
            worlds.append(NlvrLanguageFuncComposition(boxes))
        labels = [label_string == "true" for label_string in label_strings]
        correct_logical_forms = []
        incorrect_logical_forms = []
        if ignore_agenda:
            # Get 1000 shortest logical forms.
            logical_forms = walker.get_all_logical_forms(max_num_logical_forms=1000)
        else:
            # TODO (pradeep): Assuming all worlds give the same agenda.
            sentence_agenda = worlds[0].get_agenda_for_sentence(sentence)
            logical_forms = walker.get_logical_forms_with_agenda(
                sentence_agenda,
                max_num_logical_forms * 10,
                allow_partial_match=allow_partial,
            )
        for logical_form in logical_forms:
            if all([world.execute(logical_form) == label for world, label in zip(worlds, labels)]):
                if len(correct_logical_forms) <= max_num_logical_forms:
                    correct_logical_forms.append(logical_form)
            else:
                if len(incorrect_logical_forms) <= max_num_logical_forms:
                    incorrect_logical_forms.append(logical_form)
            if (
                len(correct_logical_forms) >= max_num_logical_forms
                and len(incorrect_logical_forms) >= max_num_logical_forms
            ):
                break
        num_examples += 1
        if correct_logical_forms:
            examples_w_correct_logical_forms += 1
            max_num_correct_logical_forms = max(
                max_num_correct_logical_forms, len(correct_logical_forms)
            )
            total_num_correct_logical_forms += len(correct_logical_forms)
        if prune_data and len(correct_logical_forms) == 0:
            continue
        if write_sequences:
            correct_sequences = [
                worlds[0].logical_form_to_action_sequence(logical_form)
                for logical_form in correct_logical_forms
            ]
            incorrect_sequences = [
                worlds[0].logical_form_to_action_sequence(logical_form)
                for logical_form in incorrect_logical_forms
            ]
            processed_data.append(
                {
                    "identifier": instance_id,
                    "sentence": sentence,
                    "correct_sequences": correct_sequences,
                    "incorrect_sequences": incorrect_sequences,
                    "worlds": structured_reps,
                    "labels": label_strings,
                }
            )
        else:
            processed_data.append(
                {
                    "identifier": instance_id,
                    "sentence": sentence,
                    "correct_logical_forms": correct_logical_forms,
                    "incorrect_logical_forms": incorrect_logical_forms,
                    "worlds": structured_reps,
                    "labels": label_strings,
                }
            )

    avg_num_correct_logical_forms = (
        float(total_num_correct_logical_forms) / examples_w_correct_logical_forms
    )
    print("Num of examples w/ correct logical forms: {}".format(examples_w_correct_logical_forms))
    print(
        "Max num of correct logical forms for an example: {}".format(max_num_correct_logical_forms)
    )
    print("Avg num of correct logical forms per example: {}".format(avg_num_correct_logical_forms))
    print("Writing {} instances to: {}".format(len(processed_data), output_file))
    with open(output_file, "w") as outfile:
        for instance_processed_data in processed_data:
            json.dump(instance_processed_data, outfile)
            outfile.write("\n")
        outfile.close()

    output_stats_file = os.path.splitext(output_file)[0] + "_stats.txt"
    print("Writing stats to: {}".format(output_stats_file))
    with open(output_stats_file, "w") as outfile:
        outfile.write("Input file: {}\n".format(input_file))
        outfile.write("Output file: {}\n".format(output_file))
        outfile.write(
            "Num of examples w/ correct logical forms: {}\n".format(
                examples_w_correct_logical_forms
            )
        )
        outfile.write(
            "Max num of correct logical forms for an example: {}\n".format(
                max_num_correct_logical_forms
            )
        )
        outfile.write(
            "Avg num of correct logical forms per example: {}\n".format(
                avg_num_correct_logical_forms
            )
        )
        outfile.write("Writing {} instances to: {}\n".format(len(processed_data), output_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="NLVR data file")
    parser.add_argument("output", type=str, help="Processed output")
    parser.add_argument(
        "--max-path-length",
        type=int,
        dest="max_path_length",
        help="Maximum path length for logical forms",
        default=10,
    )
    parser.add_argument(
        "--max-num-logical-forms",
        type=int,
        dest="max_num_logical_forms",
        help="Maximum number of logical forms per denotation, per question",
        default=20,
    )
    parser.add_argument(
        "--ignore-agenda",
        dest="ignore_agenda",
        help="Should we ignore the "
        "agenda and use consistency as the only signal to get logical forms?",
        action="store_true",
    )
    parser.add_argument(
        "--allow-partial",
        dest="allow_partial",
        help="Should we allow partial matches of a found path with the agenda?",
        action="store_true",
    )
    parser.add_argument(
        "--write-action-sequences",
        dest="write_sequences",
        help="If this "
        "flag is set, action sequences instead of logical forms will be written "
        "to the json file. This will avoid having to parse the logical forms again "
        "in the NlvrDatasetReader.",
        action="store_true",
    )
    parser.add_argument(
        "--prune-data",
        dest="prune_data",
        help="Should we only keep examples for which at least one correct logical-form is found?",
        action="store_true",
    )
    args = parser.parse_args()
    print(args)
    output_dir = os.path.split(args.output)[0]
    os.makedirs(output_dir, exist_ok=True)
    process_data(
        args.input,
        args.output,
        args.max_path_length,
        args.max_num_logical_forms,
        args.ignore_agenda,
        args.allow_partial,
        args.write_sequences,
        args.prune_data,
    )
