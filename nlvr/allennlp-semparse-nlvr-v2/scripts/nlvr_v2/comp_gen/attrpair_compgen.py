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

number_strings = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

BOXES = ["box", "boxes", "tower", "towers", "gray square", "gray squares"]
BLOCKS = ["item", "items", "block", "blocks", "object", "objects"]
COLORS = ["yellow", "black", "blue"]
SHAPES = ["circle", "triangle", "square"]
SIZES = ["small", "medium", "large"]
NUMBERS = [
    "one",
    "two",
    "three",
    "four",
]
# list(number_strings.keys())


def read_paired_phrases(paired_phrases_json):
    print("Reading paired data ... ")
    # Each inner list is a set of equivalent abstract phrases; e.g. ["COLOR1 as the base", "the base is COLOR1"]
    with open(paired_phrases_json) as f:
        paired_phrases_list: List[List[str]] = json.load(f)
    return paired_phrases_list


def get_grounded_test_phrases(paired_phrases_list: List[List[str]]) -> List[List[str]]:
    """Given a list of set of abstract-paired-phrases, sample grounded properties and make test phrases.

    For e.g. for an abstract set ["NUMBER1 different color", "NUMBER1 different colors", "NUMBER1 color"],
    sample NUMBER1=2, and make all grounded phrases "2 different color", "2 different colors", "2 color", "2 colors"
    as test phrases. Any instance containing any of these phrases would be a test instance.
    """

    abstractions = ["COLOR1", "COLOR2", "SHAPE1", "SHAPE2", "NUMBER1"]

    grounded_test_phrases_sets: List[List[str]] = []

    for equivalent_set in paired_phrases_list:
        # For this abstract equivalent set, these are grounded values we will use
        grounded_values = []
        for abstract_token in abstractions:
            if "COLOR" in abstract_token:
                options = COLORS
            elif "SHAPE" in abstract_token:
                options = SHAPES
            elif "NUMBER" in abstract_token:
                options = NUMBERS
            grounded_token = random.choice(options)
            grounded_values.append(grounded_token)

        # At every step of replacing abstract-token with grounded-value, this is the set that will hold these partially
        # grounded phrases
        partially_grounded_equivalent_set = copy.deepcopy(equivalent_set)
        for i, abstract_token in enumerate(abstractions):
            new_grounded_set = []
            for phrase in partially_grounded_equivalent_set:
                if abstract_token in phrase:
                    new_grounded_set.append(phrase.replace(abstract_token, grounded_values[i]))
                    if grounded_values[i] in number_strings:
                        new_grounded_set.append(phrase.replace(abstract_token, number_strings[grounded_values[i]]))
                else:
                    # This abstract token does not appear in this phrase, add the phrase as it is
                    new_grounded_set.append(phrase)
            partially_grounded_equivalent_set = copy.deepcopy(new_grounded_set)

        if set(equivalent_set) == set(partially_grounded_equivalent_set):
            # This set did not contain any abstractions, don;t add it to test phrases
            continue
        grounded_test_phrases_sets.append(partially_grounded_equivalent_set)

    return grounded_test_phrases_sets


def get_compgen_split(all_data_jsonl, paired_phrases_json, output_dir):
    all_instances: List[NlvrInstance] = read_nlvr_data(all_data_jsonl)
    print_dataset_stats(all_instances)
    total_num_ques = len(all_instances)
    identifier2instance = {instance.identifier: instance for instance in all_instances}

    paired_phrases_list = read_paired_phrases(paired_phrases_json)
    grounded_test_phrase_sets: List[List[str]] = get_grounded_test_phrases(paired_phrases_list)
    for phrase_set in grounded_test_phrase_sets:
        print(phrase_set)
    print()

    all_test_phrases = [x for y in grounded_test_phrase_sets for x in y]

    test_instance_ids = []
    for instance in all_instances:
        if any(x in instance.sentence for x in all_test_phrases):
            test_instance_ids.append(instance.identifier)
    test_instance_ids = list(set(test_instance_ids))

    print("Number of test identifiers: {}".format(len(test_instance_ids)))

    remaining_instance_ids = [instance.identifier for instance in all_instances
                              if instance.identifier not in test_instance_ids]
    dev_target = int(0.07 * total_num_ques)  # 7% of total data is # test examples
    print("Sampling {} dev-instances from {} remaining instances".format(dev_target, len(remaining_instance_ids)))
    dev_instance_ids = random.sample(remaining_instance_ids, dev_target)
    train_instance_ids = [i for i in remaining_instance_ids if i not in dev_instance_ids]

    print("Train: {}  Dev: {}  Test: {}".format(len(train_instance_ids),
                                                len(dev_instance_ids),
                                                len(test_instance_ids)))

    train_instances = [identifier2instance[i] for i in train_instance_ids]
    dev_instances = [identifier2instance[i] for i in dev_instance_ids]
    test_instances = [identifier2instance[i] for i in test_instance_ids]

    os.makedirs(output_dir, exist_ok=True)
    write_nlvr_data(train_instances, os.path.join(output_dir, "train.json"))
    write_nlvr_data(dev_instances, os.path.join(output_dir, "dev.json"))
    write_nlvr_data(test_instances, os.path.join(output_dir, "test.json"))


    #
    # truncated_sorted_struc2count = [(x,y) for (x,y) in sorted_struc2count if y > 2]
    # print(truncated_sorted_struc2count)
    # print(len(truncated_sorted_struc2count))
    # print()
    #
    # mono_sorted_struc2count = [(x, y) for (x, y) in sorted_struc2count if y == 1]
    # print(mono_sorted_struc2count)
    # print(len(mono_sorted_struc2count))
    #
    # multstructures = [structure for structure, count in structure2count.items() if count > 2]
    # # print(multstructures)
    # print(len(multstructures))
    #



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("all_data_jsonl", type=str, help="Input data file")
    parser.add_argument("paired_phrases_json", type=str, help="paired_phrases_json")
    parser.add_argument("output_dir", type=str, help="Output dir to write train.json, dev.json & test.json")

    args = parser.parse_args()
    get_compgen_split(all_data_jsonl=args.all_data_jsonl,
                      paired_phrases_json=args.paired_phrases_json,
                      output_dir=args.output_dir)