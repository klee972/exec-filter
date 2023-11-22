from typing import List, Dict, Tuple
import sys
import os
import json
import copy
import argparse
import random

random.seed(21)

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
    ),
)

from scripts.nlvr_v2.data.nlvr_instance import NlvrInstance, read_nlvr_data, write_nlvr_data, print_dataset_stats


BOXES = ["box", "boxes", "tower", "towers", "gray square", "gray squares"]
BLOCKS = ["item", "items", "block", "blocks", "object", "objects"]
COLORS = ["yellow", "black", "blue"]
SHAPES = ["circle", "triangle", "square"]
SIZES = ["small", "medium", "large"]

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


def get_structure(question: str):
    structure: str = copy.deepcopy(question)
    structure = structure.lower()

    # replacing boxes first, since "gray square" => BOX, otherwise, square would convert to a SHAPE
    for box in BOXES:
        structure = structure.replace(box, "BOX")
    for block in BLOCKS:
        structure = structure.replace(block, "BLOCK")
    for color in COLORS:
        structure = structure.replace(color, "COLOR")
    for shape in SHAPES:
        structure = structure.replace(shape, "SHAPE")
    for number in number_strings.values():
        structure = structure.replace(number, "NUMBER")
    for number in number_strings.keys():
        structure = structure.replace(number, "NUMBER")
    for size in SIZES:
        structure = structure.replace(size, "SIZE")

    structure = structure.replace("BOXs", "BOX")
    structure = structure.replace("BOXes", "BOX")
    structure = structure.replace("BLOCKs", "BLOCK")
    structure = structure.replace("COLORs", "COLOR")
    structure = structure.replace("SHAPEs", "SHAPE")
    structure = structure.replace("NUMBERs", "NUMBER")

    structure = structure.replace(" are ", " IS ")
    structure = structure.replace(" is ", " IS ")

    return structure


def sample_dev_and_test_structures(structure2count, dev_target, test_target):
    dev_structures = []
    test_structures = []

    remaining_structures = list(structure2count.keys())
    random.shuffle(remaining_structures)

    print("Starting to sample dev/test structures. Aiming for dev:{} and test:{} instances".format(dev_target,
                                                                                                   test_target))

    dev_count, test_count = 0, 0
    while dev_count < dev_target or test_count < test_target:
        if dev_count < dev_target:
            sample_struct = random.choice(remaining_structures)
            dev_structures.append(sample_struct)
            dev_count += structure2count[sample_struct]
            remaining_structures.remove(sample_struct)

        if test_count < test_target:
            sample_struct = random.choice(remaining_structures)
            test_structures.append(sample_struct)
            test_count += structure2count[sample_struct]
            remaining_structures.remove(sample_struct)

    return dev_count, dev_structures, test_count, test_structures


def get_compgen_split(all_data_jsonl, output_dir):
    all_instances: List[NlvrInstance] = read_nlvr_data(all_data_jsonl)
    print_dataset_stats(all_instances)

    identifier2instance = {}
    structure2identifiers = {}
    structure2count = {}

    for instance in all_instances:
        identifier2instance[instance.identifier] = instance
        structure = get_structure(instance.sentence)
        if structure not in structure2identifiers:
            structure2identifiers[structure] = []
            structure2count[structure] = 0
        structure2identifiers[structure].append(instance.identifier)
        structure2count[structure] += 1

    num_structures = len(structure2identifiers)
    print("Number of abstract structures : {}".format(num_structures))

    sorted_struc2count = sorted(structure2count.items(), key=lambda x: x[1], reverse=True)
    structure2count_tuples = copy.deepcopy(sorted_struc2count)

    # Keeping top-20 structures for training definitely
    train_structures = [s for (s, c) in structure2count_tuples[0:20]]
    structure2count_tuples = structure2count_tuples[20:]

    total_num_ques = len(all_instances)
    remaining_structures = [s for (s, _) in structure2count_tuples]
    # Sample test structures so that the # of utterances reaches a predefined target
    test_target = int(0.08 * total_num_ques)    # 8% of total data is # test examples
    test_structures = []
    num_test_instances = 0
    test_instance_ids = []
    print("Sampling test-structs from {} structures, aiming for {} instances".format(len(remaining_structures),
                                                                                     test_target))
    while num_test_instances < test_target:
        sample_struct = random.choice(remaining_structures)
        test_structures.append(sample_struct)
        test_instance_ids.extend(structure2identifiers[sample_struct])
        num_test_instances += structure2count[sample_struct]
        remaining_structures.remove(sample_struct)

    print("Sampled {} test-structures totalling {} instances".format(len(test_structures),
                                                                     len(test_instance_ids)))

    remaining_instance_ids = [instance.identifier for instance in all_instances
                              if instance.identifier not in test_instance_ids]
    dev_target = int(0.08 * total_num_ques)  # 8% of total data is # test examples
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("all_data_jsonl", type=str, help="Input data file")
    parser.add_argument("output_dir", type=str, help="Output dir to write train.json, dev.json & test.json")

    args = parser.parse_args()
    get_compgen_split(all_data_jsonl=args.all_data_jsonl, output_dir=args.output_dir)