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


def get_ngrams(tokens, n) -> List:
    output = []
    for i in range(len(tokens) - n + 1):
        output.append(tokens[i : i + n])
    return output


def abstract_color_shape_number(sentence):
    sentence = sentence.replace("blue", "COLOR")
    sentence = sentence.replace("black", "COLOR")
    sentence = sentence.replace("yellow", "COLOR")

    sentence = sentence.replace("triangles", "SHAPE")
    sentence = sentence.replace("squares", "SHAPE")
    sentence = sentence.replace("circles", "SHAPE")

    sentence = sentence.replace("triangle", "SHAPE")
    sentence = sentence.replace("square", "SHAPE")
    sentence = sentence.replace("circle", "SHAPE")

    sentence = sentence.replace("blocks", "OBJECT")
    sentence = sentence.replace("objects", "OBJECT")

    sentence = sentence.replace("block", "OBJECT")
    sentence = sentence.replace("object", "OBJECT")

    numbers = [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
    ]
    for num in numbers:
        sentence = sentence.replace(num, "NUMBER")

    return sentence


def ngram_stats(
    input_file: str,
    output_file: str,
    min_ngram_size: int,
    max_ngram_size: int,
) -> None:
    """
    Reads an NLVR dataset and compute stats for shared-ngrams across utterances.
    """
    ngram_dict = {}
    for line in open(input_file):
        instance_id, sentence, structured_reps, label_strings = read_json_line(line)

        sentence = sentence.replace(".", " .")
        sentence = sentence.replace("?", " ?")
        sentence = sentence.replace(",", " ,")

        sentence = abstract_color_shape_number(sentence)

        tokens = sentence.split(" ")
        for n in range(min_ngram_size, max_ngram_size + 1):
            ngrams = get_ngrams(tokens, n)
            ngrams = [" ".join(x) for x in ngrams]  # converting n-gram
            for ngram in ngrams:
                if ngram not in ngram_dict:
                    ngram_dict[ngram] = set()
                ngram_dict[ngram].add(sentence)

    # Sorting dict by number of common-sentences
    ngram_dict = {
        k: v for k, v in sorted(ngram_dict.items(), key=lambda x: len(x[1]), reverse=True)
    }
    ngram_dict = {k: v for k, v in ngram_dict.items() if len(v) >= 5}
    print("Number of ngrams: {}".format(len(ngram_dict)))

    with open(output_file, "w") as outfile:
        outfile.write("N-Gram\tCount\tUtterances\n")
        for ngram, common_sentences in ngram_dict.items():
            count = len(common_sentences)
            common_sentences = list(common_sentences)[0:5]
            outfile.write("{}\t{}\t{}\n".format(ngram, count, common_sentences))
        outfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="NLVR data file")
    parser.add_argument("output", type=str, help="Output tsv")
    parser.add_argument(
        "--min-ngram-size",
        type=int,
        dest="min_ngram_size",
        help="Minimum length of n-gram to consider for stats",
        default=3,
    )
    parser.add_argument(
        "--max-ngram-size",
        type=int,
        dest="max_ngram_size",
        help="Maximum length of n-gram to consider for stats",
        default=3,
    )
    args = parser.parse_args()
    print(args)
    ngram_stats(args.input, args.output, args.min_ngram_size, args.max_ngram_size)
