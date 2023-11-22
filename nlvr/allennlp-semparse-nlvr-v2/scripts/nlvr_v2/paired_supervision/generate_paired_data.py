#! /usr/bin/env python
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

COLORS = ["yellow", "black", "blue"]
SHAPES = ["circle", "triangle", "square"]
NUMBERS = [
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


class PairedPhrase:
    def __init__(
        self, abstract_phrases: List[str], grounded_phrases: List[str], sentences: List[str] = [],
        identifiers: List[str] = []
    ):
        self.abstract_phrases: List[str] = abstract_phrases
        self.grounded_phrases: Tuple[str] = tuple(grounded_phrases)
        self.sentences: List[str] = sentences
        self.identifiers: List[str] = identifiers
        self.cluster_id = None

    def __str__(self):
        output = "\n*********\n"
        output += "\n".join(self.abstract_phrases)
        output += "\n-----------\n"
        indices = list(range(len(self.sentences)))
        random.shuffle(indices)
        indices = indices[0:10]
        sents = [self.sentences[i] for i in indices]
        output += "\n".join(sents)
        output += "\n*********\n"
        return output


def convert_abstract_phrase_to_grounded(abstract_phrases: List[str]) -> List[List[str]]:
    """Convert a set of abstract phrases into multiple grounded sets each set containing a unique combination of
    abstract token to grounded token mapping.
    For example, input: ["COLOR1 at the base", "COLOR1 as the base"] would be converted to multiple sets, each one
    containing one value for the COLOR1 variable

    We currently limit to two colors, two shapes and one number in the phrase.
    """

    # This contains different equivalent (partially) grounded phrases.
    grounded_phrases_sets: List[List[str]] = [abstract_phrases]

    abstractions = ["COLOR1", "COLOR2", "SHAPE1", "SHAPE2", "NUMBER1"]
    for abstract_token in abstractions:
        # Go through each possible abstract token in order and expand `grounded_phrases` by considering all possible
        # groundings of this abstract token
        new_grounded_phrases_sets = []
        for equivalent_set in grounded_phrases_sets:
            # Mutate this set into multiple equivalent sets by replacing the abstract token with all its groundings.
            # For example, if this set is
            # ["yellow SHAPE1 at the base"], it should lead to three new sets ["yellow square at the base"],
            # ["yellow triangle at the base"], and ["yellow circle at the base"].
            if not all([abstract_token in x for x in equivalent_set]):
                # If phrases in this set does not contain the abstract token, mutations cannot be made
                # add this equivalent set to the final sets as it is
                new_grounded_phrases_sets.append(equivalent_set)
                continue
            if "COLOR" in abstract_token:
                options = COLORS
            elif "SHAPE" in abstract_token:
                options = SHAPES
            elif "NUMBER" in abstract_token:
                options = NUMBERS
            else:
                raise NotImplementedError
            # This equivalent set would be mutated into as many new sets as grounding options
            new_equivalent_sets = []
            for grounding_token in options:
                # Each phrase in the current equivalent set will be grounded with this token and added to the new set
                new_equivalent_set = []
                for phrase in equivalent_set:
                    new_phrase = phrase.replace(abstract_token, grounding_token)
                    new_equivalent_set.append(new_phrase)
                if grounding_token in number_strings:
                    alternate_token = number_strings[grounding_token]
                    for phrase in equivalent_set:
                        new_phrase = phrase.replace(abstract_token, alternate_token)
                        new_equivalent_set.append(new_phrase)
                if new_equivalent_set:
                    new_equivalent_sets.append(new_equivalent_set)
            # Add these new sets to the new collection
            new_grounded_phrases_sets.extend(new_equivalent_sets)
        grounded_phrases_sets = copy.deepcopy(new_grounded_phrases_sets)

    return grounded_phrases_sets


def get_contained_span(sentence: str, phrases: List[str]) -> Tuple[int, int]:
    """Get the position of the longest contained span in the sentence from a span in phrases."""
    # Sorting phrases in decreasing order of length; to select "yellow squares" over "yellow square"
    sorted_phrases = sorted(phrases, key=lambda x: len(x), reverse=True)
    char_offsets = [-1, -1]
    for phrase in sorted_phrases:
        start_position = sentence.find(phrase)
        if start_position == -1:
            # not found
            continue
        char_offsets = [start_position, start_position + len(phrase)]
    return char_offsets


def sample_paired_instance(identifier: str, paired_identifiers: List[str], paired_phrases: List[str],
                           id2instance: Dict[str, NlvrInstance], max_samples_per_phrase: int):
    """Sample `max_samples_per_phrase` paired instances for a given instance from a list of instances.

    The instances `identifier` and `paired_identifiers` are known to all share a paired-phrase from `paired_phrases`.
    The exact phrase might be different; for example identifier might have `yellow block at the base` and an instance
    could have `the base is yellow`.
    """

    tries = 100
    trynum = 0
    instance = id2instance[identifier]
    sampled_identifiers = []
    all_orig_char_offsets: List[Tuple[int, int]] = []
    all_paired_char_offsets: List[Tuple[int, int]] = []
    # paired_identifier = None
    # orig_char_offsets = None
    # paired_char_offsets = None
    while len(sampled_identifiers) < max_samples_per_phrase and trynum < tries:
        trynum += 1
        sampled_id = random.choice(paired_identifiers)
        sampled_instance = id2instance[sampled_id]
        # Check not the same sentence or already sampled
        if sampled_instance.sentence != instance.sentence and sampled_id not in sampled_identifiers:
            # Get paired phrase char-offsets in original and sampled instance
            orig_char_offsets = get_contained_span(instance.sentence, paired_phrases)
            paired_char_offsets = get_contained_span(sampled_instance.sentence, paired_phrases)
            if orig_char_offsets != [-1, -1] and paired_char_offsets != [-1, -1]:
                sampled_identifiers.append(sampled_id)
                all_orig_char_offsets.append(orig_char_offsets)
                all_paired_char_offsets.append(paired_char_offsets)

    return sampled_identifiers, all_orig_char_offsets, all_paired_char_offsets


def sample_nt_paired_instance(identifier: str, pairedphrase: PairedPhrase,
                              cluster2paired_phrases: Dict[int, List[PairedPhrase]],
                              id2instance: Dict[str, NlvrInstance],
                              max_samples_per_phrase=1):
    """Sample paired instance for the given instance which matches in an abstract manner.

    The instances `identifier` and `paired_identifiers` are known to all share a paired-phrase from `paired_phrases`.
    The exact phrase might be different; for example identifier might have `yellow block at the base` and an instance
    could have `the base is yellow`.
    """

    tries = 100
    trynum = 0
    instance = id2instance[identifier]
    sampled_identifiers = []
    all_orig_char_offsets: List[Tuple[int, int]] = []
    all_paired_char_offsets: List[Tuple[int, int]] = []
    cluster_id = pairedphrase.cluster_id
    # Sample another PairedPhrase with the same cluster, then sample an instance from that phrase
    potential_phrases: List[PairedPhrase] = cluster2paired_phrases[cluster_id]
    while len(sampled_identifiers) < max_samples_per_phrase and trynum < tries:
        trynum += 1
        sampled_phrase = random.choice(potential_phrases)
        if sampled_phrase.grounded_phrases == pairedphrase.grounded_phrases:
            continue

        potential_identifiers = sampled_phrase.identifiers
        sampled_identifer = random.choice(potential_identifiers)
        sampled_instance = id2instance[sampled_identifer]
        if sampled_instance.sentence == instance.sentence:
            continue
        orig_char_offsets = get_contained_span(instance.sentence, list(pairedphrase.grounded_phrases))
        paired_char_offsets = get_contained_span(sampled_instance.sentence, list(sampled_phrase.grounded_phrases))
        if orig_char_offsets != [-1, -1] and paired_char_offsets != [-1, -1]:
            sampled_identifiers.append(sampled_identifer)
            all_orig_char_offsets.append(orig_char_offsets)
            all_paired_char_offsets.append(paired_char_offsets)

    return sampled_identifiers, all_orig_char_offsets, all_paired_char_offsets


def make_data(
    data_jsonl: str,
    paired_phrases_json: str,
    output_jsonl: str,
    max_samples_per_phrase: int = 1,
    max_samples_per_instance: int = 1,
    add_nonterminal_matches: bool = False,
    max_nt_samples_per_instance: int = 1,
) -> None:
    """Discover and add paired examples for NLVR instances.
    Given sets of abstract paired phrases, first create sets of equivalent grounded phrases and identify instances that
    have these phrases.

    Using these sets, sample instance pairings that contain the same grounded phrase.
    """

    nlvr_instances: List[NlvrInstance] = read_nlvr_data(data_jsonl)
    id2instance: Dict[str, NlvrInstance] = {instance.identifier: instance for instance in nlvr_instances}

    print("Reading paired data ... ")
    # Each inner list is a set of equivalent abstract phrases; e.g. ["COLOR1 as the base", "the base is COLOR1"]
    with open(paired_phrases_json) as f:
        paired_phrases_list: List[List[str]] = json.load(f)

    num_abstract_phrases, num_grounded_phrases = 0, 0
    # Grounded phrase(s) that are equivalent
    paired_phrases: List[PairedPhrase] = []
    # Each abstract set is converted to multiple grounded sets by considering all possible values of the abstract tokens
    for equivalent_abstract_set in paired_phrases_list:
        # One set of equivalent abstract phrases is converted to many equivalent sets after grounding
        equivalent_grounded_sets: List[List[str]] = convert_abstract_phrase_to_grounded(
            equivalent_abstract_set
        )
        for grounded_set in equivalent_grounded_sets:
            # Instances that contain any of the equivalent phrases are paired
            paired_nlvr_sentences: List[str] = []
            paired_identifiers: List[str] = []
            for instance in nlvr_instances:
                if any([x in instance.sentence for x in grounded_set]) and \
                        instance.identifier not in paired_identifiers:
                    paired_nlvr_sentences.append(instance.sentence)
                    paired_identifiers.append(instance.identifier)
            if len(paired_nlvr_sentences) < 2:
                # No pairing without atleast 2 examples
                continue
            # Keep this paired phrase only if there are instances containing it
            paired_phrase = PairedPhrase(
                abstract_phrases=equivalent_abstract_set,
                grounded_phrases=grounded_set,
                sentences=paired_nlvr_sentences,
                identifiers=paired_identifiers,
            )
            paired_phrases.append(paired_phrase)
            num_abstract_phrases += 1
            num_grounded_phrases += len(grounded_set)

    avg_grounded_phrases_per_set = float(num_grounded_phrases) / num_abstract_phrases
    print(
        "Paired phrases generated. Num of equivalent grounded phrases: {}  "
        "Avg num of grounded phrases per set: {}".format(
            num_abstract_phrases, avg_grounded_phrases_per_set
        )
    )

    # Each PairedPhrase is grounded phrase; to add non-terminal matches, i.e., matches between "yellow on a square" w/
    # "blue on a triangle", we need to identify clusters of PairedPhrase that come from the same abstract phrase set.
    cluster2pairedphrases: Dict[int, List[PairedPhrase]] = {}
    abstractphrases2clusterid: Dict[Tuple, int] = {}
    for paired_phrase in paired_phrases:
        abstract_phrases = tuple(paired_phrase.abstract_phrases)
        if abstract_phrases not in abstractphrases2clusterid:
            abstractphrases2clusterid[abstract_phrases] = len(abstractphrases2clusterid)
        cluster_id = abstractphrases2clusterid[abstract_phrases]
        paired_phrase.cluster_id = cluster_id
        if cluster_id not in cluster2pairedphrases:
            cluster2pairedphrases[cluster_id] = []
        cluster2pairedphrases[cluster_id].append(paired_phrase)

    # For each paired-phrase (grounded), sample, for instance sample `max_samples_per_phrase` paired instances
    num_pairings_found = 0
    num_nt_pairings_found = 0
    for paired_phrase in paired_phrases:
        # Pair each instance containing this paired_phrase to `max_samples_per_phrase` other instances containing it
        paired_identifiers: List[str] = paired_phrase.identifiers
        for identifier in paired_phrase.identifiers:
            instance: NlvrInstance = id2instance[identifier]
            # For this instance, sample paired instances from paired_identifiers
            # paired_ids: List[str], all_orig_charoffsets & all_orig_charoffsets: List[Tuple[int, int]]
            paired_ids, all_orig_charoffsets, all_paired_charoffsets = sample_paired_instance(
                identifier, paired_identifiers, paired_phrase.grounded_phrases, id2instance, max_samples_per_phrase)
            if paired_ids:
                if instance.paired_examples is None:
                    instance.paired_examples = []
                for paired_id, orig_charoffsets, paired_charoffsets in zip(paired_ids,
                                                                           all_orig_charoffsets,
                                                                           all_paired_charoffsets):
                    # Found a paired instance; add it
                    paired_instance: NlvrInstance = id2instance[paired_id]
                    paired_example = {
                        "identifier": paired_id,
                        "sentence": paired_instance.sentence,
                        "structured_representations": paired_instance.structured_representations,
                        "labels": paired_instance.labels,
                        "orig_charoffsets": orig_charoffsets,
                        "paired_charoffsets": paired_charoffsets,
                        "nt_match": False,
                    }
                    instance.paired_examples.append(paired_example)
                    num_pairings_found += 1

            if not add_nonterminal_matches:
                continue
            nt_paired_ids, all_nt_orig_charoffsets, all_nt_paired_charoffsets = sample_nt_paired_instance(
                identifier=identifier, pairedphrase=paired_phrase, cluster2paired_phrases=cluster2pairedphrases,
                id2instance=id2instance)
            if nt_paired_ids:
                if instance.paired_examples is None:
                    instance.paired_examples = []
                for nt_paired_id, nt_orig_charoffsets, nt_paired_charoffsets in zip(nt_paired_ids,
                                                                                    all_nt_orig_charoffsets,
                                                                                    all_nt_paired_charoffsets):
                    # Found a paired instance; add it
                    paired_instance: NlvrInstance = id2instance[nt_paired_id]
                    paired_example = {
                        "identifier": nt_paired_id,
                        "sentence": paired_instance.sentence,
                        "structured_representations": paired_instance.structured_representations,
                        "labels": paired_instance.labels,
                        "orig_charoffsets": nt_orig_charoffsets,
                        "paired_charoffsets": nt_paired_charoffsets,
                        "nt_match": True,
                    }
                    instance.paired_examples.append(paired_example)
                    num_nt_pairings_found += 1

    print("Num of pairings found: {}".format(num_pairings_found))
    print("Num of NT pairings found: {}".format(num_nt_pairings_found))
    # Keep only `max_samples_per_instance` paired instances per example
    print("Pruning to have maximum {} paired examples per instance ...".format(max_samples_per_instance))
    print("Pruning to have maximum {} paired NT examples per instance ...".format(max_nt_samples_per_instance))
    num_pairing_made = 0
    instance_w_pairs = 0
    instance_w_full_matches = 0
    instance_w_nt_matches = 0
    for identifier, instance in id2instance.items():
        if instance.paired_examples:
            instance_w_pairs += 1
            full_matches = [x for x in instance.paired_examples if x.get("nt_match") is False]
            nt_matches = [x for x in instance.paired_examples if x.get("nt_match") is True]
            if full_matches:
                instance_w_full_matches += 1
            random.shuffle(full_matches)
            instance.paired_examples = full_matches[:max_samples_per_instance]

            if nt_matches:
                random.shuffle(nt_matches)
                if max_nt_samples_per_instance >= 1:
                    instance.paired_examples.extend(nt_matches[:int(max_nt_samples_per_instance)])
                    instance_w_nt_matches += 1
                elif random.random() <= max_nt_samples_per_instance:
                    instance.paired_examples.extend(nt_matches[:1])
                    instance_w_nt_matches += 1
            num_pairing_made += len(instance.paired_examples)

    print("Number of instances with pairings: {}".format(instance_w_pairs))
    print("Num instances w/ full matches: {} Num w/ nt matches: {}".format(instance_w_full_matches,
                                                                           instance_w_nt_matches))
    print("Number of pairings made: {}".format(num_pairing_made))

    final_instances = [instance for _, instance in id2instance.items()]
    print_dataset_stats(final_instances)
    write_nlvr_data(final_instances, output_jsonl)

    output_stats_file = os.path.splitext(output_jsonl)[0] + "_stats.txt"
    print("Writing stats to: {}".format(output_stats_file))
    with open(output_stats_file, "w") as outfile:
        outfile.write("max_samples_per_phrase: {}  ".format(max_samples_per_phrase))
        outfile.write("max_samples_per_instance: {}\n".format(max_samples_per_instance))
        outfile.write("add-nt-samples: {}\n".format(add_nonterminal_matches))
        outfile.write("Num of pairings found: {}\n".format(num_pairings_found))
        outfile.write("Num of NT pairings found: {}\n".format(num_nt_pairings_found))
        # Keep only `max_samples_per_instance` paired instances per example
        outfile.write("Pruning to have maximum {} paired examples per instance ...\n".format(max_samples_per_instance))
        outfile.write("Pruning for maximum {} paired NT examples per instance ...".format(max_nt_samples_per_instance))
        outfile.write("Number of instances with pairings: {}\n".format(instance_w_pairs))
        outfile.write("Number of pairings made: {}\n".format(num_pairing_made))
        outfile.write("Num instances w/ full matches: {} Num w/ nt matches: {}".format(instance_w_full_matches,
                                                                                       instance_w_nt_matches))
        outfile.write("Output json: {}".format(output_jsonl))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_jsonl", type=str, help="Input data file")
    parser.add_argument(
        "paired_phrases_json",
        type=str,
        help="Input file containing paired phrases",
    )
    parser.add_argument(
        "output_jsonl",
        type=str,
        help="Path to archived model.tar.gz to use for decoding",
    )
    parser.add_argument(
        '--max_samples_per_phrase',
        type=int,
        default=1,
        help="For each grounded phrase, sample these many paired instances per instance"
    )
    parser.add_argument(
        '--max_samples_per_instance',
        type=int,
        default=1,
        help="Max number of paired instance per example"
    )
    parser.add_argument(
        "--add-nt-matches",
        dest="add_nonterminal_matches",
        help="If this flag is set, pairings for abstract phrases will also be added",
        action="store_true",
    )
    parser.add_argument(
        '--max_nt_samples_per_instance',
        type=float,
        default=1,
        help="Number of non-terminal matched paired examples per instance. "
             "If <1, an example would be sampled with this probability."
    )

    args = parser.parse_args()
    make_data(args.data_jsonl, args.paired_phrases_json, args.output_jsonl,
              max_samples_per_phrase=args.max_samples_per_phrase,
              max_samples_per_instance=args.max_samples_per_instance,
              add_nonterminal_matches=args.add_nonterminal_matches,
              max_nt_samples_per_instance=args.max_nt_samples_per_instance)
